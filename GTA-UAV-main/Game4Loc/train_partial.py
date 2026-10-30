import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import average_precision_score
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from game4loc.dataset.gta import GTADatasetEval, GTADatasetTrain, get_transforms
from game4loc.evaluate.gta import get_dis, sdm
from game4loc.utils import Logger, setup_system


class LocalRetrievalModel(nn.Module):
    def __init__(self, model_name, img_size, method, local_size=4, local_dim=256,
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        self.method = method
        self.local_size = local_size
        self.local_sizes = (4, 6) if method == "E6_MS" else (local_size,)
        embed_dim = self.backbone.num_features
        self.local_projection = nn.Linear(embed_dim, local_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.dustbin_score = nn.Parameter(torch.zeros([]))

    def forward(self, image):
        tokens = self.backbone.forward_features(image)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected ViT token features [B,N,C], got {tuple(tokens.shape)}")
        global_feature = self.backbone.forward_head(tokens)
        prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        patch_tokens = tokens[:, prefix:]
        grid_h, grid_w = self.backbone.patch_embed.grid_size
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            image.shape[0], self.backbone.num_features, grid_h, grid_w
        )
        local_features = []
        for size in self.local_sizes:
            pooled = F.adaptive_avg_pool2d(patch_tokens, (size, size))
            pooled = pooled.flatten(2).transpose(1, 2)
            local_features.append(F.normalize(self.local_projection(pooled), dim=-1))
        local_features = torch.cat(local_features, dim=1)
        return F.normalize(global_feature, dim=-1), local_features

    def pair_local_score(self, local_a, local_b):
        if self.method == "E6_MS":
            # Equal weight per scale, so adding descriptors does not double Sl.
            scores, offset = [], 0
            for size in self.local_sizes:
                end = offset + size * size
                sim = torch.einsum("bmd,cnd->bcmn", local_a[:, offset:end], local_b[:, offset:end])
                scores.append(self._dustbin_transport_score(sim.flatten(0, 1)).reshape(sim.shape[:2]))
                offset = end
            return torch.stack(scores).mean(0)
        sim = torch.einsum("bmd,cnd->bcmn", local_a, local_b)
        if self.method == "E4":
            return (sim.max(dim=-1).values.mean(-1) + sim.max(dim=-2).values.mean(-1)) * 0.5
        if self.method == "E5":
            pair_shape = sim.shape[:2]
            flat_sim = sim.flatten(0, 1)
            transport = self._sinkhorn(flat_sim / 0.07)
            return (transport * flat_sim).sum(dim=(-1, -2)).reshape(pair_shape)
        pair_shape = sim.shape[:2]
        return self._dustbin_transport_score(
            sim.flatten(0, 1), evidence=self.method == "E6_EVIDENCE"
        ).reshape(pair_shape)

    @staticmethod
    def _sinkhorn(logits, iterations=7):
        batch, rows, cols = logits.shape
        log_mu = logits.new_full((batch, rows), -math.log(rows))
        log_nu = logits.new_full((batch, cols), -math.log(cols))
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iterations):
            u = log_mu - torch.logsumexp(logits + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(logits + u.unsqueeze(2), dim=1)
        return torch.exp(logits + u.unsqueeze(2) + v.unsqueeze(1))

    def _dustbin_transport(self, sim, iterations=8):
        batch, count, columns = sim.shape
        if count != columns:
            raise ValueError(f"Dustbin transport requires square local grids, got {count}x{columns}")
        scores = sim.new_zeros((batch, count + 1, count + 1))
        scores[:, :count, :count] = sim / 0.07
        scores[:, count, :] = self.dustbin_score
        scores[:, :, count] = self.dustbin_score
        total = 2.0
        log_mu = sim.new_full((batch, count + 1), -float("inf"))
        log_nu = sim.new_full((batch, count + 1), -float("inf"))
        log_mu[:, :count] = -math.log(count * total)
        log_nu[:, :count] = -math.log(count * total)
        log_mu[:, count] = math.log(0.5)
        log_nu[:, count] = math.log(0.5)
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return torch.exp(scores + u.unsqueeze(2) + v.unsqueeze(1))

    def _dustbin_transport_score(self, sim, iterations=8, evidence=False):
        transport = self._dustbin_transport(sim, iterations)[:, :-1, :-1]
        matched_mass = transport.sum(dim=(-1, -2)).clamp_min(1e-6)
        score = (transport * sim).sum(dim=(-1, -2)) / matched_mass
        if evidence:
            # Conditional correspondence ambiguity, weighted by transported mass.
            # Saturating support discounts tiny accidental matches without
            # requiring a large overlap or supervising overlap with dataset IoU.
            p = transport.float()
            mass = p.sum((-1, -2)).clamp_min(1e-8)
            row_mass, col_mass = p.sum(-1), p.sum(-2)
            row_p = p / row_mass.unsqueeze(-1).clamp_min(1e-8)
            col_p = p / col_mass.unsqueeze(-2).clamp_min(1e-8)
            row_h = -(row_p * row_p.clamp_min(1e-8).log()).sum(-1)
            col_h = -(col_p * col_p.clamp_min(1e-8).log()).sum(-2)
            uncertainty = 0.5 * (
                (row_h * row_mass).sum(-1) / (mass * math.log(sim.shape[-1]))
                + (col_h * col_mass).sum(-1) / (mass * math.log(sim.shape[-2]))
            )
            specificity = (1.0 - uncertainty).clamp(0.0, 1.0)
            support = (2.0 * mass).clamp(0.0, 1.0)
            reliability = ((support / (support + 0.1)) * specificity).clamp_min(1e-8).sqrt()
            score = score * reliability
        return score


def known_crop_views(images, rng, local_size=4):
    """Two equal-area windows with exact soft correspondence/dustbin targets.

    Equal crop areas make the geometric targets compatible with equal real
    OT marginals. Neither view must contain the other. Coordinates describe
    synthetic transforms, never dataset drone/satellite overlap annotations.
    """
    count = local_size ** 2
    views_a, views_b, targets = [], [], []
    device = images.device
    edge = torch.arange(local_size, device=device, dtype=torch.float32) / local_size
    yy, xx = torch.meshgrid(edge, edge, indexing="ij")
    x0, y0 = xx.flatten(), yy.flatten()
    x1, y1 = x0 + 1 / local_size, y0 + 1 / local_size
    for image in images:
        h, w = image.shape[-2:]
        fraction = rng.choice((0.5, 0.75))
        ch, cw = round(h * fraction), round(w * fraction)
        boxes, orders, views = [], [], []
        for _ in range(2):
            top, left = rng.randint(0, h - ch), rng.randint(0, w - cw)
            turns = rng.randrange(4)
            crop = F.interpolate(image[None, :, top:top + ch, left:left + cw],
                                 size=(h, w), mode="bilinear", align_corners=False)[0]
            views.append(torch.rot90(crop, turns, (-2, -1)))
            bx0, by0 = left / w + x0 * cw / w, top / h + y0 * ch / h
            boxes.append((bx0, by0, bx0 + cw / (w * local_size), by0 + ch / (h * local_size)))
            orders.append(torch.rot90(torch.arange(count, device=device).reshape(local_size, local_size), turns, (0, 1)).flatten())
        ax0, ay0, ax1, ay1 = boxes[0]
        bx0, by0, bx1, by1 = boxes[1]
        width = (torch.minimum(ax1[:, None], bx1[None]) - torch.maximum(ax0[:, None], bx0[None])).clamp_min(0)
        height = (torch.minimum(ay1[:, None], by1[None]) - torch.maximum(ay0[:, None], by0[None])).clamp_min(0)
        area = width * height
        area = area[orders[0]][:, orders[1]]
        cell_area = (ch / h) * (cw / w) / count
        target = area.new_zeros((count + 1, count + 1))
        target[:count, :count] = area
        target[:count, count] = (cell_area - area.sum(-1)).clamp_min(0)
        target[count, :count] = (cell_area - area.sum(-2)).clamp_min(0)
        views_a.append(views[0])
        views_b.append(views[1])
        targets.append(target)
    return torch.stack(views_a), torch.stack(views_b), torch.stack(targets)


def local_correspondence_loss(model, original_local, cropped_local, target):
    sim = torch.einsum("bmd,bnd->bmn", original_local, cropped_local)
    # FP32 protects the probability NLL from half precision underflow.
    p = model._dustbin_transport(sim.float())
    rows, cols = p.shape[-2] - 1, p.shape[-1] - 1
    row_target = target[:, :rows] / target[:, :rows].sum(-1, keepdim=True).clamp_min(1e-8)
    col_target = target[:, :, :cols] / target[:, :, :cols].sum(-2, keepdim=True).clamp_min(1e-8)
    row_prob = p[:, :rows] / p[:, :rows].sum(-1, keepdim=True).clamp_min(1e-8)
    col_prob = p[:, :, :cols] / p[:, :, :cols].sum(-2, keepdim=True).clamp_min(1e-8)
    return -0.5 * ((row_target * row_prob.clamp_min(1e-8).log()).sum(-1).mean()
                   + (col_target * col_prob.clamp_min(1e-8).log()).sum(-2).mean())


METRIC_KEYS = ("R@1", "R@5", "R@10", "Recall@top1", "AP", "SDM@1", "SDM@3",
               "SDM@5", "Dis@1", "Dis@3", "Dis@5")


def format_metrics(metrics):
    names = {"R@1": "Recall@1", "R@5": "Recall@5", "R@10": "Recall@10"}
    return " - ".join(f"{names.get(key, key)}: {metrics[key]:.4f}" for key in METRIC_KEYS)


def symmetric_info_nce(scores):
    labels = torch.arange(scores.shape[0], device=scores.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) * 0.5


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    global_features, local_features = [], []
    for images in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            global_feature, local_feature = model(images)
        global_features.append(global_feature.float())
        local_features.append(local_feature.float())
    return torch.cat(global_features), torch.cat(local_features)


@torch.no_grad()
def evaluate(model, query_loader, gallery_loader, query_ds, gallery_ds, device,
             run_root=None, epoch=None):
    query_global, query_local = extract_features(model, query_loader, device)
    gallery_global, gallery_local = extract_features(model, gallery_loader, device)
    query_global = query_global.to(device)
    query_local = query_local.to(device)
    gallery_global = gallery_global.to(device)
    gallery_local = gallery_local.to(device)

    all_scores = []
    query_chunk, gallery_chunk = 16, 128
    print("Evaluating full gallery with global + 0.5 * local scores", flush=True)
    for q_start in range(0, len(query_ds), query_chunk):
        qg = query_global[q_start:q_start + query_chunk]
        ql = query_local[q_start:q_start + query_chunk]
        score_chunks = []
        for g_start in range(0, len(gallery_ds), gallery_chunk):
            gg = gallery_global[g_start:g_start + gallery_chunk]
            gl = gallery_local[g_start:g_start + gallery_chunk]
            global_score = qg @ gg.T
            local_score = model.pair_local_score(ql, gl)
            score_chunks.append((global_score + 0.5 * local_score).cpu())
        all_scores.append(torch.cat(score_chunks, dim=1))
        if q_start % (query_chunk * 32) == 0:
            print(f"Scored {min(q_start + query_chunk, len(query_ds))}/{len(query_ds)} queries", flush=True)
    scores = torch.cat(all_scores).numpy()

    gallery_index = {name: i for i, name in enumerate(gallery_ds.images_name)}
    cmc = np.zeros(len(gallery_ds), dtype=np.float64)
    ap_values, sdm_values, dis_values, first_ranks, top10 = [], [], [], [], []
    for i, query_name in enumerate(query_ds.images_name):
        positive_names = query_ds.pairs_drone2sate_dict.get(query_name, [])
        positives = {gallery_index[name] for name in positive_names if name in gallery_index}
        order = np.argsort(scores[i])[::-1]
        hits = np.isin(order, list(positives)).astype(np.int32)
        if hits.any():
            first = int(np.flatnonzero(hits)[0])
            cmc[first:] += 1
            ap_values.append(average_precision_score(hits, np.arange(len(hits), 0, -1)))
        else:
            first = -1
        first_ranks.append(first)
        top10.append(order[:10])
        qloc = query_ds.images_center_loc_xy[i]
        sdm_values.append(sdm(qloc, [1, 3, 5], order, gallery_ds.images_center_loc_xy))
        dis_values.append(get_dis(qloc, order, gallery_ds.images_center_loc_xy, [1, 3, 5]))
    n = len(query_ds)
    cmc /= n
    sdm_mean, dis_mean = np.mean(sdm_values, axis=0), np.mean(dis_values, axis=0)
    metrics = {
        "R@1": 100 * cmc[0],
        "R@5": 100 * cmc[4],
        "R@10": 100 * cmc[9],
        # Retain Game4Loc's exact top-1% indexing convention.
        "Recall@top1": 100 * cmc[round(len(gallery_ds) * 0.01)],
        "AP": 100 * float(np.mean(ap_values)),
        "SDM@1": float(sdm_mean[0]), "SDM@3": float(sdm_mean[1]), "SDM@5": float(sdm_mean[2]),
        "Dis@1": float(dis_mean[0]), "Dis@3": float(dis_mean[1]), "Dis@5": float(dis_mean[2]),
    }
    print(format_metrics(metrics), flush=True)
    if run_root is not None:
        np.savez_compressed(os.path.join(run_root, f"query_results_epoch_{epoch:02d}.npz"),
                            query_names=np.asarray(query_ds.images_name),
                            gallery_names=np.asarray(gallery_ds.images_name),
                            top10=np.asarray(top10), first_positive_rank=np.asarray(first_ranks),
                            sdm=np.asarray(sdm_values), distance=np.asarray(dis_values))
    del query_global, query_local, gallery_global, gallery_local, scores
    torch.cuda.empty_cache()
    return metrics


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    with open(str(path) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(str(path) + ".tmp", path)


def train(args):
    setup_system(seed=args.seed, cudnn_benchmark=True, cudnn_deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = os.path.join(args.output_dir, args.model, args.run_name)
    os.makedirs(run_root, exist_ok=False)
    with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    shutil.copyfile(__file__, os.path.join(run_root, "train_partial_snapshot.py"))
    shutil.copyfile("train.sh", os.path.join(run_root, "train_sh_snapshot.sh"))
    sys.stdout = Logger(os.path.join(run_root, f"{args.run_name}.log"))
    print(f"Run: {args.run_name} | Method: {args.method} | Device: {device}")
    print(f"Cross-Area | epochs={args.epochs} | batch_size={args.batch_size} | seed={args.seed}")
    print("Train supervision: merged positive+semi-positive pairs; no IoU weights; symmetric InfoNCE")
    print("Local descriptors: " + ("4x4 + 6x6, equal-weight scale scores" if args.method == "E6_MS" else "4x4")
          + "; dimension 256; score = global + 0.5 * local", flush=True)
    print("Initialization: timm pretrained backbone; new projection; NO E6 checkpoint loaded", flush=True)
    print(f"Auxiliary crop loss: weight={args.aug_weight if args.method == 'E6_AUG' else 0}; "
          f"sources per domain={args.aug_per_domain}; extra compute only for E6_AUG", flush=True)

    model = LocalRetrievalModel(args.model, args.img_size, args.method).to(device)
    provenance = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "initial_state_sha256": state_digest(model.state_dict()),
        "checkpoint_start": None, "pretrained_backbone": True,
        "optimizer": "AdamW", "weight_decay": 0.01, "scheduler": "cosine", "warmup_epochs": 0.1,
        "train_mode": "pos_semipos", "test_mode": "pos", "iou_weights": False,
        "local_weight": 0.5, "local_sizes": model.local_sizes,
        "evidence_formula": "sqrt((support/(support+0.1))*(1-weighted_conditional_entropy))",
        "aug_geometry": "two equal-area crops, side fraction 0.5 or 0.75, independent quarter rotations",
        "evaluation": "full gallery fusion; original Game4Loc metrics; no Top-K reranking",
    }
    atomic_json(os.path.join(run_root, "provenance.json"), provenance)
    print(f"Initial state SHA256: {provenance['initial_state_sha256']}", flush=True)
    data_config = timm.data.resolve_model_data_config(model.backbone)
    val_tf, sat_tf, drone_tf = get_transforms(
        (args.img_size, args.img_size), mean=data_config["mean"], std=data_config["std"], sat_rot=True
    )
    train_ds = GTADatasetTrain(
        data_root=args.data_root,
        pairs_meta_file="cross-area-drone2sate-train.json",
        transforms_query=drone_tf,
        transforms_gallery=sat_tf,
        group_len=2,
        prob_flip=0.5,
        shuffle_batch_size=args.batch_size,
        mode="pos_semipos",
        train_ratio=1.0,
    )
    # Dedicated loader/crop RNGs keep auxiliary views from changing MES order.
    loader_rng = torch.Generator().manual_seed(args.seed)
    crop_rng = random.Random(args.seed + 10000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=False, pin_memory=True, generator=loader_rng)
    query_ds = GTADatasetEval("cross-area-drone2sate-test.json", args.data_root, "drone", mode="pos",
                              transforms=val_tf, query_mode="D2S")
    gallery_ds = GTADatasetEval("cross-area-drone2sate-test.json", args.data_root, "sate", mode="pos",
                                sate_img_dir="satellite", query_mode="D2S", transforms=val_tf)
    query_loader = DataLoader(query_ds, batch_size=args.eval_batch_size, num_workers=args.workers,
                              shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.eval_batch_size, num_workers=args.workers,
                                shuffle=False, pin_memory=True)
    print(f"Training pairs: {len(train_ds)}; queries: {len(query_ds)}; gallery: {len(gallery_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(len(train_loader) * 0.1), total_steps)
    scaler = GradScaler(enabled=args.amp)
    best_r1 = -1.0
    history = []
    sampling_history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]", flush=True)
        train_ds.shuffle()
        sample_hash = hashlib.sha256()
        for query_path, gallery_path, _ in train_ds.samples:
            sample_hash.update((query_path + "\0" + gallery_path + "\n").encode())
        sampling_history.append({"epoch": epoch, "pairs": len(train_ds), "sha256": sample_hash.hexdigest()})
        atomic_json(os.path.join(run_root, "sampling_history.json"), sampling_history)
        print(f"MES pair order SHA256: {sample_hash.hexdigest()}", flush=True)
        model.train()
        loss_total = 0.0
        retrieval_total = aug_total = 0.0
        steps = 0
        start = time.time()
        for step, (query, gallery, _) in enumerate(train_loader, 1):
            query, gallery = query.to(device, non_blocking=True), gallery.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                q_global, q_local = model(query)
                g_global, g_local = model(gallery)
                global_scores = q_global @ g_global.T
                local_scores = model.pair_local_score(q_local, g_local)
                scores = model.logit_scale.exp().clamp(max=100) * (global_scores + 0.5 * local_scores)
                retrieval_loss = symmetric_info_nce(scores)
            if not torch.isfinite(retrieval_loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
            # Free the main batch graph before encoding auxiliary crop views.
            # Accumulating both backwards before the optimizer step gives the
            # same sum of gradients without retaining both activation graphs.
            scaler.scale(retrieval_loss).backward()
            aug_loss = retrieval_loss.detach().new_zeros(())
            if args.method == "E6_AUG":
                source = torch.cat((query[:args.aug_per_domain], gallery[:args.aug_per_domain]))
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                    with autocast(enabled=args.amp):
                        a_view, b_view, target = known_crop_views(source, crop_rng)
                        _, augmented_local = model(torch.cat((a_view, b_view)))
                        a_local, b_local = augmented_local.chunk(2)
                        aug_loss = local_correspondence_loss(model, a_local, b_local, target)
                    if not torch.isfinite(aug_loss):
                        raise FloatingPointError(f"Nonfinite auxiliary loss at epoch={epoch}, step={step}")
                    scaler.scale(args.aug_weight * aug_loss).backward()
            loss = retrieval_loss.detach() + args.aug_weight * aug_loss.detach()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), 100.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_total += loss.item()
            retrieval_total += retrieval_loss.item()
            aug_total += aug_loss.item()
            steps += 1
            if step % 50 == 1:
                print(f"Iteration {step}/{len(train_loader)} loss={loss.item():.4f} "
                      f"avg={loss_total / steps:.4f} retrieval={retrieval_loss.item():.4f} aug={aug_loss.item():.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.7f} "
                      f"elapsed={time.time()-start:.1f}s", flush=True)
        print(f"Epoch {epoch} train_loss={loss_total/max(steps,1):.5f} "
              f"retrieval_loss={retrieval_total/max(steps,1):.5f} aug_loss={aug_total/max(steps,1):.5f}", flush=True)
        epoch_checkpoint = os.path.join(run_root, f"epoch_{epoch:02d}.pth")
        torch.save(model.state_dict(), epoch_checkpoint + ".tmp")
        os.replace(epoch_checkpoint + ".tmp", epoch_checkpoint)
        print(f"[Epoch {epoch}/{args.epochs}] Evaluate", flush=True)
        metrics = evaluate(model, query_loader, gallery_loader, query_ds, gallery_ds, device, run_root, epoch)
        metrics["epoch"] = epoch
        history.append(metrics)
        if metrics["R@1"] > best_r1:
            best_r1 = metrics["R@1"]
            shutil.copyfile(epoch_checkpoint, os.path.join(run_root, "best.pth"))
        shutil.copyfile(epoch_checkpoint, os.path.join(run_root, "last.pth"))
        atomic_json(os.path.join(run_root, "metrics.json"), history)
    print("Completed metrics:")
    for item in history:
        print(f"Epoch {item['epoch']}: {format_metrics(item)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["E4", "E5", "E6", "E6_MS", "E6_EVIDENCE", "E6_AUG"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default="./data/GTA-UAV-data")
    parser.add_argument("--output-dir", default="./work_dir/gta-partial")
    parser.add_argument("--model", default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--aug-weight", type=float, default=0.1)
    parser.add_argument("--aug-per-domain", type=int, default=4)
    train(parser.parse_args())
