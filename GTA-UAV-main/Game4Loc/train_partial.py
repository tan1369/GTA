import argparse
import json
import math
import os
import random
import shutil
import sys
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
from game4loc.utils import Logger, setup_system


class LocalRetrievalModel(nn.Module):
    def __init__(self, model_name, img_size, method, local_size=4, local_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size
        )
        self.method = method
        self.local_size = local_size
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
        patch_tokens = F.adaptive_avg_pool2d(patch_tokens, (self.local_size, self.local_size))
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        local_features = F.normalize(self.local_projection(patch_tokens), dim=-1)
        return F.normalize(global_feature, dim=-1), local_features

    def pair_local_score(self, local_a, local_b):
        sim = torch.einsum("bmd,cnd->bcmn", local_a, local_b)
        if self.method == "E4":
            return (sim.max(dim=-1).values.mean(-1) + sim.max(dim=-2).values.mean(-1)) * 0.5
        if self.method == "E5":
            pair_shape = sim.shape[:2]
            flat_sim = sim.flatten(0, 1)
            transport = self._sinkhorn(flat_sim / 0.07)
            return (transport * flat_sim).sum(dim=(-1, -2)).reshape(pair_shape)
        pair_shape = sim.shape[:2]
        return self._dustbin_transport_score(sim.flatten(0, 1)).reshape(pair_shape)

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

    def _dustbin_transport_score(self, sim, iterations=8):
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
        transport = torch.exp(scores + u.unsqueeze(2) + v.unsqueeze(1))[:, :count, :count]
        matched_mass = transport.sum(dim=(-1, -2)).clamp_min(1e-6)
        return (transport * sim).sum(dim=(-1, -2)) / matched_mass


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
def evaluate(model, query_loader, gallery_loader, query_ds, gallery_ds, device):
    query_global, query_local = extract_features(model, query_loader, device)
    gallery_global, gallery_local = extract_features(model, gallery_loader, device)
    query_global = query_global.to(device)
    query_local = query_local.to(device)
    gallery_global = gallery_global.to(device)
    gallery_local = gallery_local.to(device)

    all_scores = []
    query_chunk, gallery_chunk = 32, 256
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
    r1 = r5 = r10 = ap_sum = 0.0
    for i, query_name in enumerate(query_ds.images_name):
        positive_names = query_ds.pairs_drone2sate_dict.get(query_name, [])
        positives = {gallery_index[name] for name in positive_names if name in gallery_index}
        order = np.argsort(scores[i])[::-1]
        hits = np.fromiter((int(idx in positives) for idx in order), dtype=np.int32)
        r1 += float(hits[:1].any())
        r5 += float(hits[:5].any())
        r10 += float(hits[:10].any())
        if hits.any():
            ap_sum += average_precision_score(hits, np.arange(len(hits), 0, -1))
    n = len(query_ds)
    metrics = {
        "R@1": 100 * r1 / n,
        "R@5": 100 * r5 / n,
        "R@10": 100 * r10 / n,
        "AP": 100 * ap_sum / n,
    }
    print(" - ".join(f"{key}: {value:.4f}" for key, value in metrics.items()), flush=True)
    del query_global, query_local, gallery_global, gallery_local, scores
    torch.cuda.empty_cache()
    return metrics


def train(args):
    setup_system(seed=args.seed, cudnn_benchmark=True, cudnn_deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = os.path.join(args.output_dir, args.model, args.run_name)
    os.makedirs(run_root, exist_ok=False)
    with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    shutil.copyfile(__file__, os.path.join(run_root, "train_partial_snapshot.py"))
    sys.stdout = Logger(os.path.join(run_root, f"{args.run_name}.log"))
    print(f"Run: {args.run_name} | Method: {args.method} | Device: {device}")
    print(f"Cross-Area | epochs={args.epochs} | batch_size={args.batch_size} | seed={args.seed}")
    print("Train supervision: merged positive+semi-positive pairs; no IoU weights; symmetric InfoNCE")
    print("Local descriptors: adaptive 4x4 token grid, dimension 256; score = global + 0.5 * local")

    model = LocalRetrievalModel(args.model, args.img_size, args.method).to(device)
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=False, pin_memory=True)
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
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]", flush=True)
        train_ds.shuffle()
        model.train()
        loss_total = 0.0
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
                loss = symmetric_info_nce(scores)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), 100.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_total += loss.item()
            steps += 1
            if step % 50 == 1:
                print(f"Iteration {step}/{len(train_loader)} loss={loss.item():.4f} "
                      f"avg={loss_total / steps:.4f} lr={optimizer.param_groups[0]['lr']:.7f} "
                      f"elapsed={time.time()-start:.1f}s", flush=True)
        print(f"Epoch {epoch} train_loss={loss_total/max(steps,1):.5f}", flush=True)
        metrics = evaluate(model, query_loader, gallery_loader, query_ds, gallery_ds, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        if metrics["R@1"] > best_r1:
            best_r1 = metrics["R@1"]
            torch.save(model.state_dict(), os.path.join(run_root, "best.pth"))
        torch.save(model.state_dict(), os.path.join(run_root, "last.pth"))
        with open(os.path.join(run_root, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    print("Completed metrics:")
    for item in history:
        print(f"E{item['epoch']}: R@1={item['R@1']:.4f}, R@5={item['R@5']:.4f}, "
              f"R@10={item['R@10']:.4f}, AP={item['AP']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["E4", "E5", "E6"], required=True)
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
    train(parser.parse_args())
