"""Recompute full Game4Loc metrics for saved partial-matching checkpoints.

This script does not train or modify checkpoints.  It uses the same
global + 0.5 * local score used by train_partial.py and the spatial metric
definitions from game4loc.evaluate.gta.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import get_dis, sdm


class LocalRetrievalModel(nn.Module):
    """Architecture kept identical to train_partial.py, without pretraining IO."""

    def __init__(self, model_name, img_size, method, local_size=4, local_dim=256):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, img_size=img_size
        )
        self.method = method
        self.local_size = local_size
        embed_dim = self.backbone.num_features
        self.local_projection = nn.Linear(embed_dim, local_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.dustbin_score = nn.Parameter(torch.zeros([]))

    def forward(self, image):
        tokens = self.backbone.forward_features(image)
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
        pair_shape = sim.shape[:2]
        if self.method == "E5":
            flat_sim = sim.flatten(0, 1)
            transport = self._sinkhorn(flat_sim / 0.07)
            return (transport * flat_sim).sum(dim=(-1, -2)).reshape(pair_shape)
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


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    gs, ls = [], []
    for images in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            g, l = model(images)
        gs.append(g.float())
        ls.append(l.float())
    return torch.cat(gs).to(device), torch.cat(ls).to(device)


@torch.no_grad()
def evaluate_checkpoint(model, query_loader, gallery_loader, query_ds, gallery_ds, device,
                        query_chunk=8, gallery_chunk=64):
    qg, ql = extract_features(model, query_loader, device)
    gg, gl = extract_features(model, gallery_loader, device)
    score_rows = []
    for qs in range(0, len(query_ds), query_chunk):
        row_chunks = []
        qg_i, ql_i = qg[qs:qs + query_chunk], ql[qs:qs + query_chunk]
        for gs in range(0, len(gallery_ds), gallery_chunk):
            gg_i, gl_i = gg[gs:gs + gallery_chunk], gl[gs:gs + gallery_chunk]
            row_chunks.append((qg_i @ gg_i.T + 0.5 * model.pair_local_score(ql_i, gl_i)).cpu())
        score_rows.append(torch.cat(row_chunks, dim=1))
        if qs % (query_chunk * 128) == 0:
            print(f"Scored {min(qs + query_chunk, len(query_ds))}/{len(query_ds)} queries", flush=True)
    scores = torch.cat(score_rows).numpy()

    gallery_index = {name: i for i, name in enumerate(gallery_ds.images_name)}
    cmc = np.zeros(len(gallery_ds.images_name), dtype=np.float64)
    ap_values, sdm_values, dis_values = [], [], []
    for i, query_name in enumerate(query_ds.images_name):
        order = np.argsort(scores[i])[::-1]
        positives = {gallery_index[name] for name in query_ds.pairs_drone2sate_dict.get(query_name, [])
                     if name in gallery_index}
        good = np.isin(order, list(positives))
        if good.any():
            ap_values.append(average_precision_score(good.astype(np.int32),
                                                     np.arange(len(good), 0, -1)))
            first = np.where(good)[0][0]
            cmc[first:] += 1.0
        qloc = query_ds.images_center_loc_xy[i]
        sdm_values.append(sdm(qloc, [1, 3, 5], order, gallery_ds.images_center_loc_xy))
        dis_values.append(get_dis(qloc, order, gallery_ds.images_center_loc_xy, [1, 3, 5]))

    n = len(query_ds)
    cmc /= n
    sdm_mean = np.mean(np.asarray(sdm_values), axis=0)
    dis_mean = np.mean(np.asarray(dis_values), axis=0)
    top1_percent = round(len(gallery_ds) * 0.01)
    return {
        "R@1": float(cmc[0] * 100),
        "R@5": float(cmc[4] * 100),
        "R@10": float(cmc[9] * 100),
        "Recall@top1": float(cmc[top1_percent] * 100),
        "AP": float(np.mean(ap_values) * 100),
        "SDM@1": float(sdm_mean[0]),
        "SDM@3": float(sdm_mean[1]),
        "SDM@5": float(sdm_mean[2]),
        "Dis@1": float(dis_mean[0]),
        "Dis@3": float(dis_mean[1]),
        "Dis@5": float(dis_mean[2]),
    }


def format_metrics(m):
    keys = ["R@1", "R@5", "R@10", "Recall@top1", "AP", "SDM@1", "SDM@3", "SDM@5", "Dis@1", "Dis@3", "Dis@5"]
    return " - ".join(f"{k}: {m[k]:.4f}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./work_dir/gta-partial/vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    ap.add_argument("--methods", nargs="+", default=["E4_BidirectionalMax", "E5_BalancedOT", "E6_DustbinPartialOT"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--query-chunk", type=int, default=8)
    ap.add_argument("--gallery-chunk", type=int, default=64)
    args = ap.parse_args()
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    for method_dir_name in args.methods:
        method_dir = root / method_dir_name
        with open(method_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        method = cfg["method"]
        model = LocalRetrievalModel(cfg["model"], cfg["img_size"], method).to(device)
        data_cfg = timm.data.resolve_model_data_config(model.backbone)
        val_tf, _, _ = get_transforms((cfg["img_size"], cfg["img_size"]),
                                      mean=data_cfg["mean"], std=data_cfg["std"], sat_rot=False)
        data_root = cfg["data_root"]
        qds = GTADatasetEval("cross-area-drone2sate-test.json", data_root, "drone", mode="pos",
                             transforms=val_tf, query_mode="D2S")
        gds = GTADatasetEval("cross-area-drone2sate-test.json", data_root, "sate", mode="pos",
                             sate_img_dir="satellite", query_mode="D2S", transforms=val_tf)
        ql = DataLoader(qds, batch_size=cfg.get("eval_batch_size", 128), shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        gl = DataLoader(gds, batch_size=cfg.get("eval_batch_size", 128), shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        checkpoint_names = ["best.pth", "last.pth"]
        results = {}
        for checkpoint_name in checkpoint_names:
            checkpoint = method_dir / checkpoint_name
            if not checkpoint.exists():
                continue
            print(f"\nEvaluating {method_dir_name}/{checkpoint_name}", flush=True)
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=True)
            metrics = evaluate_checkpoint(model, ql, gl, qds, gds, device,
                                          args.query_chunk, args.gallery_chunk)
            results[checkpoint_name] = metrics
            print(format_metrics(metrics), flush=True)
        with open(method_dir / "supplementary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {method_dir / 'supplementary_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
