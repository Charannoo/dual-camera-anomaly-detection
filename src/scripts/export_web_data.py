"""
Export pre-computed pipeline outputs for the static 3D Results Explorer.

Run once from the project root:
    python -m src.scripts.export_web_data

Writes to web/data/{category}/{sample_id}/:
  rgb.jpg, pointcloud.json, heatmap.png, meta.json
Plus web/data/index.json listing all samples.
"""

import os
import json
import yaml
import torch
import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer
from src.models.calibration import ScoreCalibrator

DATA_ROOT = "web/data"
TARGET_POINTS = 8000
PHRASES = [
    "cracked", "contaminated", "cut", "damaged",
    "good", "hole", "thread", "discolored",
]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def defect_type_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "test" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def save_heatmap(anomaly_map, out_path):
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=72)
    ax.imshow(anomaly_map, cmap="hot", vmin=0)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=72, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_pointcloud(xyz_resized, anomaly_224):
    xs = xyz_resized[:, :, 0].flatten()
    ys = xyz_resized[:, :, 1].flatten()
    zs = xyz_resized[:, :, 2].flatten()
    intensities = anomaly_224.flatten().tolist()

    n = len(xs)
    if n > TARGET_POINTS:
        idx = np.random.default_rng(0).choice(n, TARGET_POINTS, replace=False)
        idx.sort()
        xs, ys, zs = xs[idx], ys[idx], zs[idx]
        intensities = [intensities[i] for i in idx]

    return {
        "positions": np.stack([xs, ys, zs], axis=1).round(4).tolist(),
        "anomalyIntensity": [round(v, 6) for v in intensities],
    }


def export_category(category, features_dir, checkpoint_dir):
    train_cache_path = os.path.join(features_dir, f"{category}_train.pt")
    test_cache_path = os.path.join(features_dir, f"{category}_test.pt")
    ckpt_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")

    for p in [train_cache_path, test_cache_path, ckpt_path]:
        if not os.path.exists(p):
            print(f"  SKIP {category}: missing {p}")
            return []

    train_data = torch.load(train_cache_path, map_location="cpu")
    test_data = torch.load(test_cache_path, map_location="cpu")

    model = CrossAttentionFusion()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["fusion"] if "fusion" in ckpt else ckpt)
    model.eval()

    scorer = ResidualScorer()
    with torch.no_grad():
        scorer.fit(
            train_data["rgb_feats"], train_data["depth_feats"], model,
            text_tokens=train_data["text_tokens"],
        )

    calibrator = ScoreCalibrator()
    with torch.no_grad():
        train_results = scorer.score(
            train_data["rgb_feats"], train_data["depth_feats"], model,
            text_tokens=train_data["text_tokens"],
        )
    calibrator.fit(train_results["image_score"].cpu().numpy())

    with torch.no_grad():
        test_results = scorer.score(
            test_data["rgb_feats"], test_data["depth_feats"], model,
            text_tokens=test_data["text_tokens"],
        )

    anomaly_maps = test_results["anomaly_map"].cpu().numpy()
    raw_scores = test_results["image_score"].cpu().numpy()
    cal_scores = np.array([calibrator.calibrate(s) for s in raw_scores])
    clip_sims_all = test_data["clip_sims"].numpy()
    rgb_paths = test_data["rgb_paths"]
    depth_paths = test_data["depth_paths"]
    labels = test_data["labels"].numpy()
    gts = test_data["gts"].numpy().squeeze(1)

    entries = []
    for i in range(len(labels)):
        sample_id = f"{i:04d}"
        sample_dir = os.path.join(DATA_ROOT, category, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        rgb_path = rgb_paths[i]
        depth_path = depth_paths[i]
        dtype = defect_type_from_path(rgb_path)
        anomaly_map = anomaly_maps[i]

        # --- RGB thumbnail (224x224 JPEG) ---
        rgb_img = Image.open(rgb_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        rgb_img.save(os.path.join(sample_dir, "rgb.jpg"), "JPEG", quality=90)

        # --- Heatmap PNG ---
        save_heatmap(anomaly_map, os.path.join(sample_dir, "heatmap.png"))

        # --- Point cloud (XYZ + anomaly intensity) ---
        xyz_raw = tifffile.imread(depth_path)
        xyz_resized = cv2.resize(
            xyz_raw, (224, 224), interpolation=cv2.INTER_LINEAR
        )
        pc = make_pointcloud(xyz_resized, anomaly_map)
        with open(os.path.join(sample_dir, "pointcloud.json"), "w") as f:
            json.dump(pc, f)

        # --- Metadata ---
        clip_sims = clip_sims_all[i].tolist()
        top_idx = int(np.argmax(clip_sims))
        gt_label = "normal" if labels[i] == 0 else "anomalous"

        meta = {
            "category": category,
            "defectType": dtype,
            "groundTruth": gt_label,
            "calibratedScore": round(float(cal_scores[i]), 4),
            "rawScore": round(float(raw_scores[i]), 4),
            "clipTopPhrase": PHRASES[top_idx],
            "clipSimilarity": round(clip_sims[top_idx], 4),
            "clipAllPhrases": PHRASES,
            "clipAllScores": [round(s, 4) for s in clip_sims],
        }
        with open(os.path.join(sample_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        entries.append({
            "category": category,
            "sampleId": sample_id,
            "defectType": dtype,
            "groundTruth": gt_label,
            "calibratedScore": round(float(cal_scores[i]), 4),
        })

    return entries


def main():
    config = load_config()
    features_dir = config["cache"]["features_dir"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    categories = config["dataset"]["categories"]

    os.makedirs(DATA_ROOT, exist_ok=True)

    all_entries = []
    for cat in categories:
        print(f"Exporting {cat}...", flush=True)
        entries = export_category(cat, features_dir, checkpoint_dir)
        all_entries.extend(entries)
        print(f"  {len(entries)} samples", flush=True)

    index = {"categories": categories, "samples": all_entries}
    with open(os.path.join(DATA_ROOT, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone: {len(all_entries)} samples total across {len(categories)} categories.")
    print(f"Data written to {DATA_ROOT}/")


if __name__ == "__main__":
    main()
