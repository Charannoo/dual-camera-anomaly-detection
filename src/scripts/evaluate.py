import os
import argparse
import time
import yaml
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.clip_branch import ClipBranch
from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer, MemoryBankScorer
from src.models.calibration import ScoreCalibrator

DEFECT_DESCRIPTIONS = {
    "bent": "Bent", "color": "Color", "combined": "Combined",
    "contamination": "Contamination", "crack": "Crack", "cut": "Cut",
    "good": "Normal", "hole": "Hole", "thread": "Thread",
}

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model_size_kb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024

def get_defect_type_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "test" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"

def safe_auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)

def evaluate(category, scorer_type="mahalanobis"):
    config = load_config()
    features_dir = config["cache"]["features_dir"]
    checkpoint_dir = config["model"]["checkpoint_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = os.path.join(features_dir, f"{category}_train.pt")
    test_path = os.path.join(features_dir, f"{category}_test.pt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Cached features not found. Extract them first.")

    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    model = CrossAttentionFusion().to(device)
    e2e_path = os.path.join(checkpoint_dir, f"{category}_e2e.pt")
    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")

    if os.path.exists(e2e_path):
        ckpt = torch.load(e2e_path, map_location=device)
        model.load_state_dict(ckpt["fusion"])
        print(f"Loaded E2E fusion weights from {e2e_path}")
    elif os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded fusion checkpoint from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint found, using random model.")

    model.eval()

    if scorer_type == "memorybank":
        scorer = MemoryBankScorer(k=5)
        print("Using MemoryBankScorer (k-NN, k=5)")
    else:
        scorer = ResidualScorer()
        print("Using ResidualScorer (Mahalanobis)")

    scorer.fit(train_data["rgb_feats"], train_data["depth_feats"], model,
               text_tokens=train_data["text_tokens"])

    calibrator = ScoreCalibrator()

    train_results = scorer.score(train_data["rgb_feats"], train_data["depth_feats"], model,
                                  text_tokens=train_data["text_tokens"])
    train_img_scores = train_results["image_score"].cpu().numpy()
    calibrator.fit(train_img_scores)

    test_results = scorer.score(test_data["rgb_feats"], test_data["depth_feats"], model,
                                 text_tokens=test_data["text_tokens"])

    img_scores = test_results["image_score"].cpu().numpy()
    img_labels = test_data["labels"].numpy()
    pixel_scores = test_results["anomaly_map"].cpu().numpy()
    pixel_gts = test_data["gts"].numpy().squeeze(1)

    cal_scores = np.array([calibrator.calibrate(s) for s in img_scores])

    i_auroc = safe_auroc(img_labels, img_scores)
    i_auroc_cal = safe_auroc(img_labels, cal_scores)
    p_auroc = safe_auroc(pixel_gts.flatten(), pixel_scores.flatten())

    print(f"\n{'='*60}")
    print(f"  Evaluation: {category}")
    print(f"{'='*60}")
    print(f"  I-AUROC (raw):       {i_auroc:.4f}")
    print(f"  I-AUROC (calibrated): {i_auroc_cal:.4f}")
    print(f"  P-AUROC:             {p_auroc:.4f}")

    test_rgb_paths = test_data["rgb_paths"]
    defect_types = [get_defect_type_from_path(p) for p in test_rgb_paths]
    defect_groups = defaultdict(lambda: {"scores": [], "labels": []})
    for i, dt in enumerate(defect_types):
        defect_groups[dt]["scores"].append(cal_scores[i])
        defect_groups[dt]["labels"].append(img_labels[i])

    print(f"\n  Per-defect-type breakdown (calibrated):")
    print(f"  {'Type':<20s} {'Count':>5s} {'AUROC':>8s}")
    print(f"  {'-'*35}")

    defect_aurocs = {}
    for dt in sorted(defect_groups.keys()):
        dg = defect_groups[dt]
        labels_arr = np.array(dg["labels"])
        scores_arr = np.array(dg["scores"])
        n = len(labels_arr)
        auroc = safe_auroc(labels_arr, scores_arr)
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A (1 class)"
        defect_aurocs[dt] = auroc
        print(f"  {DEFECT_DESCRIPTIONS.get(dt, dt):<20s} {n:>5d} {auroc_str:>8s}")

    fusion_size_kb = get_model_size_kb(model)
    fusion_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Fusion Model Params: {fusion_params:,}")
    print(f"  Fusion Model Size:   {fusion_size_kb:.1f} KB")

    rgb_backbone = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_backbone = DepthBackbone().to(device)
    clip_config = config.get("clip", {})
    clip_branch = ClipBranch(
        model_name=clip_config.get("model_name", "openai/clip-vit-base-patch32")
    ).to(device)

    rgb_backbone.eval()
    depth_backbone.eval()
    clip_branch.eval()

    dummy_rgb = torch.randn(1, 3, 224, 224).to(device)
    dummy_depth = torch.randn(1, 1, 224, 224).to(device)

    for _ in range(5):
        with torch.no_grad():
            f_rgb = rgb_backbone(dummy_rgb)
            f_dep = depth_backbone(dummy_depth)
            text_tok, _ = clip_branch(dummy_rgb)
            _ = scorer.score(f_rgb, f_dep, model, text_tokens=text_tok)

    num_runs = 20
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            f_rgb = rgb_backbone(dummy_rgb)
            f_dep = depth_backbone(dummy_depth)
            text_tok, _ = clip_branch(dummy_rgb)
            _ = scorer.score(f_rgb, f_dep, model, text_tokens=text_tok)
    avg_latency_ms = ((time.time() - start_time) / num_runs) * 1000
    print(f"  CPU Latency:         {avg_latency_ms:.1f} ms/image")
    print(f"{'='*60}\n")

    return {
        "category": category,
        "i_auroc": i_auroc,
        "i_auroc_cal": i_auroc_cal,
        "p_auroc": p_auroc,
        "latency_ms": avg_latency_ms,
        "fusion_params": fusion_params,
        "defect_aurocs": defect_aurocs,
    }


def evaluate_all(scorer_type="mahalanobis"):
    config = load_config()
    categories = config["dataset"]["categories"]
    all_results = []
    for cat in categories:
        try:
            result = evaluate(cat, scorer_type=scorer_type)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR evaluating {cat}: {e}")

    if all_results:
        print(f"\n{'='*75}")
        print(f"  SUMMARY — All Categories")
        print(f"{'='*75}")
        print(f"  {'Category':<16s} {'I-AUROC':>8s} {'I-AUROC-Cal':>12s} {'P-AUROC':>8s} {'Latency':>10s}")
        print(f"  {'-'*58}")
        for r in all_results:
            print(f"  {r['category']:<16s} {r['i_auroc']:>8.4f} {r['i_auroc_cal']:>12.4f} {r['p_auroc']:>8.4f} {r['latency_ms']:>9.1f}ms")

        avg_i = np.nanmean([r["i_auroc"] for r in all_results])
        avg_i_cal = np.nanmean([r["i_auroc_cal"] for r in all_results])
        avg_p = np.nanmean([r["p_auroc"] for r in all_results])
        avg_lat = np.mean([r["latency_ms"] for r in all_results])
        print(f"  {'-'*58}")
        print(f"  {'Average':<16s} {avg_i:>8.4f} {avg_i_cal:>12.4f} {avg_p:>8.4f} {avg_lat:>9.1f}ms")
        print(f"{'='*75}\n")

        csv_path = os.path.join("outputs", "results.csv")
        os.makedirs("outputs", exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("category,i_auroc,i_auroc_calibrated,p_auroc,latency_ms\n")
            for r in all_results:
                f.write(f"{r['category']},{r['i_auroc']:.4f},{r['i_auroc_cal']:.4f},{r['p_auroc']:.4f},{r['latency_ms']:.1f}\n")
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None, help="Category to evaluate (omit for all)")
    parser.add_argument("--scorer", type=str, default="mahalanobis", choices=["mahalanobis", "memorybank"])
    args = parser.parse_args()

    if args.category:
        evaluate(args.category, scorer_type=args.scorer)
    else:
        evaluate_all(scorer_type=args.scorer)
