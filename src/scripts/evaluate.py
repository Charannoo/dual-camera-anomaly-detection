import os
import argparse
import time
import yaml
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.clip_branch import ClipBranch
from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model_size_kb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_kb = (param_size + buffer_size) / 1024
    return size_all_kb

def evaluate(category):
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
    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print("WARNING: Checkpoint not found, using randomly initialized model.")

    model.eval()

    scorer = ResidualScorer()
    scorer.fit(train_data["rgb_feats"], train_data["depth_feats"], model,
               text_tokens=train_data["text_tokens"])

    results = scorer.score(test_data["rgb_feats"], test_data["depth_feats"], model,
                           text_tokens=test_data["text_tokens"])

    img_scores = results["image_score"].numpy()
    img_labels = test_data["labels"].numpy()

    pixel_scores = results["anomaly_map"].numpy()
    pixel_gts = test_data["gts"].numpy().squeeze(1)

    i_auroc = roc_auc_score(img_labels, img_scores)

    pixel_scores_flat = pixel_scores.flatten()
    pixel_gts_flat = pixel_gts.flatten()
    p_auroc = roc_auc_score(pixel_gts_flat, pixel_scores_flat)

    print(f"\n================ Evaluation: {category} ================")
    print(f"Image AUROC (I-AUROC): {i_auroc:.4f}")
    print(f"Pixel AUROC (P-AUROC): {p_auroc:.4f}")

    fusion_size_kb = get_model_size_kb(model)
    print(f"Fusion Model Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Fusion Model Weight Size: {fusion_size_kb:.2f} KB")

    # Latency benchmarking (end-to-end)
    rgb_backbone = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_backbone = DepthBackbone().to(device)
    clip_config = config.get("clip", {})
    clip_branch = ClipBranch(
        model_name=clip_config.get("model_name", "openai/clip-vit-base-patch32")
    ).to(device)

    rgb_backbone.eval()
    depth_backbone.eval()
    clip_branch.eval()

    print("Benchmarking CPU inference latency...")
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
    total_time = time.time() - start_time
    avg_latency_ms = (total_time / num_runs) * 1000
    print(f"Average CPU Inference Latency per Image: {avg_latency_ms:.2f} ms")
    print("========================================================\n")

    return {
        "i_auroc": i_auroc,
        "p_auroc": p_auroc,
        "latency_ms": avg_latency_ms,
        "fusion_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cable_gland", help="Category to evaluate")
    args = parser.parse_args()

    evaluate(args.category)
