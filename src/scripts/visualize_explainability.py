import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.clip_branch import ClipBranch
from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer

DEFECT_DESCRIPTIONS = {
    "bent": "Bent/Deformed",
    "color": "Color Anomaly",
    "combined": "Combined Defect",
    "contamination": "Contamination",
    "crack": "Crack/Fracture",
    "cut": "Cut/Scratch",
    "good": "Normal",
    "hole": "Hole/Puncture",
    "thread": "Thread Defect",
}


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_defect_type_from_path(rgb_path):
    parts = rgb_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "test" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def generate_explanations(category="cable_gland", num_per_type=2):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = load_config()
    data_dir = config["dataset"]["data_dir"]
    img_size = config["dataset"]["img_size"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    features_dir = config["cache"]["features_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rgb_backbone = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_backbone = DepthBackbone().to(device)
    clip_config = config.get("clip", {})
    clip_branch = ClipBranch(
        model_name=clip_config.get("model_name", "openai/clip-vit-base-patch32")
    ).to(device)
    fusion_model = CrossAttentionFusion().to(device)

    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "fusion" in ckpt:
            fusion_model.load_state_dict(ckpt["fusion"])
        else:
            fusion_model.load_state_dict(ckpt)
        print(f"Loaded fusion model weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    rgb_backbone.eval()
    depth_backbone.eval()
    clip_branch.eval()
    fusion_model.eval()

    train_path = os.path.join(features_dir, f"{category}_train.pt")
    train_data = torch.load(train_path)

    scorer = ResidualScorer()
    scorer.fit(train_data["rgb_feats"], train_data["depth_feats"], fusion_model,
               text_tokens=train_data["text_tokens"])

    dataset = MVTec3DADataset(data_dir, category, split="test", img_size=img_size)

    defect_groups = defaultdict(list)
    for i, lbl in enumerate(dataset.labels):
        if lbl == 1:
            dtype = get_defect_type_from_path(dataset.rgb_paths[i])
            defect_groups[dtype].append(i)

    print(f"\nDefect types found: {list(defect_groups.keys())}")
    for dt, indices in defect_groups.items():
        print(f"  {dt}: {len(indices)} samples")

    indices_to_run = []
    for dtype in sorted(defect_groups.keys()):
        indices_to_run.extend(defect_groups[dtype][:num_per_type])

    if not indices_to_run:
        print("No defective samples found!")
        return

    print(f"\nGenerating explainability outputs for {category} ({len(indices_to_run)} samples)...")
    os.makedirs("outputs/explainability", exist_ok=True)

    all_results = []

    for count, idx in enumerate(indices_to_run):
        sample = dataset[idx]
        rgb_tensor = sample["rgb"].unsqueeze(0).to(device)
        depth_tensor = sample["depth"].unsqueeze(0).to(device)
        gt_mask = sample["gt"].squeeze(0).numpy()

        dtype = get_defect_type_from_path(dataset.rgb_paths[idx])

        with torch.no_grad():
            rgb_feat = rgb_backbone(rgb_tensor)
            depth_feat = depth_backbone(depth_tensor)
            text_token, clip_sims = clip_branch(rgb_tensor)
            results = scorer.score(rgb_feat, depth_feat, fusion_model, text_tokens=text_token)

        anomaly_map = results["anomaly_map"].squeeze(0).numpy()
        image_score = results["image_score"].item()

        rgb_res = results["rgb_residual"].squeeze(0)
        depth_res = results["depth_residual"].squeeze(0)

        rgb_res_up = F.interpolate(rgb_res.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze().numpy()
        depth_res_up = F.interpolate(depth_res.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze().numpy()

        s_rgb = rgb_res.max().item()
        s_depth = depth_res.max().item()
        sum_s = s_rgb + s_depth
        ratio_rgb = s_rgb / sum_s if sum_s > 0 else 0.5

        if ratio_rgb > 0.6:
            driving_branch = "RGB (color/texture)"
        elif ratio_rgb < 0.4:
            driving_branch = "Depth (geometry/shape)"
        else:
            driving_branch = "Joint RGB+Depth"

        clip_sims_np = clip_sims.squeeze(0).cpu().numpy()
        top_phrases = ["cracked", "contaminated", "cut", "damaged", "good", "hole", "thread", "discolored"]
        phrase_scores = list(zip(top_phrases, clip_sims_np))
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        top_match = phrase_scores[0]

        confidence = 1.0 - np.exp(-0.15 * image_score)
        confidence = min(max(confidence, 0.0), 1.0)

        peak_y, peak_x = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)

        defect_label = DEFECT_DESCRIPTIONS.get(dtype, dtype)
        print(f"  [{count+1}/{len(indices_to_run)}] Sample {idx:03d} | Type: {defect_label} | Score: {image_score:.3f} | Conf: {confidence:.1%} | Driver: {driving_branch} | CLIP top: '{top_match[0]}' ({top_match[1]:.3f})")

        rgb_disp = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_disp = (rgb_disp * std + mean).clip(0, 1)
        depth_disp = depth_tensor.squeeze(0).squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        axes[0, 0].imshow(rgb_disp)
        axes[0, 0].set_title("Input RGB")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(depth_disp, cmap="jet")
        axes[0, 1].set_title("Input Depth (XYZ)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gt_mask, cmap="gray")
        axes[0, 2].set_title("Ground Truth Mask")
        axes[0, 2].axis("off")

        im3 = axes[0, 3].imshow(anomaly_map, cmap="hot")
        axes[0, 3].set_title(f"Fusion Anomaly Map\nScore: {image_score:.3f}")
        axes[0, 3].axis("off")
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

        axes[1, 0].imshow(rgb_disp)
        axes[1, 0].imshow(rgb_res_up, cmap="jet", alpha=0.5)
        axes[1, 0].set_title(f"RGB Branch Residual\n(s={s_rgb:.3f})")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(depth_disp, cmap="gray")
        axes[1, 1].imshow(depth_res_up, cmap="jet", alpha=0.5)
        axes[1, 1].set_title(f"Depth Branch Residual\n(s={s_depth:.3f})")
        axes[1, 1].axis("off")

        axes[1, 2].barh(
            [p for p, _ in phrase_scores],
            [s for _, s in phrase_scores],
            color=["#2ecc71" if p == top_match[0] else "#95a5a6" for p, _ in phrase_scores]
        )
        axes[1, 2].set_title("CLIP Text Similarity")
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].invert_yaxis()

        summary_text = (
            f"Defect: {defect_label}\n"
            f"Driver: {driving_branch}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Peak: ({peak_x}, {peak_y})\n"
            f"CLIP: '{top_match[0]}'"
        )
        axes[1, 3].text(0.1, 0.5, summary_text, transform=axes[1, 3].transAxes,
                        fontsize=11, verticalalignment="center",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[1, 3].axis("off")
        axes[1, 3].set_title("Decision Summary")

        plt.suptitle(
            f"{category.upper()} | {defect_label} | Sample {idx:03d}",
            fontsize=14, fontweight="bold", y=0.98
        )
        plt.tight_layout()

        out_path = os.path.join("outputs/explainability", f"{category}_{dtype}_sample{idx:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        all_results.append({
            "idx": idx,
            "type": dtype,
            "score": image_score,
            "confidence": confidence,
            "driver": driving_branch,
            "clip_top": top_match[0],
        })

    print(f"\nSaved {len(all_results)} visualizations to outputs/explainability/")

    summary_path = os.path.join("outputs/explainability", f"{category}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Explainability Summary: {category}\n")
        f.write(f"{'='*60}\n")
        for r in all_results:
            f.write(f"Sample {r['idx']:03d} | {r['type']:15s} | Score: {r['score']:.3f} | Conf: {r['confidence']:.1%} | Driver: {r['driver']} | CLIP: {r['clip_top']}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cable_gland")
    parser.add_argument("--all", action="store_true", help="Run all categories")
    parser.add_argument("--num_per_type", type=int, default=2)
    args = parser.parse_args()

    if args.all:
        config = load_config()
        categories = config["dataset"]["categories"]
    else:
        categories = [args.category]

    total = 0
    for cat in categories:
        print(f"\n{'='*60}")
        print(f"  CATEGORY: {cat}")
        print(f"{'='*60}")
        generate_explanations(cat, args.num_per_type)
        total += 1
    print(f"\nDone: {total} categories processed.")
