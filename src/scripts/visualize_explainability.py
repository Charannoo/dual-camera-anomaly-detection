import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.fusion import CrossAttentionFusion
from src.models.scoring import ResidualScorer

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_explanations(category="cable_gland", num_visualize=3):
    config = load_config()
    data_dir = config["dataset"]["data_dir"]
    img_size = config["dataset"]["img_size"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    features_dir = config["cache"]["features_dir"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load backbones and fusion model
    rgb_backbone = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_backbone = DepthBackbone().to(device)
    fusion_model = CrossAttentionFusion().to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    if os.path.exists(checkpoint_path):
        fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded fusion model weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    rgb_backbone.eval()
    depth_backbone.eval()
    fusion_model.eval()
    
    # Load cached train features to fit the scorer
    train_path = os.path.join(features_dir, f"{category}_train.pt")
    train_data = torch.load(train_path)
    
    scorer = ResidualScorer()
    scorer.fit(train_data["rgb_feats"], train_data["depth_feats"], fusion_model)
    
    # Load raw dataset for visual rendering
    dataset = MVTec3DADataset(data_dir, category, split="test", img_size=img_size)
    
    # Find defective indices
    defective_indices = [i for i, lbl in enumerate(dataset.labels) if lbl == 1]
    indices_to_run = defective_indices[:num_visualize]
    
    print(f"\nGenerating explainability outputs for {category} on {len(indices_to_run)} defective samples...")
    
    os.makedirs("outputs", exist_ok=True)
    
    for count, idx in enumerate(indices_to_run):
        sample = dataset[idx]
        rgb_tensor = sample["rgb"].unsqueeze(0).to(device)
        depth_tensor = sample["depth"].unsqueeze(0).to(device)
        gt_mask = sample["gt"].squeeze(0).numpy() # (224, 224)
        
        # 1. End-to-end forward pass
        with torch.no_grad():
            rgb_feat = rgb_backbone(rgb_tensor)
            depth_feat = depth_backbone(depth_tensor)
            results = scorer.score(rgb_feat, depth_feat, fusion_model)
            
        anomaly_map = results["anomaly_map"].squeeze(0).numpy() # (224, 224)
        image_score = results["image_score"].item()
        
        # Get low-res residual maps for localization
        rgb_res = results["rgb_residual"].squeeze(0)     # (14, 14)
        depth_res = results["depth_residual"].squeeze(0) # (14, 14)
        
        # Upsample residual maps for per-branch visualization
        rgb_res_up = F.interpolate(rgb_res.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze().numpy()
        depth_res_up = F.interpolate(depth_res.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze().numpy()
        
        # 2. Rule-based explanation generation
        s_rgb = rgb_res.max().item()
        s_depth = depth_res.max().item()
        sum_s = s_rgb + s_depth
        ratio_rgb = s_rgb / sum_s if sum_s > 0 else 0.5
        
        # Identify driving branch
        if ratio_rgb > 0.6:
            driving_branch = "RGB camera branch (color/texture defect)"
        elif ratio_rgb < 0.4:
            driving_branch = "Depth sensor branch (geometry/shape defect)"
        else:
            driving_branch = "both RGB and Depth sensor branches (joint defect)"
            
        # Localize anomaly peak coordinate
        peak_y_idx, peak_x_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        # Convert to relative image coordinates (percent or pixels)
        loc_x_px = int(peak_x_idx)
        loc_y_px = int(peak_y_idx)
        
        # Normalized confidence: Sigmoid-like scaling
        # (Assuming normal threshold is around 15.0, confidence scales up rapidly beyond it)
        confidence = 1.0 - np.exp(-0.015 * image_score)
        confidence = min(max(confidence, 0.0), 1.0)
        
        explanation_text = (
            f"Sample {idx:02d} | Decision: ANOMALOUS\n"
            f"  - Driven by: {driving_branch}\n"
            f"  - Primary location: x={loc_x_px}, y={loc_y_px} (pixel coordinates)\n"
            f"  - Raw Anomaly Score: {image_score:.2f} | Confidence: {confidence:.2%}\n"
        )
        print(explanation_text)
        
        # 3. Plotting and saving visualizations
        # Denormalize RGB for plotting
        rgb_disp = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_disp = (rgb_disp * std + mean).clip(0, 1)
        
        depth_disp = depth_tensor.squeeze(0).squeeze(0).numpy()
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
        axes[0].imshow(rgb_disp)
        axes[0].set_title("1. Input RGB")
        axes[0].axis("off")
        
        axes[1].imshow(depth_disp, cmap="jet")
        axes[1].set_title("2. Input Depth")
        axes[1].axis("off")
        
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("3. Ground Truth")
        axes[2].axis("off")
        
        # RGB Heatmap
        axes[3].imshow(rgb_disp)
        im3 = axes[3].imshow(rgb_res_up, cmap="jet", alpha=0.5)
        axes[3].set_title("4. RGB-driven Heatmap")
        axes[3].axis("off")
        
        # Depth Heatmap
        axes[4].imshow(depth_disp, cmap="gray")
        im4 = axes[4].imshow(depth_res_up, cmap="jet", alpha=0.5)
        axes[4].set_title("5. Depth-driven Heatmap")
        axes[4].axis("off")
        
        plt.suptitle(f"Explainability Analysis - Sample {idx:02d} (Score: {image_score:.2f})", fontsize=12, y=0.98)
        plt.tight_layout()
        
        out_path = os.path.join("outputs", f"explainability_{category}_sample{idx:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Explainability plot saved to {out_path}\n")

if __name__ == "__main__":
    generate_explanations("cable_gland", num_visualize=3)
