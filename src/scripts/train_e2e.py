import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.clip_branch import ClipBranch
from src.models.fusion import CrossAttentionFusion

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class EndToEndModel(nn.Module):
    def __init__(self, rgb_backbone, depth_backbone, clip_branch, fusion):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.clip_branch = clip_branch
        self.fusion = fusion

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_backbone(rgb)
        depth_feat = self.depth_backbone(depth)
        text_token, clip_sims = self.clip_branch(rgb)
        outputs = self.fusion(rgb_feat, depth_feat, text_token)
        return outputs, clip_sims


def train_e2e(category, epochs=None, lr_backbone=None, lr_fusion=None):
    config = load_config()
    data_dir = config["dataset"]["data_dir"]
    img_size = config["dataset"]["img_size"]
    num_epochs = epochs if epochs else config["model"]["num_epochs"]
    batch_size = config["model"]["batch_size"]
    weight_decay = config["model"]["weight_decay"]
    checkpoint_dir = config["model"]["checkpoint_dir"]

    lr_backbone = lr_backbone if lr_backbone else 1e-5
    lr_fusion = lr_fusion if lr_fusion else config["model"]["lr"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MVTec3DADataset(data_dir, category, split="train", img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Training samples: {len(train_dataset)}")

    rgb_backbone = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_backbone = DepthBackbone().to(device)
    clip_config = config.get("clip", {})
    clip_branch = ClipBranch(
        model_name=clip_config.get("model_name", "openai/clip-vit-base-patch32")
    ).to(device)
    fusion = CrossAttentionFusion().to(device)

    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    if os.path.exists(checkpoint_path):
        fusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded pretrained fusion checkpoint from {checkpoint_path}")

    rgb_backbone.unfreeze()

    model = EndToEndModel(rgb_backbone, depth_backbone, clip_branch, fusion).to(device)

    backbone_params = list(rgb_backbone.model.parameters()) + list(depth_backbone.parameters())
    fusion_params = list(fusion.parameters())
    clip_params = list(clip_branch.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": fusion_params, "lr": lr_fusion},
        {"params": clip_params, "lr": lr_fusion * 0.1},
    ], weight_decay=weight_decay)

    total_epochs = num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)
    criterion = nn.MSELoss()

    print(f"End-to-end training: {category} for {total_epochs} epochs")
    print(f"  Backbone LR: {lr_backbone:.2e} | Fusion LR: {lr_fusion:.2e}")

    for epoch in range(total_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        model.train()

        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()

            outputs, _ = model(rgb, depth)

            loss_rgb = criterion(outputs["pred_rgb"], outputs["target_rgb"])
            loss_depth = criterion(outputs["pred_depth"], outputs["target_depth"])
            loss = loss_rgb + loss_depth

            if torch.isnan(loss):
                print("WARNING: NaN loss!")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(rgb)

        scheduler.step()
        epoch_loss /= len(train_dataset)
        elapsed = time.time() - start_time
        lr0 = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1:02d}/{total_epochs:02d} | Loss: {epoch_loss:.6f} | LR: {lr0:.2e} | Time: {elapsed:.1f}s")

    full_path = os.path.join(checkpoint_dir, f"{category}_e2e.pt")
    torch.save({
        "fusion": fusion.state_dict(),
        "depth_backbone": depth_backbone.state_dict(),
        "rgb_backbone": rgb_backbone.model.state_dict(),
    }, full_path)
    print(f"Saved E2E checkpoint to {full_path}")

    torch.save(fusion.state_dict(), checkpoint_path)
    print(f"Updated fusion checkpoint: {checkpoint_path}\n")

    return fusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cable_gland")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-fusion", type=float, default=None)
    args = parser.parse_args()
    train_e2e(args.category, args.epochs, args.lr_backbone, args.lr_fusion)
