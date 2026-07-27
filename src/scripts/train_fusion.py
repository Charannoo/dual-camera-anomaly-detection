import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.fusion import CrossAttentionFusion

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(category, epochs=None):
    config = load_config()

    num_epochs = epochs if epochs is not None else config["model"]["num_epochs"]
    lr = config["model"]["lr"]
    weight_decay = config["model"]["weight_decay"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    features_dir = config["cache"]["features_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Loading cached features for {category}...")
    train_cache = os.path.join(features_dir, f"{category}_train.pt")
    if not os.path.exists(train_cache):
        print(f"ERROR: {train_cache} not found. Run extract_features.py first.")
        return

    data = torch.load(train_cache, map_location="cpu")
    rgb_feats = data["rgb_feats"]
    depth_feats = data["depth_feats"]
    text_tokens = data["text_tokens"]
    print(f"  Loaded {rgb_feats.shape[0]} train samples.")

    fusion_model = CrossAttentionFusion()
    fusion_model.train()

    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    print(f"  Fusion params: {sum(p.numel() for p in fusion_model.parameters()):,}")

    print(f"Starting training for {category}...")
    for epoch in range(num_epochs):
        start_time = time.time()

        optimizer.zero_grad()
        outputs = fusion_model(rgb_feats, depth_feats, text_tokens)
        loss_rgb = criterion(outputs["pred_rgb"], outputs["target_rgb"])
        loss_depth = criterion(outputs["pred_depth"], outputs["target_depth"])
        loss = loss_rgb + loss_depth
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {loss.item():.6f} | LR: {current_lr:.2e} | Time: {elapsed:.2f}s")

    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    torch.save({
        "fusion": fusion_model.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}\n")
    return fusion_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cable_gland", help="Category to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    train(args.category, args.epochs)
