import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.fusion import CrossAttentionFusion

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(category, epochs=None):
    config = load_config()

    num_epochs = epochs if epochs is not None else config["model"]["num_epochs"]
    batch_size = config["model"]["batch_size"]
    lr = config["model"]["lr"]
    weight_decay = config["model"]["weight_decay"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    features_dir = config["cache"]["features_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    cache_path = os.path.join(features_dir, f"{category}_train.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached features not found at {cache_path}. Run extract_features.py first.")

    print(f"Loading cached training features from {cache_path}...")
    cached_data = torch.load(cache_path)
    rgb_feats = cached_data["rgb_feats"]
    depth_feats = cached_data["depth_feats"]
    text_tokens = cached_data["text_tokens"]

    print(f"Loaded {len(rgb_feats)} training samples.")

    dataset = TensorDataset(rgb_feats, depth_feats, text_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CrossAttentionFusion().to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    print(f"Starting training for {category} on CPU...")
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0

        for batch_rgb, batch_depth, batch_text in dataloader:
            batch_rgb = batch_rgb.to(device)
            batch_depth = batch_depth.to(device)
            batch_text = batch_text.to(device)

            optimizer.zero_grad()

            outputs = model(batch_rgb, batch_depth, batch_text)

            loss_rgb = criterion(outputs["pred_rgb"], outputs["target_rgb"])
            loss_depth = criterion(outputs["pred_depth"], outputs["target_depth"])
            loss = loss_rgb + loss_depth

            if torch.isnan(loss):
                print("WARNING: NaN loss detected!")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_rgb)

        epoch_loss /= len(dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")

    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_fusion.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}\n")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cable_gland", help="Category to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    args = parser.parse_args()

    train(args.category, args.epochs)
