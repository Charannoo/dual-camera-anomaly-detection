import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mvtec_3d import MVTec3DADataset
from src.models.rgb_backbone import RGBBackbone
from src.models.depth_backbone import DepthBackbone
from src.models.clip_branch import ClipBranch

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_and_cache(category_filter=None):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = load_config()
    data_dir = config["dataset"]["data_dir"]
    categories = config["dataset"]["categories"]
    img_size = config["dataset"]["img_size"]
    features_dir = config["cache"]["features_dir"]
    os.makedirs(features_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading models...")
    rgb_model = RGBBackbone(config["model"]["rgb_backbone"]).to(device)
    depth_model = DepthBackbone().to(device)
    clip_config = config.get("clip", {})
    clip_model = ClipBranch(
        model_name=clip_config.get("model_name", "openai/clip-vit-base-patch32")
    ).to(device)

    rgb_model.eval()
    depth_model.eval()
    clip_model.eval()

    if category_filter:
        categories = [c for c in categories if c in category_filter]

    for category in categories:
        for split in ["train", "test"]:
            print(f"--- Extracting features for {category} ({split}) ---")
            dataset = MVTec3DADataset(data_dir, category, split, img_size)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            rgb_feats_list = []
            depth_feats_list = []
            text_tokens_list = []
            sims_list = []
            gt_list = []
            label_list = []
            rgb_paths_list = []
            depth_paths_list = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Processing batches"):
                    rgb = batch["rgb"].to(device)
                    depth = batch["depth"].to(device)
                    gt = batch["gt"].to(device)
                    labels = batch["label"].to(device)

                    rgb_feat = rgb_model(rgb)
                    depth_feat = depth_model(depth)
                    text_token, sims = clip_model(rgb)

                    rgb_feats_list.append(rgb_feat.cpu())
                    depth_feats_list.append(depth_feat.cpu())
                    text_tokens_list.append(text_token.cpu())
                    sims_list.append(sims.cpu())
                    gt_list.append(gt.cpu())
                    label_list.append(labels.cpu())
                    rgb_paths_list.extend(batch["rgb_path"])
                    depth_paths_list.extend(batch["depth_path"])

            rgb_feats = torch.cat(rgb_feats_list, dim=0)
            depth_feats = torch.cat(depth_feats_list, dim=0)
            text_tokens = torch.cat(text_tokens_list, dim=0)
            sims = torch.cat(sims_list, dim=0)
            gts = torch.cat(gt_list, dim=0)
            labels = torch.cat(label_list, dim=0)

            cache_dict = {
                "rgb_feats": rgb_feats,
                "depth_feats": depth_feats,
                "text_tokens": text_tokens,
                "clip_sims": sims,
                "gts": gts,
                "labels": labels,
                "rgb_paths": rgb_paths_list,
                "depth_paths": depth_paths_list,
            }

            cache_path = os.path.join(features_dir, f"{category}_{split}.pt")
            torch.save(cache_dict, cache_path)
            print(f"Saved cached features to {cache_path}")
            print(f"  RGB shape: {rgb_feats.shape}")
            print(f"  Depth shape: {depth_feats.shape}")
            print(f"  Text token shape: {text_tokens.shape}")
            print(f"  CLIP sims shape: {sims.shape}")

if __name__ == "__main__":
    extract_and_cache()
