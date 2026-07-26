import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import tifffile

class MVTec3DADataset(Dataset):
    def __init__(self, data_dir, category, split="train", img_size=224):
        self.data_dir = data_dir
        self.category = category
        self.split = split
        self.img_size = img_size
        
        self.rgb_paths = []
        self.depth_paths = []
        self.gt_paths = []
        self.labels = [] # 0 for normal, 1 for abnormal
        
        self._load_dataset()
        
    def _load_dataset(self):
        cat_path = os.path.join(self.data_dir, self.category)
        
        if self.split == "train":
            # Train only contains good samples
            rgb_dir = os.path.join(cat_path, "train", "good", "rgb")
            xyz_dir = os.path.join(cat_path, "train", "good", "xyz")
            
            rgbs = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
            xyzs = sorted(glob.glob(os.path.join(xyz_dir, "*.tiff")))
            
            for r, x in zip(rgbs, xyzs):
                self.rgb_paths.append(r)
                self.depth_paths.append(x)
                self.gt_paths.append(None)
                self.labels.append(0)
        else:
            # Test split contains both good and defective classes
            test_dir = os.path.join(cat_path, "test")
            class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            
            for cls in class_dirs:
                rgb_dir = os.path.join(test_dir, cls, "rgb")
                xyz_dir = os.path.join(test_dir, cls, "xyz")
                gt_dir = os.path.join(test_dir, cls, "gt")
                
                rgbs = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                xyzs = sorted(glob.glob(os.path.join(xyz_dir, "*.tiff")))
                
                label = 0 if cls == "good" else 1
                
                for i in range(len(rgbs)):
                    self.rgb_paths.append(rgbs[i])
                    self.depth_paths.append(xyzs[i])
                    self.labels.append(label)
                    
                    if label == 1:
                        # Find corresponding gt mask
                        base_name = os.path.basename(rgbs[i])
                        gt_path = os.path.join(gt_dir, base_name)
                        if os.path.exists(gt_path):
                            self.gt_paths.append(gt_path)
                        else:
                            # Fallback if names differ slightly
                            gt_candidates = glob.glob(os.path.join(gt_dir, "*.png"))
                            if len(gt_candidates) > i:
                                self.gt_paths.append(gt_candidates[i])
                            else:
                                self.gt_paths.append(None)
                    else:
                        self.gt_paths.append(None)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]
        
        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = rgb.resize((self.img_size, self.img_size), Image.BILINEAR)
        rgb_arr = np.array(rgb).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_arr = (rgb_arr - mean) / std
        rgb_tensor = torch.tensor(rgb_arr).permute(2, 0, 1) # (3, H, W)
        
        # Load Depth
        # XYZ point cloud: shape (H, W, 3)
        xyz = tifffile.imread(depth_path)
        # Extract Z channel
        z_channel = xyz[:, :, 2]
        z_channel = np.nan_to_num(z_channel, nan=0.0)
        
        # Resize depth using OpenCV
        z_resized = cv2.resize(z_channel, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize depth
        min_val, max_val = z_resized.min(), z_resized.max()
        if max_val - min_val > 1e-6:
            z_normalized = (z_resized - min_val) / (max_val - min_val)
        else:
            z_normalized = np.zeros_like(z_resized)
            
        depth_tensor = torch.tensor(z_normalized).unsqueeze(0).float() # (1, H, W)
        
        # Load Ground Truth Mask
        if gt_path is not None:
            gt = Image.open(gt_path).convert("L")
            gt = gt.resize((self.img_size, self.img_size), Image.NEAREST)
            gt_arr = np.array(gt)
            gt_binary = (gt_arr > 127).astype(np.float32)
            gt_tensor = torch.tensor(gt_binary).unsqueeze(0) # (1, H, W)
        else:
            gt_tensor = torch.zeros((1, self.img_size, self.img_size))
            
        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "gt": gt_tensor,
            "label": label,
            "rgb_path": rgb_path,
            "depth_path": depth_path
        }


def generate_mock_dataset(data_dir="data", num_train=10, num_test=5):
    """Generates a small synthetic MVTec 3D-AD style dataset for testing."""
    for category in ["cable_gland", "foam", "cookie", "potato"]:
        # Paths
        train_rgb = os.path.join(data_dir, category, "train", "good", "rgb")
        train_xyz = os.path.join(data_dir, category, "train", "good", "xyz")
        test_good_rgb = os.path.join(data_dir, category, "test", "good", "rgb")
        test_good_xyz = os.path.join(data_dir, category, "test", "good", "xyz")
        test_def_rgb = os.path.join(data_dir, category, "test", "crack", "rgb")
        test_def_xyz = os.path.join(data_dir, category, "test", "crack", "xyz")
        test_def_gt = os.path.join(data_dir, category, "test", "crack", "gt")
        
        for d in [train_rgb, train_xyz, test_good_rgb, test_good_xyz, test_def_rgb, test_def_xyz, test_def_gt]:
            os.makedirs(d, exist_ok=True)
            
        # Helper to write XYZ tiff
        def save_dummy_xyz(path, defect_region=None):
            # Create a simple dome shape or plane
            y, x = np.mgrid[0:256, 0:256]
            z = 100.0 - ((x - 128)**2 + (y - 128)**2) / 500.0
            if defect_region is not None:
                # Add a dent in depth
                mask = np.zeros((256, 256), dtype=bool)
                cv2.rectangle(mask.view(np.uint8), defect_region[0], defect_region[1], 1, -1)
                z[mask] -= 15.0
            # Construct XYZ array
            xyz = np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=-1)
            tifffile.imwrite(path, xyz)
            
        # Helper to write RGB image
        def save_dummy_rgb(path, defect_region=None):
            # Normal background
            img = np.ones((256, 256, 3), dtype=np.uint8) * 180
            # Add some texture/grid
            cv2.line(img, (0, 128), (256, 128), (150, 150, 150), 2)
            cv2.line(img, (128, 0), (128, 256), (150, 150, 150), 2)
            if defect_region is not None:
                # Add a dark scratch/defect
                cv2.rectangle(img, defect_region[0], defect_region[1], (50, 50, 50), -1)
            Image.fromarray(img).save(path)

        # Generate train
        for i in range(num_train):
            save_dummy_rgb(os.path.join(train_rgb, f"{i:03d}.png"))
            save_dummy_xyz(os.path.join(train_xyz, f"{i:03d}.tiff"))
            
        # Generate test good
        for i in range(num_test):
            save_dummy_rgb(os.path.join(test_good_rgb, f"{i:03d}.png"))
            save_dummy_xyz(os.path.join(test_good_xyz, f"{i:03d}.tiff"))
            
        # Generate test defective (with defect)
        for i in range(num_test):
            defect_start = (64 + i * 10, 64 + i * 10)
            defect_end = (96 + i * 10, 96 + i * 10)
            
            rgb_path = os.path.join(test_def_rgb, f"{i:03d}.png")
            xyz_path = os.path.join(test_def_xyz, f"{i:03d}.tiff")
            gt_path = os.path.join(test_def_gt, f"{i:03d}.png")
            
            save_dummy_rgb(rgb_path, (defect_start, defect_end))
            save_dummy_xyz(xyz_path, (defect_start, defect_end))
            
            # Save GT mask
            gt = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(gt, defect_start, defect_end, 255, -1)
            Image.fromarray(gt).save(gt_path)

        print(f"Generated mock data for category: {category}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Running Dataset Smoke Test...")
    # Generate mock dataset
    generate_mock_dataset(num_train=5, num_test=3)
    
    # Load dataset
    ds = MVTec3DADataset(data_dir="data", category="cable_gland", split="test", img_size=224)
    print(f"Loaded {len(ds)} test samples.")
    
    sample = ds[0] # A defective sample (index 0 is defective because 'crack' sorts before 'good')
    rgb = sample["rgb"]
    depth = sample["depth"]
    gt = sample["gt"]
    label = sample["label"]
    
    print(f"RGB Shape: {rgb.shape}, Mean: {rgb.mean():.4f}")
    print(f"Depth Shape: {depth.shape}, Mean: {depth.mean():.4f}")
    print(f"GT Mask Shape: {gt.shape}, Max: {gt.max():.4f}")
    print(f"Label: {label} (Expected 1 for defective)")
    
    # Save a verification visualization
    # Convert tensor back for display
    rgb_disp = rgb.permute(1, 2, 0).numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb_disp = (rgb_disp * std + mean).clip(0, 1)
    
    depth_disp = depth[0].numpy()
    gt_disp = gt[0].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(rgb_disp)
    axes[0].set_title("RGB Image")
    axes[1].imshow(depth_disp, cmap="jet")
    axes[1].set_title("Depth Map")
    axes[2].imshow(gt_disp, cmap="gray")
    axes[2].set_title("GT Defect Mask")
    
    os.makedirs("outputs", exist_ok=True)
    out_img_path = "outputs/stage1_smoke_test.png"
    plt.tight_layout()
    plt.savefig(out_img_path)
    print(f"Smoke test visualization saved to {out_img_path}")
