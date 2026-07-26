import torch
import torch.nn as nn

class DepthBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.early_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.early_conv(x)

if __name__ == "__main__":
    print("Testing DepthBackbone (trainable)...")
    model = DepthBackbone()
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output feature shape: {out.shape} (Expected: [2, 64, 14, 14])")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    assert trainable == total, "All parameters should be trainable"
    print("DepthBackbone test passed.")
