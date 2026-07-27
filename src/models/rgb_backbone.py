import torch
import torch.nn as nn
import timm

class RGBBackbone(nn.Module):
    def __init__(self, model_name="mobilenetv3_small_100"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def forward(self, x):
        if not any(p.requires_grad for p in self.model.parameters()):
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)
        stage3_features = features[3]
        return stage3_features

if __name__ == "__main__":
    print("Testing RGBBackbone...")
    model = RGBBackbone()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output feature shape: {out.shape} (Expected: [2, 48, 14, 14])")
    print("RGBBackbone test passed.")
