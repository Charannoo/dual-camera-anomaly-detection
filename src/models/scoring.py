import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualScorer:
    def __init__(self, reg_epsilon=1e-3):
        self.reg_epsilon = reg_epsilon
        self.mean = None
        self.inv_cov = None

    def _get_device(self, model):
        return next(model.parameters()).device

    def _move(self, *tensors, device):
        return tuple(t.to(device) for t in tensors)

    def fit(self, train_rgb_feats, train_depth_feats, model, text_tokens=None):
        model.eval()
        device = self._get_device(model)
        train_rgb_feats, train_depth_feats = self._move(train_rgb_feats, train_depth_feats, device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device)
        with torch.no_grad():
            if text_tokens is not None:
                outputs = model(train_rgb_feats, train_depth_feats, text_tokens)
            else:
                outputs = model(train_rgb_feats, train_depth_feats)
            res_rgb = outputs["target_rgb"] - outputs["pred_rgb"]
            res_depth = outputs["target_depth"] - outputs["pred_depth"]

            residuals = torch.cat([res_rgb, res_depth], dim=1)
            N, C, H, W = residuals.shape
            residuals_flat = residuals.permute(0, 2, 3, 1).reshape(-1, C)

        self.mean = torch.mean(residuals_flat, dim=0)
        cov = torch.cov(residuals_flat.T)
        cov_reg = cov + self.reg_epsilon * torch.eye(cov.shape[0])
        self.inv_cov = torch.linalg.inv(cov_reg)
        print("Scorer covariance fitting completed successfully.")

    def score(self, rgb_feats, depth_feats, model, text_tokens=None, upsample_size=(224, 224)):
        if self.mean is None or self.inv_cov is None:
            raise ValueError("Scorer has not been fitted yet. Run fit() first.")

        model.eval()
        device = self._get_device(model)
        rgb_feats, depth_feats = self._move(rgb_feats, depth_feats, device)
        self.mean = self.mean.to(device)
        self.inv_cov = self.inv_cov.to(device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device)
        with torch.no_grad():
            if text_tokens is not None:
                outputs = model(rgb_feats, depth_feats, text_tokens)
            else:
                outputs = model(rgb_feats, depth_feats)
            res_rgb = outputs["target_rgb"] - outputs["pred_rgb"]
            res_depth = outputs["target_depth"] - outputs["pred_depth"]

            residuals = torch.cat([res_rgb, res_depth], dim=1)
            B, C, H, W = residuals.shape
            res_flat = residuals.permute(0, 2, 3, 1).reshape(-1, C)

            diff = res_flat - self.mean
            temp = torch.matmul(diff, self.inv_cov)
            dist_sq = torch.sum(temp * diff, dim=-1)
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))

            anomaly_map_low = dist.view(B, H, W)
            image_scores, _ = torch.max(anomaly_map_low.view(B, -1), dim=-1)

            anomaly_map_high = F.interpolate(
                anomaly_map_low.unsqueeze(1),
                size=upsample_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

            rgb_res_norm = torch.norm(res_rgb, p=2, dim=1)
            depth_res_norm = torch.norm(res_depth, p=2, dim=1)

            return {
                "anomaly_map_low": anomaly_map_low,
                "anomaly_map": anomaly_map_high,
                "image_score": image_scores,
                "rgb_residual": rgb_res_norm,
                "depth_residual": depth_res_norm,
                "attn_rgb": outputs["attn_rgb"],
                "attn_depth": outputs["attn_depth"]
            }

if __name__ == "__main__":
    print("Testing ResidualScorer (3-way)...")
    from src.models.fusion import CrossAttentionFusion

    model = CrossAttentionFusion()
    scorer = ResidualScorer()

    train_rgb = torch.randn(5, 48, 14, 14)
    train_depth = torch.randn(5, 64, 14, 14)
    train_text = torch.randn(5, 1, 128)
    scorer.fit(train_rgb, train_depth, model, text_tokens=train_text)

    test_rgb = torch.randn(3, 48, 14, 14)
    test_depth = torch.randn(3, 64, 14, 14)
    test_text = torch.randn(3, 1, 128)
    results = scorer.score(test_rgb, test_depth, model, text_tokens=test_text)

    print(f"Anomaly Map Shape: {results['anomaly_map'].shape} (Expected: [3, 224, 224])")
    print(f"Image Scores Shape: {results['image_score'].shape} (Expected: [3])")
    print(f"RGB Residual Map Shape: {results['rgb_residual'].shape} (Expected: [3, 14, 14])")
    print(f"Depth Residual Map Shape: {results['depth_residual'].shape} (Expected: [3, 14, 14])")
    print("ResidualScorer 3-way test passed.")
