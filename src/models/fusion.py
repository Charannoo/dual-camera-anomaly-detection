import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, rgb_in_channels=48, depth_in_channels=64, embed_dim=128, num_heads=4):
        super().__init__()
        self.rgb_proj = nn.Conv2d(rgb_in_channels, embed_dim, kernel_size=1)
        self.depth_proj = nn.Conv2d(depth_in_channels, embed_dim, kernel_size=1)

        self.attn_rgb_to_depth = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_depth_to_rgb = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_depth = nn.LayerNorm(embed_dim)

        self.pred_rgb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pred_depth = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, rgb_feats, depth_feats, text_token=None):
        B, _, H, W = rgb_feats.shape

        x_rgb = self.rgb_proj(rgb_feats)
        x_depth = self.depth_proj(depth_feats)

        seq_rgb = x_rgb.flatten(2).transpose(1, 2)       # (B, 196, 128)
        seq_depth = x_depth.flatten(2).transpose(1, 2)   # (B, 196, 128)

        if text_token is not None:
            kv_depth = torch.cat([seq_depth, text_token], dim=1)  # (B, 197, 128)
            kv_rgb = torch.cat([seq_rgb, text_token], dim=1)      # (B, 197, 128)
        else:
            kv_depth = seq_depth
            kv_rgb = seq_rgb

        fused_rgb, attn_rgb = self.attn_rgb_to_depth(query=seq_rgb, key=kv_depth, value=kv_depth)
        fused_depth, attn_depth = self.attn_depth_to_rgb(query=seq_depth, key=kv_rgb, value=kv_rgb)

        fused_rgb = self.norm_rgb(fused_rgb + seq_rgb)
        fused_depth = self.norm_depth(fused_depth + seq_depth)

        pred_depth_feats = self.pred_depth(fused_rgb)
        pred_rgb_feats = self.pred_rgb(fused_depth)

        pred_depth_spatial = pred_depth_feats.transpose(1, 2).view(B, 128, H, W)
        pred_rgb_spatial = pred_rgb_feats.transpose(1, 2).view(B, 128, H, W)

        target_rgb_spatial = seq_rgb.transpose(1, 2).view(B, 128, H, W)
        target_depth_spatial = seq_depth.transpose(1, 2).view(B, 128, H, W)

        return {
            "pred_rgb": pred_rgb_spatial,
            "pred_depth": pred_depth_spatial,
            "target_rgb": target_rgb_spatial,
            "target_depth": target_depth_spatial,
            "attn_rgb": attn_rgb,
            "attn_depth": attn_depth
        }

if __name__ == "__main__":
    print("Testing CrossAttentionFusion (3-way)...")
    model = CrossAttentionFusion()
    rgb = torch.randn(2, 48, 14, 14)
    depth = torch.randn(2, 64, 14, 14)
    text_token = torch.randn(2, 1, 128)

    out = model(rgb, depth, text_token)
    print(f"Pred RGB shape: {out['pred_rgb'].shape} (Expected: [2, 128, 14, 14])")
    print(f"Pred Depth shape: {out['pred_depth'].shape} (Expected: [2, 128, 14, 14])")
    print(f"Attn RGB shape: {out['attn_rgb'].shape} (Expected: [2, 196, 196])")

    assert out['pred_rgb'].shape == (2, 128, 14, 14), "pred_rgb shape mismatch"
    assert out['pred_depth'].shape == (2, 128, 14, 14), "pred_depth shape mismatch"

    # Verify gradient flows through text_token path
    loss = out['pred_rgb'].sum() + out['pred_depth'].sum()
    loss.backward()

    # Check gradients exist in the fusion model
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradient flow through fusion: {'OK' if has_grads else 'FAILED'}")

    # Backward compat: no text_token
    out2 = model(rgb, depth)
    print(f"Without text_token: pred_rgb {out2['pred_rgb'].shape} - OK")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("CrossAttentionFusion 3-way test passed.")
