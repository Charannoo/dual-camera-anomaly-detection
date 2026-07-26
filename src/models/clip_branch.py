import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PHRASES = [
    "a photo of a normal intact object",
    "a photo of a cracked or broken object",
    "a photo of a scratched surface defect",
    "a photo of a dent or deformation",
    "a photo of a discolored or stained area",
    "a photo of a missing part or hole",
    "a photo of a normal clean surface",
    "a photo of a defective damaged object",
]


class ClipBranch(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", phrases=None):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.model_name = model_name
        self.phrases = phrases or DEFAULT_PHRASES

        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        image_embed_dim = self.clip.config.projection_dim  # 512 (post-projection output of get_image_features)
        text_embed_dim = self.clip.config.projection_dim   # 512 (post-projection output of get_text_features)

        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self._phrase_embeddings = None

    @torch.no_grad()
    def _encode_phrases(self, device):
        if self._phrase_embeddings is not None and self._phrase_embeddings.device == device:
            return self._phrase_embeddings
        tokens = self.tokenizer(
            self.phrases, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        text_features = self.clip.get_text_features(**tokens)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        text_features = F.normalize(text_features, dim=-1)
        self._phrase_embeddings = text_features
        return self._phrase_embeddings

    def forward(self, rgb_image):
        device = rgb_image.device
        phrase_emb = self._encode_phrases(device)

        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=rgb_image)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            image_features = F.normalize(image_features, dim=-1)

        image_proj = self.image_proj(image_features)   # (B, 128)
        phrase_emb_proj = self.text_proj(phrase_emb)    # (num_phrases, 128)

        sims = torch.matmul(image_proj, phrase_emb_proj.T)  # (B, num_phrases)
        sims = sims / sims.norm(dim=-1, keepdim=True)

        weighted = torch.matmul(sims, phrase_emb_proj)  # (B, 128)
        text_token = weighted.unsqueeze(1)               # (B, 1, 128)

        return text_token, sims


if __name__ == "__main__":
    print("Testing ClipBranch...")
    model = ClipBranch()
    rgb = torch.randn(2, 3, 224, 224)
    text_token, sims = model(rgb)
    print(f"RGB input shape: {rgb.shape}")
    print(f"Text token shape: {text_token.shape} (Expected: [2, 1, 128])")
    print(f"Similarity shape: {sims.shape} (Expected: [2, 8])")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    print("ClipBranch test passed.")
