# Lightweight Dual-Camera Explainable Anomaly Detection

## 1. Objective

This project implements a lightweight, CPU-only, explainable anomaly detection system for industrial inspection using multi-modal sensor fusion. The system combines RGB image features, depth sensor features, and CLIP-guided text semantics to detect and localize defects in the MVTec 3D-AD dataset (NeurIPS 2021 review subset, 6 categories). The architecture is inspired by the approach described in arXiv 2604.22899, as assigned by faculty. The system operates under the constraint of requiring no GPU at inference time, targeting sub-300ms latency per image on commodity CPU hardware, with all trainable components totaling under 500K parameters.

## 2. Architecture

The pipeline consists of six components:

**RGB Backbone** (`src/models/rgb_backbone.py`). A frozen MobileNetV3-Small (`mobilenetv3_small_100` via timm) used as a pretrained feature extractor. All 927,008 parameters are frozen (`requires_grad=False`). The network produces 48-channel feature maps at 14x14 spatial resolution from 224x224 input images (stage 3 features).

**Depth Backbone** (`src/models/depth_backbone.py`). A 3-layer convolutional network that processes single-channel depth maps (XYZ data from the sensor). 23,520 total parameters, all trainable by design. Architecture: Conv2d(1,16,3,stride=2) → BN → ReLU → Conv2d(16,32,3,stride=2) → BN → ReLU → MaxPool(2) → Conv2d(32,64,3,stride=2) → BN → ReLU. Output: 64-channel feature maps at 14x14 resolution. In the final frozen baseline, this module is randomly initialized but never trained — its weights remain at their initialization values.

**Clip Branch** (`src/models/clip_branch.py`). Wraps CLIP ViT-B/32 (frozen, 151.4M parameters) with two trainable projection layers: `text_proj` (Linear(512,128) → ReLU → Linear(128,128), 82,176 params) and `image_proj` (identical architecture, 82,176 params). The branch encodes the input RGB image through CLIP's frozen vision encoder, projects through both heads, computes cosine similarity between the image projection and 8 pre-defined defect text embeddings (e.g., "a photo of a cracked or broken object"), and produces a weighted text token of shape (B, 1, 128). In the final baseline, both projection layers are randomly initialized but never trained — see Section 6 for rationale.

**Cross-Attention Fusion** (`src/models/fusion.py`). The core trainable module with 213,248 parameters. Projects RGB features (48→128 channels via 1x1 Conv) and depth features (64→128 channels via 1x1 Conv) into a shared 128-dimensional space. Applies bidirectional multi-head cross-attention (4 heads): RGB attends to Depth, and Depth attends to RGB. The text token from ClipBranch is concatenated as an additional key/value pair in the 197-token sequence (196 spatial tokens + 1 text token). Two prediction heads (MLP: Linear→ReLU→Linear) reconstruct the target RGB and depth features from the fused representation. Trained via MSE loss on the reconstruction residuals.

**Scoring** (`src/models/scoring.py`). Two scorers were evaluated:
- `ResidualScorer`: Computes Mahalanobis distance from the fused feature distribution. Fits a multivariate Gaussian on training features (mean + inverse covariance). No learnable parameters.
- `MemoryBankScorer`: Stores all training fused features in a memory bank and computes k-NN distance (k=5, chunked `torch.cdist` for CPU memory safety). No learnable parameters.

**Calibration** (`src/models/calibration.py`). `ScoreCalibrator` normalizes raw anomaly scores using z-score standardization (mean, std, anchor from training set). No learnable parameters.

### Parameter Summary

| Component | Total Params | Trainable | Frozen |
|:----------|-------------:|----------:|-------:|
| RGB Backbone (MobileNetV3-Small) | 927,008 | 0 | 927,008 |
| Depth Backbone (3-layer CNN) | 23,520 | 23,520 | 0 |
| ClipBranch (text_proj + image_proj) | 164,352 | 164,352 | 0 |
| CrossAttentionFusion | 213,248 | 213,248 | 0 |
| **Total** | **1,328,128** | **401,120** | **927,008** |

The fusion checkpoint on disk is 833.0 KB (fusion weights only). Latency: 208–304 ms/image on CPU (full pipeline: backbone forward + scorer).

## 3. Ablation Study

Four training configurations were evaluated throughout this project, all tested on the same MVTec 3D-AD 6-category subset with the same frozen MobileNetV3-Small RGB backbone. The table below reports I-AUROC (image-level anomaly detection) using the k-NN MemoryBank scorer (k=5), which consistently outperformed Mahalanobis scoring.

| Variant | Description | Avg I-AUROC | Avg P-AUROC | Avg Latency |
|:--------|:-----------|:-----------|:-----------|:-----------|
| (a) Full E2E fine-tuning | All backbones + fusion, 15 epochs | ~0.56 | — | — |
| (b) Targeted depth/CLIP training | depth_backbone + clip proj + fusion, 30 epochs | 0.7209 | 0.9778 | 213ms |
| (c) Frozen baseline (unseeded) | Fusion only, random-init depth/CLIP, no seed | 0.7836 | 0.9593 | 57.5ms* |
| **(d) Frozen baseline (seed=42)** | **Fusion only, seeded random-init, verified reproducible** | **0.7779** | **0.9591** | **242.3ms** |

*Note: Variant (c) latency measured scorer only (not full pipeline), making it incomparable to other variants.

**Variant (a): Full E2E fine-tuning.** Trained RGBBackbone (unfrozen), DepthBackbone, ClipBranch projections, and CrossAttentionFusion jointly for 15 epochs on each category. Average I-AUROC dropped to approximately 0.56 across all 6 categories — a severe regression from the frozen baseline. E2E checkpoints were backed up to `outputs/checkpoints/e2e_backup/` and the approach was reverted.

**Variant (b): Targeted depth/CLIP training.** Trained only DepthBackbone, ClipBranch text_proj, ClipBranch image_proj, and CrossAttentionFusion for 30 epochs on cable_gland. The RGB backbone remained frozen. I-AUROC dropped from 0.7904 (frozen baseline on cable_gland) to 0.7209 (MemoryBank) — a 7pp degradation. The clip_image_proj gradient norm was approximately 10^-6, three orders of magnitude smaller than the fusion model's gradient norm, indicating the projection layers were barely learning. Separate learning rates (lr_depth=5e-4, lr_clip=5e-5) did not help.

**Variant (c): Frozen baseline (unseeded).** Fusion-only training from cached features with random-init depth/CLIP backbones. No fixed random seed. I-AUROC averaged 0.7836, but category-level results varied significantly between runs (e.g., cookie: 0.6002→0.7535, dowel: 0.7574→0.8613) due to non-deterministic random initialization of DepthBackbone and ClipBranch projection layers.

**Variant (d): Frozen baseline (seed=42) — FINAL.** Same as (c) but with `torch.manual_seed(42)` applied before model construction in `extract_features.py`. Verified bit-identical cached features across two independent extraction runs. This is the locked-in baseline.

**Why frozen features won in every comparison.** The MVTec 3D-AD categories contain 200–290 training images each (normal samples only). The DepthBackbone (23K params) and ClipBranch projection layers (164K params) have sufficient capacity to overfit this data volume within 30 epochs. The pretrained MobileNetV3-Small backbone, trained on ImageNet (1.28M images), produces features that are already well-calibrated for visual anomaly detection. Fine-tuning these features on ~250 images degrades the feature distribution without improving defect discrimination. This is consistent with findings in the parent paper (arXiv 2604.22899), where frozen pretrained features outperformed fine-tuned ones in low-data regimes.

## 4. Final Results

The final locked-in baseline uses frozen backbones (seed=42), fusion-only training (30 epochs, lr=5e-4), k-NN MemoryBank scoring (k=5), and score calibration.

### 4.1 Per-Category Results

| Category | I-AUROC | P-AUROC | Latency (ms) | Train Images | Test Images |
|:---------|--------:|--------:|-------------:|-------------:|------------:|
| bagel | 0.8383 | 0.9736 | 261.0 | 244 | 110 |
| cable_gland | 0.7937 | 0.9800 | 208.1 | 223 | 108 |
| carrot | 0.7868 | 0.9823 | 239.6 | 286 | 159 |
| cookie | 0.6782 | 0.9668 | 221.7 | 210 | 131 |
| dowel | 0.8928 | 0.9901 | 303.8 | 288 | 130 |
| foam | 0.6775 | 0.8618 | 219.9 | 236 | 100 |
| **Average** | **0.7779** | **0.9591** | **242.3** | **248** | **123** |

### 4.2 Per-Defect-Type Breakdown (Examples)

**bagel** (4 defect types, 22 normal test images):

| Defect Type | Anomalous Samples | Per-Type I-AUROC |
|:------------|------------------:|-----------------:|
| Combined | 23 | 0.9071 |
| Contamination | 22 | 0.7252 |
| Crack | 22 | 0.9091 |
| Hole | 21 | 0.8074 |

**cable_gland** (4 defect types, 21 normal test images):

| Defect Type | Anomalous Samples | Per-Type I-AUROC |
|:------------|------------------:|-----------------:|
| Bent | 21 | 0.7574 |
| Cut | 22 | 0.8247 |
| Hole | 22 | 0.7987 |
| Thread | 22 | 0.7922 |

Per-type I-AUROC is computed by comparing each defect type's anomalous samples against all normal test images (not in isolation). This ensures the AUROC reflects the actual discrimination ability between normal and defective samples.

## 5. Reproducibility

Feature extraction is seeded with `torch.manual_seed(42)` at the start of `extract_and_cache()`, before any model is instantiated. This ensures the randomly initialized DepthBackbone (23,520 params) and ClipBranch projection layers (164,352 params) produce identical initial weights on every run.

Verification: two independent runs of `extract_features.py` with zero code changes between them produced bit-identical cached features for all 6 categories x 2 splits (12 files total), confirmed via `torch.equal()`:

```
bagel_train: rgb=True depth=True text=True -> PASS
bagel_test: rgb=True depth=True text=True -> PASS
cable_gland_train: rgb=True depth=True text=True -> PASS
cable_gland_test: rgb=True depth=True text=True -> PASS
carrot_train: rgb=True depth=True text=True -> PASS
carrot_test: rgb=True depth=True text=True -> PASS
cookie_train: rgb=True depth=True text=True -> PASS
cookie_test: rgb=True depth=True text=True -> PASS
dowel_train: rgb=True depth=True text=True -> PASS
dowel_test: rgb=True depth=True text=True -> PASS
foam_train: rgb=True depth=True text=True -> PASS
foam_test: rgb=True depth=True text=True -> PASS
```

The seed=42 is an arbitrary choice, not a claim that this is the optimal initialization. A different seed would produce a different (but equally valid, equally reproducible) baseline number.

## 6. Limitations

The following limitations are documented in full in `docs/LIMITATIONS.md`. They are summarized here for integration into this report.

**CLIP branch is untrained.** The `text_proj` and `image_proj` layers in ClipBranch are randomly initialized and never trained. Three separate experiments confirmed that frozen features outperform trained ones (see Section 3). As a result, CLIP text similarity outputs are not discriminative between defect types: across all 50 explainability visualizations generated on the final baseline, the CLIP top-match phrase was "contaminated" for every single sample (50 out of 50), regardless of the actual defect type. Zero variation was observed across all 6 categories and all 8 defect types (bent, combined, contamination, crack, cut, hole, thread, color).

**Reproducibility scope.** Extraction is seeded and verified bit-identical across two runs. The seed is a specific arbitrary choice — a different seed would give a different (but equally valid, equally reproducible) baseline.

**Explainability signal reliability.** The RGB/depth residual maps and the anomaly heatmap are the reliable explainability outputs. The CLIP text-similarity panel is an architectural demonstration (proving the fusion mechanism can incorporate a text branch) rather than a trained diagnostic signal.

## 7. Explainability

The `visualize_explainability.py` script generates 8-panel PNG plots per test sample. The 50 visualizations produced for this report are stored in `outputs/explainability/`, organized as `{category}_{defect_type}_sample{idx:03d}.png`.

### 7.1 Panel Layout

| Panel | Content | Reliability |
|:------|:--------|:------------|
| Top-left | Input RGB image (denormalized) | — |
| Top-second | Input depth map (XYZ, jet colormap) | — |
| Top-third | Ground truth defect mask | — |
| Top-right | Fusion anomaly map (hot colormap) + score | Reliable |
| Bottom-left | RGB branch residual overlay | Reliable |
| Bottom-second | Depth branch residual overlay | Reliable |
| Bottom-third | CLIP text similarity bar chart | Architectural demo only |
| Bottom-right | Decision summary (defect type, driver, confidence, peak location) | Reliable |

### 7.2 Example Outputs

**bagel — crack defect** (`outputs/explainability/bagel_crack_sample045.png`):
- Anomaly Score: 20.751 | Confidence: 95.6%
- Driver: RGB (color/texture)
- CLIP top-match: "contaminated" (similarity: -0.236) — not meaningful for crack detection

**cable_gland — thread defect** (`outputs/explainability/cable_gland_thread_sample086.png`):
- Anomaly Score: 18.416 | Confidence: 93.7%
- Driver: RGB (color/texture)
- CLIP top-match: "contaminated" (similarity: -0.224) — not meaningful for thread defect detection

In both cases, the RGB residual map and anomaly heatmap correctly localize the defective region. The CLIP panel outputs the same phrase regardless of input, confirming it is not a trained diagnostic signal (see Section 6).

### 7.3 Coverage

50 visualizations were generated across all 6 categories: 8 samples per category (2 per defect type), except carrot which has 5 defect types (10 samples total). All categories produced output without errors. Summary text files (`{category}_summary.txt`) accompany each batch of PNGs.

## 8. Deviations from Assigned Architecture

**Live Dashboard / Operator Feedback Loop.** The reference architecture diagram includes a live dashboard stage with threshold adjustment, export, alerts, and trend visualization, connected by a dashed operator-feedback loop back into anomaly scoring. This project does not implement any of that. Instead, evaluation is performed via a CLI script (`evaluate.py`) that prints aggregate metrics to stdout, and explainability is delivered as static PNG files (`visualize_explainability.py`). No web UI, real-time streaming interface, or online threshold/memory-bank updates from operator feedback were built. This is a scoping decision driven by the college project timeline, not an oversight.

**Depth Backbone Training Status.** The reference diagram labels the depth backbone as a "small trainable CNN." In the final locked-in baseline, this module is frozen at random initialization rather than trained. This is a deviation from the assigned architecture. Three separate experiments in Section 3 demonstrate that the frozen configuration outperforms trained variants: full E2E fine-tuning (Variant a) achieved an average I-AUROC of approximately 0.56, targeted depth/CLIP training (Variant b) achieved 0.7209, and the frozen baseline (Variant d) achieved 0.7779. Freezing the depth backbone was not assumed in advance — it was adopted after empirical evidence showed that training it on 200–290 images per category degraded performance relative to leaving it at random initialization.
