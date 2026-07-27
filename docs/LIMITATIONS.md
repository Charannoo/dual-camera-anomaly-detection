# Limitations

## CLIP Branch Is Untrained

The `text_proj` and `image_proj` layers in `ClipBranch` are randomly initialized and never trained. This was a deliberate final choice, not an oversight. Three separate experiments confirmed that frozen features outperform trained ones:

1. **Full E2E fine-tuning** (15 epochs, all backbones + fusion): avg I-AUROC dropped from 0.7367 to 0.56. E2E checkpoints backed up to `outputs/checkpoints/e2e_backup/`.
2. **Targeted depth/CLIP training** (30 epochs, only depth_backbone + clip projections + fusion): avg I-AUROC dropped to 0.5610 (Mahalanobis) / 0.7209 (MemoryBank) on cable_gland alone, below the 0.7904 frozen baseline.
3. **Separate learning rates** for depth/CLIP vs fusion (lr_depth=5e-4, lr_clip=5e-5): same pattern — trained features degraded anomaly discrimination.

The pretrained CLIP vision encoder already produces useful features through the frozen RGB backbone. The projection layers were added to demonstrate the architectural mechanism for incorporating text-guided defect descriptions, but training them on this dataset's limited normal-only data leads to overfitting that harms generalization.

**Concrete consequence:** CLIP text similarity outputs are not discriminative between defect types. Across all 50 explainability visualizations generated on the final baseline (seed=42), the CLIP top-match phrase was "contaminated" for every single sample — 50 out of 50 — regardless of the actual defect type (bent, combined, contamination, crack, cut, hole, thread, color). Zero variation was observed across all 6 categories. The CLIP text branch currently functions as an anomaly presence detector (any non-normal input triggers a high anomaly score), not a defect type classifier.

## Reproducibility Scope

Feature extraction is seeded with `torch.manual_seed(42)` and verified bit-identical across two independent extraction runs. This means:

- The DepthBackbone (randomly initialized CNN) and ClipBranch projection layers (randomly initialized linear layers) produce identical features every time the pipeline is run from a clean `outputs/` directory.
- A different seed would produce a different (but equally valid, equally reproducible) baseline number. The seed=42 choice is arbitrary — it is not claimed to be optimal.
- The RGBBackbone uses pretrained MobileNetV3-Small weights (deterministic, not seed-dependent).
- The CLIP model loads pretrained ViT-B/32 weights (deterministic, not seed-dependent).

## Explainability Signal Reliability

The explainability outputs contain four panels of varying reliability:

1. **Fusion Anomaly Map** — reliable. This is the primary output of the anomaly detection pipeline. It localizes anomalous regions by combining RGB, depth, and text branch residuals via the trained fusion model.

2. **RGB Branch Residual** — reliable. Shows where the RGB (color/texture) branch detects deviation from learned normal patterns. Consistently the dominant driving signal across all categories (all 50 samples showed "Driver: RGB (color/texture)").

3. **Depth Branch Residual** — reliable when nonzero. Shows geometry/shape anomalies from the depth sensor. In the current baseline, the RGB residual consistently dominates — the depth branch provides complementary signal but rarely drives the decision alone.

4. **CLIP Text Similarity bar chart** — **not a trained diagnostic signal.** This panel demonstrates the architectural mechanism: the fusion model can accept a text token derived from CLIP text embeddings weighted by cosine similarity to defect descriptions. However, since the projection layers are untrained, the similarity scores are not meaningful for defect classification. The panel is retained as proof-of-concept for the text-guided fusion architecture. Do not interpret the CLIP phrase bar chart as a diagnostic indicator in the current model.
