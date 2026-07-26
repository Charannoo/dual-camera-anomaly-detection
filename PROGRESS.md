# Project Progress: Lightweight Dual-Camera Explainable Anomaly Detection

## Stage 0: Environment Setup — ✅ DONE
PyTorch `2.11.0+cpu`, CUDA = `False`, all packages verified.

## Stage 1: Dataset Pipeline — ✅ DONE
`MVTec3DADataset` with mock generator. Tensor shapes: RGB `(3,224,224)`, Depth `(1,224,224)`, GT `(1,224,224)`.

## Stage 2: Feature Extraction & Caching — ✅ DONE
Frozen MobileNetV3-Small → `[48,14,14]`. Frozen 3-layer Depth CNN → `[64,14,14]`. Cached via `extract_features.py`.

## Stage 3: Fusion + Cross-Modal Training — ✅ DONE
`CrossAttentionFusion`: 213,248 trainable params. Loss decreases monotonically (~0.04 s/epoch CPU). No NaNs.

## Stage 4: Anomaly Scoring — ✅ DONE
Mahalanobis on 256-dim residuals. Normal avg score ≈ 13.7, Defective avg ≈ 314.6.

## Stage 5: Evaluation — ✅ DONE
| Category     | I-AUROC | P-AUROC | Latency (ms) |
|---|---|---|---|
| cable_gland  | 1.0000  | 0.9995  | ~28 ms       |
| foam         | 1.0000  | 0.9995  | ~37 ms       |

## Stage 6: Explainability — ✅ DONE
Per-branch (RGB/Depth) residual heatmaps at 224×224. Rule-based sentences (branch, location, confidence).
Verified: anomaly peak lands within actual defect rectangle for all 3 test samples.

## Stage 7: Extend to More Categories — ✅ DONE
Added `cookie` and `potato`. Both trained and evaluated, I-AUROC = 1.0000, P-AUROC = 0.9995.

## Stage 8: Report Artifacts — ✅ DONE
`src/scripts/generate_report.py` produces the full Markdown results table below.
`README.md` and `.gitignore` created.

### Final Results Table

| Category    | I-AUROC | P-AUROC | Latency(ms) | Params  |
| ----------- | ------- | ------- | ----------- | ------- |
| cable_gland | 1.0000  | 0.9995  | 27.7        | 213,248 |
| foam        | 1.0000  | 0.9995  | 36.7        | 213,248 |
| cookie      | 1.0000  | 0.9995  | 18.4        | 213,248 |
| potato      | 1.0000  | 0.9995  | 17.9        | 213,248 |

> ⚠️ On **synthetic** mock data. Plug in real MVTec 3D-AD folders and re-run the pipeline for final numbers.

## All stages COMPLETE. Pipeline is ready for real data.
