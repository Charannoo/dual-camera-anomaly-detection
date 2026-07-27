# Lightweight Dual-Camera Explainable Anomaly Detection

A sub-1M parameter, unsupervised anomaly detection pipeline for MVTec 3D-AD data.
Uses a frozen MobileNetV3-Small RGB backbone + a lightweight Depth CNN, fused via a single
cross-attention layer trained to predict cross-modal features on normal samples only.

---

## Architecture

```
RGB Image  →  MobileNetV3-Small (frozen)  →  [48, 14, 14] features
Depth Map  →  3-Layer CNN (frozen early)  →  [64, 14, 14] features
                     ↓
          CrossAttentionFusion (dim=128, 4 heads)
           RGB tokens attend to Depth tokens & vice versa
                     ↓
        Cross-Modal Prediction Heads (MSE on residuals)
                     ↓
        Mahalanobis Distance Scoring → Anomaly Maps
                     ↓
    Per-branch heatmaps + rule-based text explanation
```

## Constraints Met

| Constraint | Value |
|---|---|
| Hardware | CPU-only (no CUDA) |
| Trainable params | 213,248 (< 1M) |
| Fusion model size | 833 KB on disk |
| Latency | ~15–22 ms/image on CPU |
| Training data | Normal samples only |
| Backbone training | Frozen (inference-only) |

---

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline Usage

### 1. Prepare Data
Drop MVTec 3D-AD category folders inside `data/`. The expected folder structure is:
```
data/
  cable_gland/
    train/good/{rgb,xyz}/
    test/{good,<defect_type>}/{rgb,xyz,gt}/
  foam/
    ...
```

> For testing without the real dataset, run the smoke test which auto-generates synthetic data:
> ```bash
> python -m src.datasets.mvtec_3d
> ```

### 2. Extract Features
```bash
python -m src.scripts.extract_features
```

### 3. Train Fusion Model
```bash
python -m src.scripts.train_fusion --category cable_gland --epochs 10
python -m src.scripts.train_fusion --category foam --epochs 10
```

### 4. Evaluate
```bash
python -m src.scripts.evaluate --category cable_gland
python -m src.scripts.evaluate --category foam
```

### 5. Explainability Visualizations
```bash
python -m src.scripts.visualize_explainability
```
Outputs per-branch heatmaps and rule-based explanations to `outputs/`.

---

## Results (on synthetic mock dataset)

| Category     | I-AUROC | P-AUROC | Latency (ms) | Params  |
|---|---|---|---|---|
| cable_gland  | 1.0000  | 0.9995  | 18.6         | 213,248 |
| foam         | 1.0000  | 0.9995  | 18.4         | 213,248 |
| cookie       | 1.0000  | 0.9995  | 21.7         | 213,248 |
| potato       | 1.0000  | 0.9995  | 13.5         | 213,248 |

> ⚠️ These results are on **synthetic** mock data. Real MVTec 3D-AD results will differ.

---

## Explainability

Every prediction is traceable to:
1. **Which branch drove it** — RGB (color/texture) or Depth (geometry/shape)
2. **Which region** — pixel-level anomaly heatmap at native resolution (224×224)
3. **Confidence** — normalized score from the Mahalanobis distance distribution

Example output:
```
Sample 00 | Decision: ANOMALOUS
  - Driven by: Depth sensor branch (geometry/shape defect)
  - Primary location: x=85, y=80 (pixel coordinates)
  - Raw Anomaly Score: 314.53 | Confidence: 99.11%
```

---

## Project Structure

```
anomaly-detection/
  data/               # MVTec 3D-AD categories (gitignored)
  src/
    datasets/         # MVTec3DADataset + mock generator
    models/           # rgb_backbone, depth_backbone, fusion, scoring
    scripts/          # extract_features, train_fusion, evaluate, visualize_explainability
  configs/            # config.yaml
  outputs/            # features, checkpoints, eval results (gitignored)
  requirements.txt
  README.md
  PROGRESS.md
  verify_env.py
```
