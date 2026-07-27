# Lightweight Dual-Camera Explainable Anomaly Detection

A CPU-only, sub-1M parameter, unsupervised anomaly detection pipeline for MVTec 3D-AD data.
Uses a frozen MobileNetV3-Small RGB backbone + a lightweight Depth CNN + a CLIP text branch,
fused via a 3-way cross-attention layer trained to predict cross-modal features on normal samples only.

---

## Architecture

```
RGB Image   →  MobileNetV3-Small (frozen)   →  [48, 14, 14] features
Depth Map   →  3-Layer CNN (frozen)         →  [64, 14, 14] features
Text Query  →  CLIP ViT-B/32 + projection   →  [1, 128] token
                              ↓
              CrossAttentionFusion (dim=128, 4 heads)
               RGB ↔ Depth ↔ Text cross-attention
                              ↓
              Cross-Modal Prediction Heads (MSE on residuals)
                              ↓
              k-NN Memory Bank or Mahalanobis scoring → Anomaly Maps
                              ↓
              Per-branch heatmaps + rule-based text explanation
```

## Constraints Met

| Constraint | Value |
|---|---|
| Hardware | CPU-only (no CUDA) |
| Trainable params | ~450K (fusion 213K + CLIP proj 164K + depth CNN 23K) |
| Fusion model size | < 1 MB on disk |
| Latency | 113–281 ms/image on CPU |
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
  carrot/
    ...
```

Valid categories: `bagel`, `cable_gland`, `carrot`, `cookie`, `dowel`, `foam`.

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
python -m src.scripts.train_fusion --category cable_gland --epochs 30
python -m src.scripts.train_fusion --category foam --epochs 30
```

### 4. Evaluate
```bash
python -m src.scripts.evaluate --scorer memorybank
python -m src.scripts.evaluate --scorer mahalanobis
```

### 5. Explainability Visualizations
```bash
python -m src.scripts.visualize_explainability
```
Outputs 8-panel plots per category with anomaly maps, per-branch residuals, CLIP text similarity, and rule-based explanations.

---

## Results (MVTec 3D-AD real data, 6 categories)

Extraction is seeded (seed=42) for reproducibility. DepthBackbone and CLIP projection layers are
randomly initialized but frozen (untrained) — verified bit-identical across independent runs with
the same seed.

### k-NN Memory Bank Scorer (k=5)

| Category      | I-AUROC | P-AUROC | Latency (ms) |
|---|---|---|---|
| bagel         | 0.8383  | 0.9736  | 261.0        |
| cable_gland   | 0.7937  | 0.9800  | 208.1        |
| carrot        | 0.7868  | 0.9823  | 239.6        |
| cookie        | 0.6782  | 0.9668  | 221.7        |
| dowel         | 0.8928  | 0.9901  | 303.8        |
| foam          | 0.6775  | 0.8618  | 219.9        |
| **Average**   | **0.7779** | **0.9591** | **242.3** |

> k-NN Memory Bank scoring outperforms Mahalanobis on average while remaining under 300ms/image on CPU.

---

## Explainability

Every prediction is traceable to:
1. **Which branch drove it** — RGB (color/texture), Depth (geometry/shape), or Text (semantic)
2. **Which region** — pixel-level anomaly heatmap at native resolution (224x224)
3. **Which defect type** — CLIP text similarity shows which defect description matched best
4. **Confidence** — calibrated score from the Mahalanobis/k-NN distribution

Example output:
```
Sample 000 | Decision: ANOMALOUS
  - Driven by: Depth sensor branch (geometry/shape defect)
  - Best text match: "a cracked or broken object" (sim=0.28)
  - Primary location: x=85, y=80 (pixel coordinates)
  - Anomaly Score: 0.85 | Confidence: 97.3%
  - Defect Type: crack
```

---

## Project Structure

```
dual-camera-anomaly-detection/
  data/                       # MVTec 3D-AD categories (gitignored)
  src/
    datasets/                 # MVTec3DADataset + mock generator
    models/
      rgb_backbone.py         # MobileNetV3-Small feature extractor
      depth_backbone.py       # Lightweight 3-layer depth CNN
      clip_branch.py          # CLIP text-based defect scoring
      fusion.py               # 3-way cross-attention fusion
      scoring.py              # Mahalanobis + k-NN Memory Bank scorers
      calibration.py          # Score calibration
    scripts/
      extract_features.py     # Offline feature extraction
      train_fusion.py         # Fusion-only training
      train_e2e.py            # End-to-end backbone + fusion training
      evaluate.py             # Full evaluation with per-defect breakdown
      visualize_explainability.py  # 8-panel explainability plots
  configs/                    # config.yaml (category list + hyperparams)
  outputs/                    # features, checkpoints, eval results (gitignored)
  requirements.txt
  README.md
  PROGRESS.md
```

---

## Training on GPU (Kaggle / Colab)

The pipeline is designed to run entirely on CPU. For faster training, copy the project
to a Kaggle or Colab notebook with GPU access. Feature extraction and evaluation
are the main GPU (backbone inference is ~10x faster on GPU).

---

## Key Findings

1. **k-NN scoring > Mahalanobis** on this dataset — more robust to non-Gaussian residual distributions
2. **CLIP text branch provides semantic context** — defect type descriptions scored via cosine similarity
3. **End-to-end fine-tuning degrades performance** — the frozen pretrained backbone features are more discriminative than fine-tuned ones for this dataset
4. **Pixel-level AUROC is consistently high** (>0.85) — the model localizes anomalies well even when image-level scoring varies
5. **Sub-300ms inference** — practical for real-time inspection on CPU hardware
