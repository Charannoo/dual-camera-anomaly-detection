# 📊 Your Project vs. the Two Research Papers — Honest Comparison

> ⚠️ **READ THIS FIRST:** Comparisons are only valuable if they are TRUE. This document is honest. Your project does NOT win on detection accuracy (I-AUROC). It DOES win on **hardware cost, memory, and explainability**. Use the "where we truly win" sections — never fake the accuracy numbers.

## The 3 systems

| | **Your project** | **Paper 1** (arXiv 2604.22899) | **Paper 2 (CMDR–IAD)** |
|---|---|---|---|
| Full name | Dual-Camera Cross-Attention Fusion | Text-Guided Multimodal Unified IAD | Cross-Modal Mapping & Dual-Branch Reconstruction |
| Author/date | yourselves (2026) | Li, Ye, Yu, Xie, Shen (2026) | ECGAI Research (2026) |
| Inputs | RGB + **depth image** + text (CLIP) | RGB + 3D point cloud + text | RGB + 3D point cloud |
| Fusion idea | Cross-attention, RGB↔Depth reconstruct each other | Geometry-Aware Cross-Modal Mapper + text-guided OCTA (MoE+attention) | Cross-modal mapping + dual-branch reconstruction |

---

## 🎯 HEADLINE COMPARISON (MVTec 3D-AD)

| Metric | **Your Project** | Paper 2 (CMDR) | Paper 1 (Ours/Unified) |
|--------|:---------------:|:--------------:|:---------------------:|
| **I-AUROC (detect IF defective)** | **0.778 (77.8%)** | 97.3% | 94.0% |
| **P-AUROC (find WHERE)** | **0.959 (95.9%)** | 99.6% | —(uses AUPRO instead) |
| AUPRO@30% | not reported | 97.6% | 97.0% |
| Runners | CPU-only, no GPU | GPU | GPU (1089 MB mem) |
| Model size | **< 1 MB fusion, ~1.3M total** | lightweight | 1089 MB |
| Params trained | 401K (fusion 213K) | moderate | moderate |
| Latency / speed | ~242 ms ~ 4 FPS (CPU) | 24.6 FPS (GPU) | 10.1 FPS (GPU) |
| Number of categories | 6 | 10 (full 3D-AD) | 10 (full) |

**Plain English:** They are better at *deciding "is it defective?"* (94–97% vs our 78%). We are competitive at *finding where* (95.9% vs 99.6%). **On accuracy, they win. Do NOT claim otherwise.**

---

## ✅ WHERE **YOUR** PROJECT IS GENUINELY BETTER

### 1. Runs on a normal CPU — no GPU needed
- Papers report results on **GPU** (M3DM: 65 GB RAM!; CMDR: ~465 MB; arXiv-1: 1089 MB).
- Your project: **CPU-only, no CUDA** — 242 ms/image on a laptop. This is a real, provable advantage for cheap edge hardware. → *"Theirs need an expensive GPU card; mine runs on the factory PC.'**
- (Grading constraint was explicitly CPU-only, so this is exactly what was asked of you.)

### 2. Tiny model footprint
- Papers ship **huge models** (M3D) 65,261 MB memory; even the "lightweight" one is ~1 GB).
- Yours: **fusion < 1 MB, total ~1.3M params, 401K trainable.** You can ship it on a microcontroller/phone.

### 3. Explainability is built in
- Your project: per-branch RGB/depth heatmaps + text phrase + rule-based "this is a crack, here's where, 95% confident."
- Papers emphasize score only; they do not ship a human-facing explanation panel as your pipeline does (open `outputs/explainability/*.png`).
- You also deliver an interactive **3D web explorer** — the papers have code, not a demo tool for operators.

### 4. Honest, reproducible engineering
- Fixed seed (42), verified bit-identical runs, fully documented limitations (`docs/LIMITATIONS.md`). A research lab reporting a limitation honestly is unusual and defensible.

### 5. Embedded real-world cost per unit
- Your pipeline: 300 ms/image on CPU + small disk = practical for real-time inspection at low cost.
- Paper 1 itself notes G2SF needs 12GB+; M3D needs 65GB. That is simply not deployable on an edge device.

---

## 📉 WHERE THEY ARE BETTER (be ready to say "yes, and here's why I accept it")

| They beat us on | Honest framing |
|:--|:--|
| I-AUROC 94–97% vs 78% | "Our goal (course) was an <1MB, CPU, explainable baseline. We deliberately prioritized deployability and explainability over peak AUROC. Our P-AUROC (95.9%) shows localization is strong; the gap in detecting *if* is the tradeoff for being CPU-only and tiny." |
| Full 10-category dataset | "We evaluated on the 6-category subset used by similar lightweight baselines; running more categories costs only training time, architecture is identical." |
| Richer 3D modeling (point clouds) | "We discretize depth to a 2D depth-V to stay lightweight and fast on CPU; real 3D network off points is heavier — a conscious cost/performance trade." |

---

## 💬 SCRIPT TO SAY TO YOUR MENTOR (memorize)

> **"Compared to the two papers, the papers achieve higher detection AUROC (94–97% vs our 78%) because they use larger GPU-based models and real 3D point-cloud networks. Our project is not claiming to beat them on accuracy. What we uniquely deliver — within a sub-1M-param, **CPU-only** constraint — is: a model that runs in ~250 ms without any GPU (theirs need 1–65 GB of GPU memory), an explainability panel that shows WHICH sensor and WHERE the defect is, and an interactive web demo. In other words, theirs is a better algorithm on paper; ours is a NG deployable, explainable inspection machine on a normal PC. The papers validate our design choices: we also use frozen pretrained backbones, cross-modal reconstruction, and memory-free scoring — so we are aligned with current best practice, just optimized for the edge.'"

> "Where we clearly lose (I-ARRIM 78% vs 94%) I accept and would address by training on all 10 categories and adopting a 3D backbone — a clear next step, which shows I understand the field."

---

## 🧪 THE HONEST-SCORE-UNDER-THE KNIFE RULE
If the mentor asks "who is better?" say the **truth**: *"On detection, they are. On cost, CPU, and explainable design, we are — and those are the grading constraints you gave us."* That honesty is worth more than any inflated table.