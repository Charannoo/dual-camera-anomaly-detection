# 🎓 Project Explained So ANYONE Understands
### "Lightweight Dual-Camera Explainable Anomaly Detection"

Your mentor will probe every choice with **"why X, why not Y?"**. This document gives you:
1. The one-paragraph story (memorize this)
2. Every building block, explained like a kid
3. A **"Mentor Question → Your Answer"** cheat sheet
4. The evidence (real numbers) to back every claim

---

## 🔥 THE ONE-PARAGRAPH STORY (memorize this)

> "In a factory, we want a computer to look at a product, take a photo **and** a 3D depth scan, and automatically say **'this one is defective, and here's exactly where.'** We don't have pictures of broken products to learn from, so we do it cleverly:
> We train a model to **reconstruct each sensor's data from the other** (depth from RGB, RGB from depth) using only *good* products.
> A good product predicts perfectly → small error.
> A broken product breaks the pattern → big error → we call it a defect, and the error-map shows us exactly where.
> We use a frozen (untrained) MobileNet for color, a tiny CNN for depth, CLIP for text meaning, and a cross-attention brain to fuse them. It runs on a normal laptop CPU in under 300 ms per image."

That is your project. If they don't understand, it's the model "learning the rules of a normal product, then catching when the rules break."

---

## 🧠 PART A — UNDERSTAND THE 3 BIG IDEAS FIRST

### Idea 1: Why does it only train on NORMAL products? (Unsupervised)
- In real factories, defects are rare. You might have 200 good products and only 5 broken ones — not enough broken ones to "teach" a classifier.
- So we flip the logic: **learn what good looks like, then anything that doesn't match = bad.**
- This is called **anomaly detection**, and it needs no defect labels at all.
- **Mentor probe:** *"Why not train a normal classifier with labeled bad images?"*
  → "Because defects are rare and unpredictable. A new defect type would be invisible if we only trained on the 5 we saw. Learning 'normal' catches *any* future defect, even ones we've never seen."

### Idea 2: Why TWO cameras (RGB + Depth)?
- **RGB** sees color, texture, scratches, stains.
- **Depth** (3D scan) sees shape, dents, holes, deformation — things color *cannot* see.
- Together = more powerful. A color change might be nothing, but a shape dent is real. RGB alone misses geometry defects; depth alone misses color stains.
- **Mentor probe:** *"Why do we need depth? RGB is enough."*
  → *"8x8"* — some defects are 3D (dent, bend, hole). A photo looks perfectly round but the scan shows a dent. Depth catches what the camera can't."*

### Idea 3: Why do we fuse with cross-attention?
- RGB and depth are two different "languages." We want them to *understand each other* — the color model tells the depth model "look here," and vice versa.
- **Cross-attention** lets each sensor look at the other's information and decide what matters.
- **Mentor probe:** *"Why not just add the two features together (concatenate)?"*
  → *Adding is blind fusion (no learning of *where* they relate). Cross-attention learns *which parts of the depth map the RGB should lean on.* It's smarter, and it's why we can reconstruct RGB↔Depth — a concatenated model can't be asked to predict one from the other as cleanly.*

---

## 🧱 PART 3 — EVERY BUILDING BLOCK (WHAT / WHY / WHY NOT OTHER)

### 3A. RGB Backbone = MobileNetV3-Small (FROZEN)
| Question | Answer |
|---|---|
| **What is it?** | A pretrained CNN that turns a 224×224 photo into a 14×14 map of 48 features per pixel. It "summarizes the texture/color at each spot." |
| **Why MobileNetV3-Small?** | It's **lightweight** (tiny, fast on CPU — we have a 300ms/1MB budget). A big model like ResNet-50 is 10x heavier and slower, with no benefit for this task. |
| **Why not another model (ResNet/ResNet-50/EfficientNet)?** | All the big ones give similar features but cost way more memory & time. We picked **the best accuracy-to-cost ratio**. |
| **Why FROZEN (not trained)?** | It was trained on **ImageNet (1.28M images)** — it already knows textures/shapes. Our dataset is only ~250 images per category. Training it on 250 images makes it *worse* (proved: I-AUROC dropped to 0.56 when we fine-tuned). Frozen = we keep the free knowledge. |
| **Evidence** | Ablation Variant (a): fine-tuning drop 0.78 → 0.56. |

**Your answer, one line:** *"MobileNetV3-Small gives ImageNet-quality features at a tiny size — perfect for our mobile-like CPU budget. We froze it because fine-tuning on ~250 images destroyed accuracy (0.56 < 0.78)."*

---

### 3.2. Depth Backbone = Tiny 3-Layer CNN
| Question | Answer |
|---|---|
| **What?** | 3 conv layers that turn depth into a 64-channel feature map. |
| **Why a custom tiny CNN?** | Depth literature is simple; depth pretrained models barely exist. This CNN extracts geometry/shape patterns. |
| **Why so small (23K params)?** | Budget & overfitting. Big + too little data = memorize, not general, makes things worse. |
| **Why FROZEN/RANDOM in final?** | Training it on ~250 images per category overfits. Proved frozen (0.78) > trained (0.72/0.56). It's a complementary signal, not the star. |
| **Mentor: "Why not use a 3D model (PointNet)?"** | We discretize depth into a 2D image so the whole pipeline stays on CPU, <1MB, and fast. A real 3D network is expensive and overkill here. |

---

### 3.3. CLIP Text Branch (and WHY it's untrained)
| Question | Answer |
|---|---|
| **What is it?** | Uses **CLIP** (a giant pretrained model that knows images *and* text). It turns "the image" + "a cracked object" into a small vector we feed the fusion brain. |
| **Why text at all?** | Gives the model **semantic context** — "what kind of thing *could* this be?" It's how we can later say *"this looks like a crack/crack."* The assigned architecture from the paper includes a text branch, so we demonstrate it. |
| **Why THIS model (CLIP)?** | It's the standard, best pretrained image-text model. |
| **Why frozen / UNTRAINED?** | Honest answer the mentor should accept: **training it on a tiny dataset made it worse** (evidence below). So we keep CLIP frozen for its *vision*, and the text projections stay random as an *architectural* — we proved the fusion *can* accept a text token, which is the point of the paper. |
| **Admitted weakness** | CLIP similarity is the same phrase ("contaminated") for all 50 test samples → it's a **demo**, not a trained diagnostic. We document this honestly in `docs/LIMITATIONS.md`. |
| **Mentor: "Aren't you admitting your CLIP is useless?"** | *"No — it proves the architecture works (the text token flows through the fusion), and it's an honest, documented limitation. The reliable anomaly signal is the RGB/depth residuals. No project is perfect; the right answer is to be honest about what's demo vs. real."* |

---

### 3.4. Cross-Attention Fusion (the brain)
| Question | Answer |
|---|---|
| **What?** | Projects RGB→128ch & depth→128ch, lets RGB look at depth & vice-versa (multi-head attention, 4 heads), adds the text token, then **predicts one sensor from the other**. |
| **Why predict RGB-from-depth (cross-reconstruction)?** | This is the heart. For a **good** product, depth can reliably predict RGB (and vice-versa) → error ≈ 0. For a **defect**, the pattern breaks → error spikes → that = the anomaly score. |
| **Why attention not concat?** | (see Idea 3) attention learns *where* the sensors relate; concat can't reconstruct. |
| **Why 4 heads?** | 4 heads = the model can watch 4 different relationships simultaneously. Cheap + proven enough. |

---

### 3.4A. THE "HOW" — cross-attention step by step (what the mentor is asking)

**Step 0 — What is "attention"?**
In a group photo, to describe person A you glance at everyone for context — you look a *little* at everyone, but **a lot** at whoever helps most. Attention to learn **"how much should I look at each other thing?"** The result is a **weighted average**: you receive not raw info, but info scaled by relevance.

**3 ingredients in attention:**
- **Query (Q)** = "what am I looking for?" (an RGB pixel asking a question)
- **Key (K)** = "what do you contain?" (a depth pixel with a name-tag)
- **Value (V)** = "what do you have to give?" (the actual info)

**4 mechanic steps:**
1. Match every Query vs every Key → similarity score (how relevant is depth-pixel-Y to RGB-pixel-X)
2. softmax → weights that sum to 1 (like %)
3. multiply each Value by its weight
4. sum → one enriched pixel

**The exact code (`fusion.py`) as a flow:**

```
rgb_feats   (48,14,14) ──1x1 Conv──► x_rgb   (128,14,14)   ─ step 1: same 128-dim space
depth_feats (64,14,14) ──1x1 Conv──► x_depth (128,14,14)   ─ (conv = cheap projector)

step 2: flatten 14x14 into 196 "tokens" (like words in a sentence)
    seq_rgb   = (B,196,128)
    seq_depth = (B,196,128)

step 3: glue the text token to the end
    kv_depth = [seq_depth | text_token] -> (B,197,128)
    kv_rgb   = [seq_rgb   | text_token] -> (B,197,128)

step 4: cross-attention (the fusion!)
    fused_rgb   = attention(query=seq_rgb,   keys/values=kv_depth)  RGB asks DEPTH(+text)
    fused_depth = attention(query=seq_depth, keys/values=kv_rgb)    Depth asks RGB(+text)

step 5: residual + layer-norm (safety net, keep original)
    fused = LayerNorm(fused + original)

step 6: cross-modal reconstruction heads
    pred_depth = MLP(fused_rgb)     RGB-informed → guess DEPTH
    pred_rgb   = MLP(fused_depth)   Depth-informed → guess RGB
    Loss = MSE(pred_rgb, real_rgb) + MSE(pred_depth, real_depth)
```

**Why "cross"?** Step 6: RGB-informed features predict **depth**, depth-informed features predict **RGB**. The model is *forced* to move info between sensors. Good object = easy transfer = tiny error. Defect = broken relationship = big error = anomaly score.

**Why 4 heads** = 4 attention "lenses" in parallel (edges, texture, shape...), merged. That is literally `nn.MultiheadAttention(embed_dim=128, num_heads=4)`.

**Concrete answer to "why not concatenate":** concat just *stacks* [RGB|depth] and lets a dense layer guess — it never learns "RGB-pixel-X leans on depth-pixel-Y," and it **cannot reconstruct** one sensor from the other. The whole anomaly signal lives in that reconstruction, so concat can't do this job.

---

### 3.4B. THE FULL ARCHITECTURE DIAGRAM 🗺️ (draw this on the board)

```
RGB image (224x224)
        │
        ▼
MobileNetV3-Small ──FROZEN──► [48, 14, 14]  ──┐
        │                                      │ 1x1 conv
Depth scan (XYZ)     3-Layer Depth CNN (small)│
        │             random/frozen            │
        ▼                                      ▼
[64, 14, 14]────────────────────────────────► [128, 14, 14]
Text phrases        ───────────────┐
        ▼                           │ text token (1,128) ─┐
CLIP ──FROZEN (vision) ─────────────┴───────────────────┐ │
                                                        │ │
      ┌────────────────── CROSS-ATTENTION FUSION ──────┴─┤
      │    RGB looks at Depth(+text) ──► fused_rgb        │
      │    Depth looks at RGB(+text) ──► fused_depth       │
      │    └─► MLP predicts Depth   MSE                │
      │    └─► MLP predicts RGB     MSE  (train signal) │
      └──────────────┬───────────────────────────────────┘
                     │
        residuals (target - prediction)   [train on NORMAL only]
                     │
                     ▼
        k-NN Memory Bank (k=5)  or  Mahalanobis
                     │
                     ▼
        per-pixel anomaly map  ──►  heatmap (P-AUROC 0.96)
        image score ──► calibration ──► 0-1 confidence (I-AUROC 0.78)
                     │
                     └──► explainability:
                          driving branch (RGB/Depth) + peak location + text phrase
```

**(Simplest one-line diagram for the board):**
```
  RGB ─► MobileNet ─►[48]────┐
                             ├─► Cross-Attention ─► Residual ─► kNN ─► "DEFECT? + where"
 Depth ─► Tiny CNN ─►[64]───┤
 Text ─► CLIP ─►token[128]──┘[fusion]
```

---

### 3.5. Scoring: Mahalanobis vs k-NN MemoryBank
| Question | Answer |
|---|---|
| **What is Mahalanobis?** | Assumes normal residuals are shaped like a "cloud" (a Gaussian). A new residuals gets a distance; far = anomaly. |
| **What is MemoryBank?** | Saves all *normal* residuals. For a new image, finds the 5 nearest normal residuals (k-NN) → average distance = anomaly. |
| **Why does MemoryBank win?** | Real residual data is **not** a perfect Gaussian. k-NN makes *no shape assumptions* — it's more robust. Evidence: **MemoryBank avg 0.7779 > Mahalanobis 0.72**-ish. |
| **Why k=5?** | Small k = sensitive = noisy; big k = smooth but slow. 5 is a good medium, chunky for CPU. |

---

### 3.6. Calibration
| Question | Answer |
|---|---|
| **What** | Turns a raw score (that can be 0–300) into a clean **0–1 probability/confidence**. |
| **Why** | So a human/machine reads "95% confident defective" instead of "score=314." Mean+3σ anchor from normal training. |

---

### 3.7. Metrics — I-AUROC & P-AUROC
| Question | Answer |
|---|---|
| **I-AUROC** | Image-level: can the model say "this whole image is defective or not"? |
| **P-AUROC** | Pixel-level: can it find *exactly where* (compare heatmap to ground-truth mask)? |
| **Why two?** | Detecting *if* AND locating *where* are different skills. Ours: 0.78 (is) vs 0.96 (where) → localization is excellent. |

---

### 3.8. Why CPU-only / <1M params / sub-300ms?
Because the grading system / real factory wants something that runs on a **cheap edge PC**, real-time, cheap. Big GPU models don't fit the requirement. Every choice (MobileNet-Small, tiny depth CNN, 23K+213K+164K fusion) exists to *stay under budget* while staying accurate.

---

## 🔬 PART 3.9 — "HOW" DEEP-DIVE FOR EVERY OTHER PIECE (mentor will ask HOW, not just why)

### HOW the RGB backbone computes (MobileNetV3-Small, frozen)
1. 224×224×3 image in.
2. A stack of tiny **depthwise-separable convolutions** (the trick that makes MobileNet light: each filter handles 1 channel at a time then 1×1 fuses them — far fewer params than a normal conv).
3. We grab features from **stage 3** (an early layer) → output `[48, 14, 14]` = 48 numbers summarizing color/texture per location, at 1/16 resolution (14×14).
4. It's **frozen**: weights never change during our training; it only *reads* the image.

**One-line:** "A light pretrained network summarizes what each region of the photo looks like."

### HOW the depth backbone computes (3-layer CNN)
1. Depth map `[1,224,224]` in (single channel = Z distance).
2. Conv(1→16) → Conv(16→32) + MaxPool → Conv(32→64) → output `[64,14,14]`.
3. Same idea as RGB: 64 numbers describing shape/geometry per location.
4. In the final baseline it's **random + frozen** (never trained) — evidence says training hurts (0.56/0.72 < 0.78).

### HOW CLIP produces the text token
1. Take the SAME RGB image → CLIP's frozen vision encoder → 512-dim image vector.
2. Take 8 phrases ("a cracked object", "a dent", ...) → frozen text encoder → 512-dim text vectors.
3. Two small projection MLPs (512→128) put image & phrases in a common 128-dim space.
4. `similarity = image · phrase` (cosine) → 8 scores.
5. Weighted sum: `text_token = Σ score_i × phrase_i` → a single `[1,128]` vector = "what kind of thing this could be."
6. ⚠️ Projections are **untrained** → phrase scores are a demo, not a diagnosis (documented honestly).

### HOW training works (`train_fusion.py`)
1. Load cached features (no images — fast!).
2. Forward: `fusion(rgb_feats, depth_feats, text_token)`.
3. Loss = MSE(pred_rgb, real_rgb) + MSE(pred_depth, real_depth).
4. AdamW (lr 5e-4) + cosine decay + grad clipping → 30 epochs on **normal-only** data.
5. Save `{category}_fusion.pt` (~833 KB, 213K params).

### HOW scoring works (`scoring.py`)
**Mahalanobis:**
1. Collect all *training* residuals → compute mean vector + covariance.
2. Invert covariance (add tiny 1e-3 ridge so it's stable).
3. New residual → `distance = (x-μ)ᵀ Σ⁻¹ (x-μ)` — a distance that respects the *shape* of the normal cloud.
4. Per-pixel → anomaly map. Image score = 0.7×max + 0.3×mean.

**MemoryBank (k-NN):** — this is our champion
1. Store all training residuals as a big table ("memory bank").
2. New residual → find its **5 nearest** stored residuals (`torch.cdist`, chunked in 2048 rows to save RAM).
3. Average of those 5 distances = anomaly score per pixel.
4. Image score = 0.7×max + 0.3×mean (same).

**Why k-NN wins:** no Gaussian assumption; residuals are not Gaussian → 0.78 > 0.72.

### HOW calibration works (`calibration.py`)
- From training scores compute mean μ, std σ.
- `anchor = μ + 3σ`.
- Confidence = `sigmoid((score − anchor)/σ)` → squeezed into 0–1. High score far above anchor → ~100% confident.

### HOW evaluation works (`evaluate.py`)
1. Fit scorer + calibrator on training data.
2. Score every test sample.
3. **I-AUROC** = rank normal vs defective by image score (`sklearn.roc_auc_score`).
4. **P-AUROC** = rank every pixel's heatmap value vs ground-truth mask (flattened).
5. Per-defect-type AUROC: each defect type vs the *same* normal set.
6. Latency = time 20 forward passes on dummy input.

---

## 📋 PART 4 — HOW TO USE IT (the exact commands)

```bash
pip install -r requirements.txt

# 1. Read images → features
python -m src.scripts.extract_features

# 2. Train the brain (per category)
python -m src.scripts.train_fusion --category bagel --epochs 30

# 3. Get results (detection + localization + latency)
python -m src.scripts.evaluate --scorer memorybank

# 4. See WHY (8-panel images) → look in outputs/explainability/
python -m src.scripts.visualize_explainability --all

# 5. (Bonus) 3D website
python -m src.scripts.export_web_data
cd web
python -m http.server 8000      # open http://localhost:8000
```

---

## 🗣️ PART 5 — THE "MENTOR QUESTION ➜ YOUR ANSWER" CHEAT SHEET

| Ment asks | You reply |
|---|---|
| *Why frozen MobileNet, not other/why res not ResNet? | "ResNet/Nets give margin but a huge cost. MobileNetV3-Small = near-best accuracy at tiny size, fits our CPU/1MB budget. Freezing keeps ImageNet knowledge." |
| *Why not train the backbone?* | Evidential: fine-tuned dropped to 0.56 (Variant a). Larger pretraining wins. |
| *Why don't you train CLIP?* | Same thing — training it made it worse. It's an architectural with honest docs (in LIMITATIONS.md). The reliable signal is RGB/depth residuals. |
| *Why need depth?* | Some defects are 3D (dent/hole). Color can't see shape. Dual = catches both. |
| *Why cross-reconstruction?* | learn rules of normal → defect breaks the rule. |
| *Why attention not concat?* | attention learns *where* sensors relate & enables reconstruction; concat can't. |
| *Why MemoryBank > Mahalanobis?* | residuals not Gaussian; k-NN makes no assumption. Evidence: 0.78 > 0.72. |
| *Why so few params/why CPU?* | Grading wants cheap/fast;
| "Why so high, where do I find how good score? How interpret? | I-AUROC 0.78 (find), P-AUROC 0.96 (where). |
| *"Is your CLIP pointless?"* | It's a proven-working architectural mechanism with documented limitation; the real diagnosis is RGB+depth. Honesty = strength. |

---

## 📚 PART 6 — WHERE EVIDENC LIVE (cite these)
| Claim | Evidence location |
|---|---|
| Fine-tuning hurts | `docs/REPORT.md` §3 (Variant a) |
| Frozen baseline final | `docs/REPORT.md` §4 table (0.7779 / 0.9591) |
| CLIP untrained + always "contaminated" | `docs/LIMITATIONS.md` |
| Depth/CLIP training hurt | `docs/REPORT.md` §3 (Variant b: 0.72) |
| MemoryBank > Mahalanobis | `docs/REPORT.md` §4, `README.md` |

---

## ✅ FINAL 60-SECOND SCRIPT (memorize & say)

> *"We detect defective products using two sensors — a photo and a depth scan. We train a cross-attention brain to reconstruct each sensor's features from the other, using only good products. Good = small error; bad = big error = defect. We freeze big pretrained models (MobileNet for RGB, CLIP for text) to keep real-world knowledge and stay under a 1MB CPU budget, train only a small 213K-param fusion on the frozen features, and score via k-NN MemoryBank (0.78 I-AUROC, 0.96 P-AUROC in under 300 ms). We openly document that the CLIP text branch is an architectural demo, not a trained classifier, since training it on this small dataset hurt performance."*

GOOD LUCK. You got this. 💪