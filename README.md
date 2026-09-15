# Cross-Generator AI Image Detector

**Ensemble feature engineering achieves 95.1% accuracy detecting AI-generated images from generators never seen during training — vs 54.9% for CNN-only baseline.**

---

## Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Ensemble (Frequency + Texture + CNN) | **95.1% ± 3.7%** | **97.5% ± 2.0%** | **0.989** |
| Frequency-Only Baseline | 89.3% ± 6.1% | 94.7% ± 4.3% | 0.938 |
| CNN-Only Baseline | 54.9% ± 8.2% | 71.3% ± 12.1% | 0.612 |

**Per-generator accuracy (each generator held out once):**

| Generator | Architecture | Accuracy | Precision | Recall |
|-----------|-------------|----------|-----------|--------|
| SDXL | U-Net (improved) | 98.7% | 99.1% | 98.3% |
| FLUX.1-schnell | Diffusion Transformer | 96.7% | 96.9% | 96.5% |
| SD 1.5 | U-Net (classic) | 90.0% | 91.2% | 88.8% |

---

## The Problem

Detectors trained on one generator fail completely on another. A CNN fine-tuned on DALL-E images scores near chance on Midjourney images because it learns generator-specific artifacts rather than universal generation signatures. This is the cross-generator generalization problem — the reason AI image detection is harder than it looks.

**NTIRE 2026 at CVPR ran a dedicated challenge on this problem because no existing solution generalizes reliably.**

---

## Why the Ensemble Works

CNN features overfit to the specific visual style of each generator. Frequency domain artifacts and texture statistics are more architectural — they reflect how diffusion models upsample and attend, which varies less across prompts and more across architectures.

```
Single image
      │
      ├── DCT Frequency Analysis (7D)
      │   High-frequency region (rows 32-64, cols 32-64)
      │   Mean, std, max, 75th/90th/95th percentiles
      │   → Captures upsampling artifacts from diffusion process
      │
      ├── GLCM Texture Features (20D)
      │   4 orientations: 0°, 45°, 90°, 135°
      │   5 properties each: contrast, dissimilarity,
      │   homogeneity, energy, correlation
      │   → Detects unnatural texture uniformity in generated images
      │
      └── EfficientNet-B0 CNN (1280D)
          Penultimate layer, global average pooling
          → Learned visual representations from ImageNet
                    │
              Concatenate (1307D)
                    │
              StandardScaler
                    │
            Logistic Regression
                    │
              P(AI-generated)
```

---

## Experimental Design

3-fold rotation validation — each AI generator is held out once:

```
Fold 1: Train [FLUX + SD15 + COCO real] → Test on SDXL
Fold 2: Train [SDXL + SD15 + COCO real] → Test on FLUX
Fold 3: Train [SDXL + FLUX + COCO real] → Test on SD15
```

COCO real photos remain in all training folds. Results reported as mean ± std across 3 folds.

**Dataset:** 150 images per source × 4 sources (SDXL, FLUX, SD15, COCO) = 600 total. Balanced classes eliminate need for weighted metrics.

**Key design decision:** Three architecturally distinct generators — SD 1.5 (classic U-Net), SDXL (improved U-Net), FLUX (Diffusion Transformer). The architecture diversity tests whether features generalize across fundamentally different generation approaches, not just different checkpoints of the same architecture.

---

## Setup

```bash
git clone https://github.com/oladri-renuka/cross-generator-ai-detector
cd cross-generator-ai-detector
pip install -r requirements.txt
```

**Run demo:**
```bash
python gradio_demo.py
# Opens at http://localhost:7860
```

**Full pipeline (generates images, extracts features, trains, evaluates):**
```bash
IMAGES_PER_GENERATOR=150 python main.py full -y
```

---

## Key Findings

**CNN-only fails at cross-generator generalization (54.9%)** because it learns generator-specific visual style rather than universal generation signatures. The frequency and texture features provide generator-agnostic signals.

**SD 1.5 is hardest to detect (90.0%)** — older model with frequency artifacts closer to real photos. SDXL and FLUX are easier despite being newer — their higher quality outputs have more distinct frequency signatures.

**Frequency features alone reach 89.3%** — showing that DCT analysis captures the core generalization signal. CNN features add 6 more points by catching patterns frequency analysis misses.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| CNN features | EfficientNet-B0 (timm) |
| Frequency features | OpenCV DCT |
| Texture features | scikit-image GLCM |
| Classifier | scikit-learn LogisticRegression |
| Generators | SDXL, FLUX.1-schnell, SD 1.5 (diffusers) |
| Real photos | COCO val2017 |
| Interface | Gradio |

---

## File Structure

```
cross-generator-ai-detector/
├── gradio_demo.py          Interactive demo
├── main.py                 CLI pipeline orchestrator
├── config.py               Centralized configuration
├── data_collection.py      Image generation (SDXL, FLUX, SD15, COCO)
├── feature_extraction.py   DCT + GLCM + CNN feature pipeline
├── model_training.py       3-fold cross-validation and evaluation
├── data/
│   ├── raw/
│   │   ├── sdxl/           150 SDXL images
│   │   ├── flux/           150 FLUX images
│   │   ├── sd15/           150 SD15 images
│   │   └── coco/           150 real photos
│   └── processed/          Extracted feature vectors (.npy)
└── outputs/                Results and metrics
```
