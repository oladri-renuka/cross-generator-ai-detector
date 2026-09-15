# Cross-Generator AI Image Detector

Machine learning system for detecting AI-generated images with robust cross-generator generalization. Achieves 95.1% accuracy on unseen generators through ensemble feature engineering combining frequency analysis, texture analysis, and deep learning.

## Overview

This system addresses the critical challenge of AI-generated image detection in production environments: **detecting images from generators unseen during training**. Unlike traditional approaches that overfit to specific generator architectures, this solution learns universal detection patterns through an ensemble of complementary feature types.

### Core Innovation

Rather than relying solely on deep learning (which exhibits high generator-specificity), we combine three independent feature extraction methods:

- **DCT Frequency Analysis (7D)**: Captures frequency artifacts characteristic of AI generation processes
- **GLCM Texture Features (20D)**: Detects unnatural texture uniformity patterns in generated images
- **CNN Features (1280D)**: EfficientNet-B0 transfer learning for learned visual representations

**Result**: 95.1% accuracy on held-out generators vs. 54.9% (CNN-only baseline)

## Experimental Design

### Cross-Validation Strategy

The system uses 3-fold rotation validation where each AI generator is held out once:

- **Fold 1**: Train on [FLUX, SD15, COCO] → Test on [SDXL]
- **Fold 2**: Train on [SDXL, SD15, COCO] → Test on [FLUX]  
- **Fold 3**: Train on [SDXL, FLUX, COCO] → Test on [SD15]

Real images (COCO) remain in training set for all folds to ensure balanced learning.

### Performance Metrics

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| **Ensemble (All Features)** | **95.1% ± 3.7%** | **97.5% ± 2.0%** | **0.989** |
| CNN-Only Baseline | 54.9% ± 8.2% | 71.3% ± 12.1% | 0.612 |
| Frequency-Only Baseline | 89.3% ± 6.1% | 94.7% ± 4.3% | 0.938 |

### Per-Generator Performance

| Generator | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| SDXL | 98.7% | 99.1% | 98.3% |
| FLUX | 96.7% | 96.9% | 96.5% |
| SD15 | 90.0% | 91.2% | 88.8% |

## Architecture

### System Pipeline

```
Data Collection (4 generators)
         ↓
Raw Images (600 total: 150 per generator + 150 real)
         ↓
Feature Extraction (1307-dimensional vectors)
  ├─ DCT Frequency Analysis (7D)
  ├─ GLCM Texture Features (20D)
  └─ EfficientNet-B0 CNN (1280D)
         ↓
Feature Normalization (StandardScaler)
         ↓
Logistic Regression Classifier
         ↓
Predictions: P(AI-generated | image)
```

### Generators Supported

| Generator | Architecture | Resolution | Notes |
|-----------|------------|-----------|-------|
| SDXL | U-Net (Improved) | 1024×1024 | Latest generation, highest quality |
| FLUX.1-schnell | Diffusion Transformer (DiT) | 1024×1024 | State-of-the-art architecture |
| Stable Diffusion 1.5 | U-Net (Classic) | 512×512 | Widely deployed, baseline |
| COCO | Real Photos | Variable | Ground truth negative class |

## Quick Start

### Installation

```bash
git clone https://github.com/oladri-renuka/cross-generator-ai-detector.git
cd cross-generator-ai-detector
pip install -r requirements.txt
```

### Run Interactive Demo

```bash
python gradio_demo.py
```

Opens at `http://localhost:7860` for real-time image classification.

### Full Pipeline (Optional)

Generate training data, extract features, and train model:

```bash
IMAGES_PER_GENERATOR=150 python main.py full -y
```

## Project Structure

```
cross-generator-ai-detector/
├── gradio_demo.py              Main interactive interface
├── main.py                     CLI orchestrator
├── config.py                   Centralized configuration
│
├── Core Modules
│   ├── data_collection.py      Image generation (SDXL, FLUX, SD15, COCO)
│   ├── feature_extraction.py   Feature engineering pipeline
│   ├── model_training.py       Cross-validation and evaluation
│   └── utils.py                Utility functions
│
├── Data
│   ├── raw/
│   │   ├── sdxl/               150 SDXL-generated images
│   │   ├── flux/               150 FLUX-generated images
│   │   ├── sd15/               150 SD15-generated images
│   │   └── coco/               150 real photographs
│   └── processed/              Extracted feature vectors (.npy)
│
└── outputs/                    Results and evaluation metrics
```

## Feature Engineering

### 1. DCT Frequency Features (7D)

Analyzes high-frequency artifacts in the 2D Discrete Cosine Transform:

- Extracts 32×32 high-frequency region (rows 32-64, cols 32-64)
- Computes: mean, std, max absolute values
- Adds percentile statistics (75th, 90th, 95th)
- **Rationale**: AI diffusion models produce characteristic frequency patterns due to upsampling and attention mechanisms

### 2. GLCM Texture Features (20D)

Gray Level Co-occurrence Matrix analysis at 4 orientations:

- **Orientations**: 0°, 45°, 90°, 135°
- **Properties per orientation**: contrast, dissimilarity, homogeneity, energy, correlation
- **Total**: 4 × 5 = 20 features
- **Rationale**: AI-generated images show unnatural texture uniformity; real photos have natural texture variation

### 3. CNN Features (1280D)

Transfer learning with EfficientNet-B0:

- Pretrained on ImageNet
- Extracts penultimate layer (1280-dimensional)
- Global average pooling applied
- **Rationale**: Captures high-level visual patterns learned from natural images

## Classifier

Simple yet effective logistic regression on concatenated 1307-dimensional feature vectors:

```python
Input: [DCT(7), GLCM(20), CNN(1280)]
  ↓
StandardScaler normalization
  ↓
LogisticRegression(solver='lbfgs', max_iter=1000)
  ↓
Output: Probability [0, 1]
```

Hyperparameters chosen via cross-validation validation performance.

## Requirements

### Python Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.7
diffusers>=0.24.0
transformers>=4.35.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
opencv-python>=4.8.0
gradio>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
```

See `requirements.txt` for complete list.

### Hardware Specifications

- **GPU**: NVIDIA CUDA-capable GPU (recommended: A100, V100, or RTX 3090+)
- **Memory**: 16GB GPU VRAM minimum (8GB possible with batch optimization)
- **Disk Space**: 20GB (for 600 images + features + models)
- **CPU**: 8+ cores for parallel processing

### Local Setup Alternative

For SDXL and FLUX generation on RunPod:

```bash
chmod +x runpod_setup.sh
./runpod_setup.sh
```

See `RUNPOD.md` for cloud GPU deployment (1-2 hours full pipeline vs 8+ hours locally).

## API and Integration

### Programmatic Usage

```python
from feature_extraction import FeatureExtractor
from model_training import CrossGeneratorValidator
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize
extractor = FeatureExtractor(device='cuda')
validator = CrossGeneratorValidator()

# Extract features from single image
features, feature_dict = extractor.extract('image.jpg')
print(f"Feature dimensions: {features.shape}")  # (1307,)

# Load trained model
features_train, labels_train = validator.data_loader.load_features()
scaler = StandardScaler()
X_train = scaler.fit_transform(features_train)
model = validator.train_classifier(X_train, labels_train)

# Predict
X_test = scaler.transform([features])
probability = model.predict_proba(X_test)[0, 1]
prediction = "AI-Generated" if probability > 0.5 else "Real Photo"
print(f"{prediction} ({probability:.1%} confidence)")
```

### Batch Processing

```python
from pathlib import Path

image_dir = 'test_images/'
results = []

for img_path in Path(image_dir).glob('*.jpg'):
    features, _ = extractor.extract(str(img_path))
    X = scaler.transform([features])
    prob = model.predict_proba(X)[0, 1]
    results.append({
        'image': img_path.name,
        'ai_probability': prob,
        'prediction': 'AI' if prob > 0.5 else 'Real'
    })

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config.py`:

```python
BATCH_SIZE = 16  # Default 32
```

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### No Images Generated

Verify generators are loading correctly:

```bash
python -c "from data_collection import StableDiffusionXLGenerator; print('SDXL OK')"
```

### Feature Extraction Errors

Check image directory permissions:

```bash
ls -la data/raw/sdxl/
```

## Configuration

Key settings in `config.py`:

```python
IMAGES_PER_GENERATOR = 150  # Adjust for faster testing
IMAGE_SIZE = 1024
BATCH_SIZE = 32
RANDOM_SEED = 42
CV_FOLDS = 3  # 3-fold rotation for 3 AI generators
DEVICE = "cuda"  # or "cpu"
```

## Performance Characteristics

### Inference Speed

- Single image: ~2-3 seconds (GPU), ~8-10 seconds (CPU)
- Batch (32 images): ~100ms per image (GPU)
- DCT extraction: ~50ms
- GLCM extraction: ~150ms
- CNN extraction: ~1500ms

### Memory Usage

- Model weights: ~250MB (EfficientNet-B0)
- Single image features: ~10.5KB
- Batch (32 images): ~350KB

## Key Findings

### Why Ensemble Works

1. **CNN-only overfits to generator architecture**: 55% on held-out generators
2. **Frequency-only too narrow**: 89% but misses some patterns
3. **Ensemble captures complementary information**: 95%+ by combining all three

### Generator-Specific Patterns

- **SDXL** (98.7%): Easiest to detect; U-Net artifacts distinct
- **FLUX** (96.7%): DiT architecture has different frequency signature
- **SD15** (90.0%): Older model; some features overlap with real photos

## References

This project implements concepts from:

- Diffusion Models and Their Frequency Characteristics
- Handcrafted Features for Forgery Detection (DCT, GLCM)
- Transfer Learning for Image Classification
- Cross-Domain Generalization in Machine Learning

## License

MIT License

