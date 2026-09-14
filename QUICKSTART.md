# ⚡ Quick Start Guide

Get up and running in 5 minutes.

## Step 1: Setup (2 min)

```bash
cd Cross_Generator_AI_Image_Detector

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## Step 2: Configure APIs (2 min)

Edit `.env` and add your API keys:

```bash
# Option A: Edit .env file
nano .env

# Option B: Export as environment variables
export OPENAI_API_KEY="sk-..."
export REPLICATE_API_TOKEN="..."
export IDEOGRAM_API_KEY="..."
```

**Get your keys:**
- OpenAI: https://platform.openai.com/account/api-keys
- Replicate: https://replicate.com/account/api-tokens  
- Ideogram: https://www.ideogram.ai/

## Step 3: (Optional) Download COCO Dataset

For real photos (negative class):

```bash
# Download val2017 images from cocodataset.org
# Place in: data/raw/coco_raw/val2017/
# Place annotations: data/raw/coco_raw/instances_val2017.json
```

## Step 4: Run Pipeline

### Full Pipeline (Recommended)
```bash
# Collects data, extracts features, trains, launches demo
python main.py full -y
```

### Step-by-Step
```bash
# 1. Collect 500 images per generator
python main.py collect

# 2. Extract features (DCT, GLCM, CNN)
python main.py extract

# 3. Train and evaluate with cross-validation
python main.py train

# 4. Launch interactive Gradio demo
python main.py demo
```

## Expected Output

After running the pipeline:

```
[✓] Data collected: 2000 AI images + 500 real photos
[✓] Features extracted: 2500 images × 1307 features
[✓] Cross-validation complete:
    - Ensemble (All Features): 75% accuracy ✓
    - CNN-Only Baseline: 55% accuracy
    - Frequency-Only Baseline: 60% accuracy
[✓] Gradio demo running at http://localhost:7860
```

## Using the Demo

1. Open http://localhost:7860
2. Upload any image (JPG, PNG)
3. Get:
   - Probability it's AI-generated
   - Confidence score
   - Feature importance breakdown
   - Prediction history

## File Structure After Run

```
data/
├── raw/
│   ├── dalle3/         # 500 DALL-E images
│   ├── stable_diffusion/  # 500 Stable Diffusion images
│   ├── ideogram/       # 500 Ideogram images
│   └── coco/          # 500 real photos
└── processed/
    ├── dalle3_features.npy
    ├── stable_diffusion_features.npy
    ├── ideogram_features.npy
    └── coco_features.npy

outputs/
└── cv_results_*.json  # Cross-validation results + plots
```

## Troubleshooting

### "No module named X"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Set in `.env`:
```
DEVICE=cpu
```

### API errors during data collection
- Check API keys are valid
- Check you have credits/quota
- Use test mode with fewer images

### "No features found"
Run data collection first:
```bash
python main.py collect -y
```

## Key Results to Check

After `python main.py train`, look for:

1. **Ensemble Accuracy**: Should be ~75%+
2. **Feature Importance**: All three feature types should contribute
3. **Per-Generator Accuracy**: Consistent across DALL-E, SD, Ideogram
4. **Cross-Validation Variance**: Low std dev shows stable results

## Next Steps

### For Research
- [ ] Try different feature combinations
- [ ] Add more generators
- [ ] Implement additional handcrafted features

### For Production
- [ ] Optimize inference latency
- [ ] Add batch prediction API
- [ ] Deploy to cloud (AWS/GCP/Azure)

### For Learning
- [ ] Read the feature engineering code
- [ ] Study the cross-validation approach
- [ ] Analyze feature importance plots

## Advanced Usage

### Batch Prediction
```python
from feature_extraction import FeatureExtractor
from model_training import DataLoader
import numpy as np

extractor = FeatureExtractor()
loader = DataLoader()

# Load trained model
features, labels, _ = loader.load_features()
# ... train your model ...

# Predict on images
for image_path in ["img1.png", "img2.png"]:
    features, _ = extractor.extract(image_path)
    pred = model.predict_proba([features])[0, 1]
    print(f"{image_path}: {pred:.1%} AI")
```

### Custom Feature Extraction
```python
from feature_extraction import DCTFrequencyExtractor, GLCMTextureExtractor
import numpy as np
from PIL import Image

img = Image.open("test.png").convert("RGB")
img_array = np.array(img)

dct_features = DCTFrequencyExtractor.extract(img_array)
glcm_features = GLCMTextureExtractor.extract(img_array)

print(f"DCT: {len(dct_features)} features")
print(f"GLCM: {len(glcm_features)} features")
```

## Key Files to Know

- **main.py** - CLI entry point
- **data_collection.py** - Generate images from 4 generators
- **feature_extraction.py** - DCT, GLCM, CNN features
- **model_training.py** - Cross-validation pipeline
- **gradio_demo.py** - Interactive web interface
- **config.py** - Centralized configuration
- **utils.py** - Helper utilities

## Performance Tips

| Task | Speed Up |
|------|----------|
| Feature extraction | Use GPU (set `DEVICE=cuda`) |
| Data collection | Parallel API requests (edit `data_collection.py`) |
| Training | Already fast (<1 min) |
| Demo inference | Uses cached features |

---

**Questions?** Check README.md for detailed documentation.

**Ready?** Run: `python main.py full -y` 🚀
