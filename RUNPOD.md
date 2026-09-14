# Running on RunPod - Quick Start

## Why RunPod?

| Metric | Local M4 | RunPod A100 |
|--------|----------|-----------|
| **Generation Speed** | 6-7 sec/image | 2-3 sec/image |
| **500 SDXL images** | ~50 min | ~15 min |
| **All 4 generators** | ~8+ hours | ~1-1.5 hours |
| **Total Pipeline** | ~9+ hours | ~2 hours |
| **Cost** | $0 | ~$2-3 |

---

## Setup on RunPod

### 1. Launch RunPod Pod

1. Go to https://runpod.io
2. Start an **A100** or **H100** on-demand pod (~$0.30-0.50/hr)
3. Click "Connect" → Get SSH connection string

### 2. Clone & Setup

```bash
# SSH into RunPod
ssh username@pod_id.runpod.io

# Clone your repo
git clone https://github.com/oladri-renuka/cross-generator-ai-detector.git
cd cross-generator-ai-detector

# Make setup script executable
chmod +x runpod_setup.sh

# Run setup & pipeline
./runpod_setup.sh
```

**That's it!** The script will:
- ✅ Install all dependencies
- ✅ Verify CUDA/GPU
- ✅ Generate 4 × 500 AI images
- ✅ Extract 1307-dim features per image
- ✅ Train 4-fold cross-validation
- ✅ Save results

---

## What Happens

```
[1/4] SDXL generation:        ~15 min (500 images)
[2/4] FLUX.1-schnell:         ~12 min (500 images, fast)
[3/4] SD 1.5 generation:      ~20 min (500 images)
[4/4] COCO processing:        ~2 min  (500 images)

Feature Extraction:           ~10 min (2000 images)
4-Fold Cross-Validation:      ~5 min
=====================================
Total:                        ~1.5 hours
```

---

## Download Results

### While Pod is Running

```bash
# In new terminal on your machine
scp -r username@pod_id.runpod.io:/workspace/cross-generator-ai-detector/data ./
scp -r username@pod_id.runpod.io:/workspace/cross-generator-ai-detector/outputs ./
```

### After Completion

```bash
# Download everything
rsync -avz username@pod_id.runpod.io:/workspace/cross-generator-ai-detector/ ./results/
```

---

## Monitor Progress

```bash
# SSH into RunPod, check image count
watch -n 5 'ls data/raw/sdxl/ | wc -l'
watch -n 5 'ls data/raw/flux/ | wc -l'
watch -n 5 'ls data/raw/sd15/ | wc -l'
```

---

## Environment

The `.env.runpod` file is pre-configured for RunPod:

```bash
DEVICE=cuda          # Uses GPU
NUM_WORKERS=8        # Parallel processing
```

No API keys needed! All 3 AI generators run **locally** on the GPU.

---

## Architecture Diversity

This is what makes the cross-generator claim strong:

| Generator | Architecture | Base Model | Context |
|-----------|-------------|-----------|---------|
| **SDXL** | U-Net (Improved) | Diffusion | Newest, highest quality |
| **FLUX** | Diffusion Transformer (DiT) | NEW | Newest architecture |
| **SD 1.5** | U-Net (Classic) | Diffusion | Classic, widely used |
| **COCO** | N/A | Real Photos | Ground truth |

**4-Fold CV**: Hold out each AI generator once, train on remaining 2 AI + real photos.

---

## Cost Estimate

- **A100 Pod**: ~$0.30/hr × 2 hours = ~$0.60
- **H100 Pod**: ~$0.50/hr × 1.5 hours = ~$0.75

Very cheap compared to 8+ hours of local generation!

---

## Tips

### Speed Up
- Reduce `IMAGES_PER_GENERATOR` in `.env` for testing
- FLUX is already super fast (~2 sec/image on A100)

### Resume If Interrupted
- Checkpoints saved every 50 images
- Restart pod, run `./runpod_setup.sh` again
- Skips already-generated images

### Debug
```bash
# Check GPU
nvidia-smi

# Check installation
python -c "from diffusers import StableDiffusionXLPipeline; print('✓ Installed')"

# Run just one generator
python -c "from data_collection import StableDiffusionXLGenerator; gen = StableDiffusionXLGenerator(); print('✓ SDXL working')"
```

---

## After Pipeline

Results in:
- `data/raw/` - 2000 AI images + 500 real photos
- `data/processed/` - Features for each image
- `outputs/` - Cross-validation results (accuracy, AUC, plots)

Download and analyze locally!
