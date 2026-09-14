"""
Centralized configuration for AI Image Detector project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
IDEOGRAM_API_KEY = os.getenv("IDEOGRAM_API_KEY")

# Model Configuration
DEVICE = os.getenv("DEVICE", "cuda")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Data Collection
IMAGES_PER_GENERATOR = int(os.getenv("IMAGES_PER_GENERATOR", "500"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Training
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "4"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
HIDDEN_UNITS = 256
LEARNING_RATE = 1e-3
EPOCHS = 50

# Feature Extraction
DCT_HIGH_FREQ_START = 32
DCT_HIGH_FREQ_END = 64
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, 1, 2, 3]

# Feature Dimensions
DCT_FEATURES_DIM = 7
GLCM_FEATURES_DIM = 20
CNN_FEATURES_DIM = 1280
TOTAL_FEATURES_DIM = DCT_FEATURES_DIM + GLCM_FEATURES_DIM + CNN_FEATURES_DIM

# Generators
GENERATORS = {
    "sdxl": {
        "name": "SDXL (U-Net Diffusion, Improved)",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "size": "1024x1024",
        "architecture": "U-Net",
    },
    "flux": {
        "name": "FLUX.1-schnell (Diffusion Transformer)",
        "model": "black-forest-labs/FLUX.1-schnell",
        "size": "1024x1024",
        "architecture": "DiT (Diffusion Transformer)",
    },
    "sd15": {
        "name": "SD 1.5 (U-Net Diffusion, Classic)",
        "model": "runwayml/stable-diffusion-v1-5",
        "size": "512x512",
        "architecture": "U-Net",
    },
}

REAL_LABEL = "coco"

# Classifier
CLASSIFIER_TYPE = "logistic_regression"
CLASSIFIER_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# Gradio Configuration
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "True").lower() == "true"

# Thresholds
AI_PROBABILITY_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.7

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)

    print("\nDirectories:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Outputs: {OUTPUTS_DIR}")

    print("\nData Collection:")
    print(f"  Images per generator: {IMAGES_PER_GENERATOR}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Total images: {IMAGES_PER_GENERATOR * 4 + IMAGES_PER_GENERATOR} (4 AI + 1 Real)")

    print("\nFeatures:")
    print(f"  DCT features: {DCT_FEATURES_DIM}")
    print(f"  GLCM features: {GLCM_FEATURES_DIM}")
    print(f"  CNN features: {CNN_FEATURES_DIM}")
    print(f"  Total features: {TOTAL_FEATURES_DIM}")

    print("\nTraining:")
    print(f"  Cross-validation folds: {CV_FOLDS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Device: {DEVICE}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_config()
