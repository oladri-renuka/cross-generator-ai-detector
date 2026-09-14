"""
Feature extraction combining DCT frequency features, GLCM texture features,
and CNN features for robust AI image detection across generators.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from timm import create_model
from tqdm import tqdm
import json


class DCTFrequencyExtractor:
    """Extract frequency features using 2D Discrete Cosine Transform."""

    @staticmethod
    def extract(image_array: np.ndarray) -> np.ndarray:
        """Extract DCT frequency features from image."""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        gray = cv2.resize(gray, (256, 256))

        dct = cv2.dct(np.float32(gray) / 255.0)

        high_freq_region = dct[32:64, 32:64]

        features = np.array([
            np.mean(np.abs(high_freq_region)),
            np.std(np.abs(high_freq_region)),
            np.max(np.abs(high_freq_region)),
            np.sum(np.abs(high_freq_region)),
            np.percentile(np.abs(high_freq_region), 75),
            np.percentile(np.abs(high_freq_region), 90),
            np.percentile(np.abs(high_freq_region), 95),
        ])

        return features


class GLCMTextureExtractor:
    """Extract texture features using Gray Level Co-occurrence Matrix."""

    @staticmethod
    def extract(image_array: np.ndarray, distances: List[int] = None) -> np.ndarray:
        """Extract GLCM texture features from image."""
        if distances is None:
            distances = [1]

        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        gray = cv2.resize(gray, (128, 128))
        gray = (gray / 256).astype(np.uint8)

        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        features = []

        for distance in distances:
            glcm = graycomatrix(gray, distances=[distance], angles=angles, levels=256)

            for angle_idx in range(glcm.shape[3]):
                glcm_slice = glcm[:, :, 0, angle_idx]

                contrast = graycoprops(glcm_slice, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm_slice, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm_slice, 'homogeneity')[0, 0]
                energy = graycoprops(glcm_slice, 'energy')[0, 0]
                correlation = graycoprops(glcm_slice, 'correlation')[0, 0]

                features.extend([contrast, dissimilarity, homogeneity, energy, correlation])

        return np.array(features)


class CNNFeatureExtractor:
    """Extract learned features from EfficientNet-B0."""

    def __init__(self, model_name: str = "efficientnet_b0", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract CNN features from image."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.cpu().numpy().flatten()


class FeatureExtractor:
    """Unified feature extractor combining all three feature types."""

    def __init__(self, device: str = None):
        self.dct_extractor = DCTFrequencyExtractor()
        self.glcm_extractor = GLCMTextureExtractor()
        self.cnn_extractor = CNNFeatureExtractor(device=device)

        self.feature_dims = {
            "dct": 7,
            "glcm": 20,
            "cnn": 1280,
        }

    def extract(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Extract all features from single image."""
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        dct_features = self.dct_extractor.extract(img_array)
        glcm_features = self.glcm_extractor.extract(img_array)
        cnn_features = self.cnn_extractor.extract(img)

        combined_features = np.concatenate([dct_features, glcm_features, cnn_features])

        feature_dict = {
            "dct": dct_features.tolist(),
            "glcm": glcm_features.tolist(),
            "cnn": cnn_features.tolist(),
            "combined": combined_features.tolist(),
        }

        return combined_features, feature_dict

    def extract_batch(self, image_dir: str, output_path: str = None, checkpoint_interval: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from batch of images with checkpointing."""
        image_files = sorted([
            f for f in Path(image_dir).glob("*.png") + Path(image_dir).glob("*.jpg")
            if f.is_file()
        ])

        all_features = []
        metadata = []
        checkpoint_dir = Path(output_path).parent / f"{Path(output_path).stem}_checkpoints" if output_path else None

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for i, img_file in enumerate(tqdm(image_files, desc=f"Extracting features from {Path(image_dir).name}")):
            try:
                features, feature_dict = self.extract(str(img_file))
                all_features.append(features)

                metadata.append({
                    "file": img_file.name,
                    "feature_dims": {k: len(v) for k, v in feature_dict.items()},
                })

                # Save checkpoint every N images
                if (i + 1) % checkpoint_interval == 0 and checkpoint_dir:
                    checkpoint_num = (i + 1) // checkpoint_interval
                    checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_num:03d}.npy"
                    np.save(checkpoint_path, np.array(all_features))

                    meta_checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_num:03d}_metadata.json"
                    with open(meta_checkpoint_path, "w") as f:
                        json.dump(metadata, f)

            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        all_features = np.array(all_features)

        if output_path:
            np.save(output_path, all_features)
            meta_path = output_path.replace(".npy", "_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            # Clean up checkpoints after successful completion
            if checkpoint_dir and checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
                print(f"[✓] Cleaned up checkpoints after successful extraction")

        return all_features, metadata

    def get_feature_size(self) -> int:
        """Get total feature dimension."""
        return sum(self.feature_dims.values())


def main():
    """Test feature extraction pipeline."""

    print("=" * 60)
    print("Feature Extraction Pipeline")
    print("=" * 60)

    extractor = FeatureExtractor()

    print(f"\nFeature dimensions:")
    print(f"  DCT: {extractor.feature_dims['dct']}")
    print(f"  GLCM: {extractor.feature_dims['glcm']}")
    print(f"  CNN (EfficientNet-B0): {extractor.feature_dims['cnn']}")
    print(f"  Total: {extractor.get_feature_size()}")

    print("\nProcessing data directories...")

    generators = ["dalle3", "stable_diffusion", "ideogram", "coco"]

    for generator in generators:
        image_dir = f"data/raw/{generator}"

        if Path(image_dir).exists():
            print(f"\n[*] Processing {generator}...")
            features, metadata = extractor.extract_batch(
                image_dir,
                f"data/processed/{generator}_features.npy"
            )
            print(f"    Extracted {len(features)} images, shape: {features.shape}")
        else:
            print(f"\n[!] Skipping {generator} - directory not found")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
