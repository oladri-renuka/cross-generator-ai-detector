"""
Data collection from multiple AI image generators and COCO dataset.
Generates diverse prompts for robust feature extraction.
"""

import os
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm
import cv2

try:
    import replicate
except ImportError:
    pass

try:
    from diffusers import StableDiffusionXLPipeline
    import torch
except ImportError:
    pass


class PromptGenerator:
    """Generate diverse prompts for AI image generation."""

    CATEGORIES = {
        "landscapes": [
            "A serene mountain landscape at sunset with snow peaks",
            "Tropical beach with crystal clear turquoise water and palm trees",
            "Dense forest with fog and ancient trees",
            "Desert dunes at golden hour with camel caravan",
            "Aurora borealis over Arctic wilderness",
        ],
        "portraits": [
            "Portrait of a woman with ethereal lighting and flowing hair",
            "Professional headshot of a bearded man in business attire",
            "Child playing in sunlit field with joyful expression",
            "Elderly person with wise expression and detailed facial features",
            "Cyberpunk character with neon makeup and tech aesthetic",
        ],
        "objects": [
            "Vintage typewriter on wooden desk with documents",
            "Fresh fruit arrangement in a ceramic bowl",
            "Ornate jewelry and gemstones on velvet surface",
            "Antique clock with intricate mechanical details",
            "Colorful books stacked on a shelf",
        ],
        "abstract": [
            "Abstract liquid paint mixing in water with vibrant colors",
            "Geometric shapes and fractals in cosmic space",
            "Swirling nebula with stars and cosmic dust",
            "Abstract oil painting with bold brushstrokes",
            "Kaleidoscopic pattern with symmetrical colors",
        ],
        "architecture": [
            "Modern glass skyscraper at night with city lights",
            "Historic stone cathedral with intricate details",
            "Minimalist interior design with natural light",
            "Japanese temple surrounded by cherry blossom trees",
            "Futuristic space station architecture",
        ],
    }

    @classmethod
    def generate_prompts(cls, num_prompts: int = 500) -> List[str]:
        """Generate diverse prompts for all categories."""
        prompts = []
        per_category = num_prompts // len(cls.CATEGORIES)

        for category, samples in cls.CATEGORIES.items():
            for _ in range(per_category):
                base_prompt = random.choice(samples)
                quality_modifiers = random.choice([
                    ", highly detailed, professional photography",
                    ", cinematic lighting, 4K",
                    ", oil painting style, masterpiece",
                    ", digital art, trending on artstation",
                    ", studio lighting, high quality",
                ])
                prompts.append(base_prompt + quality_modifiers)

        return prompts[:num_prompts]


class FluxSchnellGenerator:
    """Generate images using FLUX.1-schnell locally (FREE!)."""

    def __init__(self, device: str = None):
        self.device = device or os.getenv("DEVICE", "cuda")
        print(f"[*] Loading FLUX.1-schnell model on {self.device}...")

        try:
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16
            )
            self.pipe = self.pipe.to(self.device)
            print(f"[✓] FLUX.1-schnell model loaded successfully")
        except Exception as e:
            print(f"[!] Error loading FLUX: {e}")
            self.pipe = None

    def generate(self, prompt: str) -> Tuple[Image.Image, Dict]:
        """Generate single image from prompt."""
        if not self.pipe:
            print("Error: Pipeline not loaded")
            return None, {}

        try:
            image = self.pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=4,
                max_sequence_length=512
            ).images[0]

            metadata = {
                "generator": "flux_schnell",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "resolution": "1024x1024",
            }

            return image, metadata
        except Exception as e:
            print(f"FLUX generation error: {e}")
            return None, {}

    def generate_batch(self, prompts: List[str], output_dir: str) -> int:
        """Generate multiple images locally."""
        os.makedirs(output_dir, exist_ok=True)
        generated = 0
        metadata_list = []

        for i, prompt in enumerate(tqdm(prompts, desc="FLUX.1-schnell generation (local)")):
            try:
                image, metadata = self.generate(prompt)
                if image:
                    img_path = os.path.join(output_dir, f"flux__{i:04d}.png")
                    image.save(img_path)
                    metadata_list.append(metadata)
                    generated += 1
            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                continue

        with open(os.path.join(output_dir, "flux_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return generated


class FluxProGenerator:
    """Generate images using Flux Pro 2 via OpenRouter."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/images"

    def generate(self, prompt: str) -> Tuple[Image.Image, Dict]:
        """Generate single image from prompt."""
        try:
            if not self.api_key:
                print("ERROR: OPENROUTER_API_KEY not set in .env")
                return None, {}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "black-forest-labs/flux.2-pro",
                "prompt": prompt,
                "n": 1,
            }

            response = requests.post(self.base_url, json=payload, headers=headers, timeout=120)

            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    image_data = data["data"][0]

                    if "b64_json" in image_data:
                        import base64
                        image_bytes = base64.b64decode(image_data["b64_json"])
                        image = Image.open(BytesIO(image_bytes))
                    elif "url" in image_data:
                        img_response = requests.get(image_data["url"], timeout=10)
                        image = Image.open(BytesIO(img_response.content))
                    else:
                        print(f"Unknown image format in response")
                        return None, {}

                    metadata = {
                        "generator": "flux_pro",
                        "prompt": prompt,
                        "timestamp": datetime.now().isoformat(),
                        "resolution": "1024x1024",
                    }

                    return image, metadata
            else:
                print(f"OpenRouter API error: {response.status_code} - {response.text[:200]}")
                return None, {}
        except Exception as e:
            print(f"Flux Pro error: {e}")
            return None, {}

    def generate_batch(self, prompts: List[str], output_dir: str) -> int:
        """Generate multiple images with rate limiting."""
        os.makedirs(output_dir, exist_ok=True)
        generated = 0
        metadata_list = []

        for i, prompt in enumerate(tqdm(prompts, desc="DALL-E generation")):
            try:
                result = self.generate(prompt)
                if result and len(result) == 2:
                    image, metadata = result
                    if image:
                        img_path = os.path.join(output_dir, f"dalle__{i:04d}.png")
                        image.save(img_path)
                        metadata_list.append(metadata)
                        generated += 1
            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                continue

            time.sleep(1)

        with open(os.path.join(output_dir, "dalle_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return generated


class StableDiffusionXLGenerator:
    """Generate images using Stable Diffusion XL locally (FREE!)."""

    def __init__(self, device: str = None):
        self.device = device or os.getenv("DEVICE", "mps")
        print(f"[*] Loading SDXL model on {self.device}...")

        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.pipe = self.pipe.to(self.device)
            print(f"[✓] SDXL model loaded successfully")
        except Exception as e:
            print(f"[!] Error loading SDXL: {e}")
            print(f"[!] Install: pip install diffusers transformers accelerate safetensors")
            self.pipe = None

    def generate(self, prompt: str) -> Tuple[Image.Image, Dict]:
        """Generate single image from prompt."""
        if not self.pipe:
            print("Error: Pipeline not loaded")
            return None, {}

        try:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=1024,
                width=1024
            ).images[0]

            metadata = {
                "generator": "sdxl_local",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "resolution": "1024x1024",
            }

            return image, metadata
        except Exception as e:
            print(f"SDXL generation error: {e}")
            return None, {}

    def generate_batch(self, prompts: List[str], output_dir: str) -> int:
        """Generate multiple images locally."""
        os.makedirs(output_dir, exist_ok=True)
        generated = 0
        metadata_list = []

        for i, prompt in enumerate(tqdm(prompts, desc="SDXL generation (local)")):
            try:
                image, metadata = self.generate(prompt)
                if image:
                    img_path = os.path.join(output_dir, f"sdxl__{i:04d}.png")
                    image.save(img_path)
                    metadata_list.append(metadata)
                    generated += 1
            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                continue

        with open(os.path.join(output_dir, "sdxl_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return generated


class IdeogramGenerator:
    """Generate images using Ideogram (Midjourney alternative)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("IDEOGRAM_API_KEY")
        self.base_url = "https://api.ideogram.ai/generate"

    def generate(self, prompt: str) -> Tuple[Image.Image, Dict]:
        """Generate single image from prompt."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "prompt": prompt,
                "image_quality": "QUALITY_STANDARD",
                "aspect_ratio": "ASPECT_1_1",
            }

            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "images" in data and len(data["images"]) > 0:
                    image_data = data["images"][0].get("url")
                    if image_data.startswith("data:"):
                        image_bytes = image_data.split(",")[1]
                        image = Image.open(BytesIO(__import__("base64").b64decode(image_bytes)))
                    else:
                        img_response = requests.get(image_data, timeout=10)
                        image = Image.open(BytesIO(img_response.content))

                    metadata = {
                        "generator": "ideogram",
                        "prompt": prompt,
                        "timestamp": datetime.now().isoformat(),
                        "resolution": "1024x1024",
                    }

                    return image, metadata
        except Exception as e:
            print(f"Ideogram error: {e}")

        return None, {}

    def generate_batch(self, prompts: List[str], output_dir: str) -> int:
        """Generate multiple images with rate limiting."""
        os.makedirs(output_dir, exist_ok=True)
        generated = 0
        metadata_list = []

        for i, prompt in enumerate(tqdm(prompts, desc="Ideogram generation")):
            image, metadata = self.generate(prompt)

            if image:
                img_path = os.path.join(output_dir, f"ideogram__{i:04d}.png")
                image.save(img_path)
                metadata_list.append(metadata)
                generated += 1

            time.sleep(1)

        with open(os.path.join(output_dir, "ideogram_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return generated


class COCODatasetLoader:
    """Load real photos from COCO dataset."""

    @staticmethod
    def download_and_prepare(output_dir: str, num_images: int = 500):
        """Download COCO dataset and extract real photos."""
        os.makedirs(output_dir, exist_ok=True)

        coco_dir = os.path.join(output_dir, "coco_raw")
        os.makedirs(coco_dir, exist_ok=True)

        print(f"Note: COCO dataset should be downloaded from cocodataset.org")
        print(f"Expected structure: {coco_dir}/val2017/ with images")
        print(f"And {coco_dir}/instances_val2017.json for annotations")

        return coco_dir

    @staticmethod
    def process_coco_images(coco_dir: str, output_dir: str, num_images: int = 500):
        """Process downloaded COCO images."""
        images_dir = os.path.join(coco_dir, "val2017")

        if not os.path.exists(images_dir):
            print(f"ERROR: COCO images directory not found at {images_dir}")
            return 0

        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        selected = random.sample(image_files, min(num_images, len(image_files)))

        metadata_list = []

        for i, filename in enumerate(tqdm(selected, desc="Processing COCO images")):
            src = os.path.join(images_dir, filename)
            dst = os.path.join(output_dir, f"coco__{i:04d}.jpg")

            img = Image.open(src)
            img = img.convert("RGB")

            if img.size[0] > 1024 or img.size[1] > 1024:
                img.thumbnail((1024, 1024))

            img.save(dst)

            metadata_list.append({
                "generator": "coco_real",
                "source_file": filename,
                "timestamp": datetime.now().isoformat(),
                "resolution": f"{img.size[0]}x{img.size[1]}",
            })

        with open(os.path.join(output_dir, "coco_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return len(metadata_list)


class SD15Generator:
    """Generate images using Stable Diffusion 1.5 locally (FREE!)."""

    def __init__(self, device: str = None):
        self.device = device or os.getenv("DEVICE", "cuda")
        print(f"[*] Loading SD 1.5 model on {self.device}...")

        try:
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
            print(f"[✓] SD 1.5 model loaded successfully")
        except Exception as e:
            print(f"[!] Error loading SD 1.5: {e}")
            self.pipe = None

    def generate(self, prompt: str) -> Tuple[Image.Image, Dict]:
        """Generate single image from prompt."""
        if not self.pipe:
            print("Error: Pipeline not loaded")
            return None, {}

        try:
            image = self.pipe(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]

            metadata = {
                "generator": "sd15",
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "resolution": "512x512",
            }

            return image, metadata
        except Exception as e:
            print(f"SD 1.5 generation error: {e}")
            return None, {}

    def generate_batch(self, prompts: List[str], output_dir: str) -> int:
        """Generate multiple images locally."""
        os.makedirs(output_dir, exist_ok=True)
        generated = 0
        metadata_list = []

        for i, prompt in enumerate(tqdm(prompts, desc="SD 1.5 generation (local)")):
            try:
                image, metadata = self.generate(prompt)
                if image:
                    img_path = os.path.join(output_dir, f"sd15__{i:04d}.png")
                    image.save(img_path)
                    metadata_list.append(metadata)
                    generated += 1
            except Exception as e:
                print(f"Error processing prompt {i}: {e}")
                continue

        with open(os.path.join(output_dir, "sd15_metadata.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        return generated


def main():
    """Main data collection pipeline."""
    from config import IMAGES_PER_GENERATOR

    # Setup
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)

    prompts = PromptGenerator.generate_prompts(IMAGES_PER_GENERATOR)

    print("=" * 60)
    print("AI Image Detector - Data Collection")
    print("=" * 60)

    # Generator 1: SDXL (Improved U-Net Diffusion)
    print("\n[1/4] Generating SDXL images (local, U-Net based)...")
    sdxl_gen = StableDiffusionXLGenerator()
    sdxl_gen.generate_batch(prompts, os.path.join(raw_data_dir, "sdxl"))

    # Generator 2: FLUX.1-schnell (Diffusion Transformer - NEW ARCHITECTURE!)
    print("\n[2/4] Generating FLUX.1-schnell images (local, DiT based)...")
    flux_gen = FluxSchnellGenerator()
    flux_gen.generate_batch(prompts, os.path.join(raw_data_dir, "flux"))

    # Generator 3: SD 1.5 (Original U-Net Diffusion)
    print("\n[3/4] Generating SD 1.5 images (local, classic U-Net)...")
    sd15_gen = SD15Generator()
    sd15_gen.generate_batch(prompts, os.path.join(raw_data_dir, "sd15"))

    # Generator 4: COCO (Real Photos - Ground Truth)
    print("\n[4/4] Processing COCO real photos (ground truth)...")
    coco_prep_dir = COCODatasetLoader.download_and_prepare(raw_data_dir)
    COCODatasetLoader.process_coco_images(coco_prep_dir, os.path.join(raw_data_dir, "coco"))

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Total images collected in: {raw_data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
