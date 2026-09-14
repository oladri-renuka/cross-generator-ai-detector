"""
Main orchestration script for AI Image Detector pipeline.
Coordinates data collection, feature extraction, training, and deployment.
"""

import sys
import argparse
from pathlib import Path


def setup_environment():
    """Verify and setup environment."""
    print("=" * 70)
    print("AI Image Detector - Environment Setup")
    print("=" * 70)

    required_dirs = ["data/raw", "data/processed", "models", "outputs", "logs"]

    for dir_name in required_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"[✓] {dir_name}/")

    print("\n[✓] Environment setup complete!\n")


def cmd_collect(args):
    """Collect data from generators."""
    print("\n" + "=" * 70)
    print("Step 1: Data Collection")
    print("=" * 70)
    print("\nThis step generates images from 4 AI generators and downloads COCO dataset.")
    print("\nSetup required:")
    print("  1. OpenAI API Key: https://platform.openai.com/account/api-keys")
    print("  2. Replicate API Token: https://replicate.com/account/api-tokens")
    print("  3. Ideogram API Key (or Midjourney): https://www.ideogram.ai/")
    print("  4. COCO Dataset: Download from https://cocodataset.org/")
    print("     - Place val2017/ images in data/raw/coco_raw/")
    print("     - Place instances_val2017.json in data/raw/coco_raw/")
    print("\nEnvironment variables needed:")
    print("  export OPENAI_API_KEY='...'")
    print("  export REPLICATE_API_TOKEN='...'")
    print("  export IDEOGRAM_API_KEY='...'")

    if not args.skip_confirm:
        resp = input("\nReady to proceed with data collection? (y/n): ").strip().lower()
        if resp != 'y':
            print("Skipping data collection.")
            return

    try:
        from data_collection import main as collect_main
        collect_main()
    except ImportError as e:
        print(f"[!] Error importing data_collection: {e}")
        print("    Run: pip install -r requirements.txt")


def cmd_extract(args):
    """Extract features from collected images."""
    print("\n" + "=" * 70)
    print("Step 2: Feature Extraction")
    print("=" * 70)
    print("\nExtracting DCT frequency, GLCM texture, and CNN features...")
    print("This will process all images in data/raw/ and save to data/processed/")

    try:
        from feature_extraction import main as extract_main
        extract_main()
    except ImportError as e:
        print(f"[!] Error importing feature_extraction: {e}")
        print("    Run: pip install -r requirements.txt")


def cmd_train(args):
    """Train and evaluate models."""
    print("\n" + "=" * 70)
    print("Step 3: Model Training & Evaluation")
    print("=" * 70)
    print("\nRunning 4-fold cross-validation with rotating held-out generators...")
    print("Will train 3 models:")
    print("  1. Ensemble (DCT + GLCM + CNN)")
    print("  2. CNN-Only Baseline")
    print("  3. Frequency-Only Baseline")

    try:
        from model_training import main as train_main
        train_main()
    except ImportError as e:
        print(f"[!] Error importing model_training: {e}")
        print("    Run: pip install -r requirements.txt")


def cmd_demo(args):
    """Launch Gradio demo."""
    print("\n" + "=" * 70)
    print("Step 4: Launch Demo")
    print("=" * 70)

    try:
        from gradio_demo import main as demo_main
        demo_main()
    except ImportError as e:
        print(f"[!] Error importing gradio_demo: {e}")
        print("    Run: pip install -r requirements.txt")


def cmd_full(args):
    """Run full pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE - AI Image Detector")
    print("=" * 70)

    setup_environment()

    print("\n[1/4] Collecting data...")
    cmd_collect(argparse.Namespace(skip_confirm=args.yes))

    print("\n[2/4] Extracting features...")
    cmd_extract(args)

    print("\n[3/4] Training models...")
    cmd_train(args)

    print("\n[4/4] Launching demo...")
    if not args.skip_demo:
        cmd_demo(args)


def main():
    parser = argparse.ArgumentParser(
        description="AI Image Detector - Cross-Generator Generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect      # Collect images from generators
  python main.py extract      # Extract features from collected images
  python main.py train        # Train and evaluate models
  python main.py demo         # Launch Gradio interface
  python main.py full -y      # Run complete pipeline (no confirmation)
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    collect_parser = subparsers.add_parser("collect", help="Collect images from generators")
    collect_parser.add_argument("--skip-confirm", "-y", action="store_true",
                              help="Skip confirmation prompts")
    collect_parser.set_defaults(func=cmd_collect)

    extract_parser = subparsers.add_parser("extract", help="Extract features from images")
    extract_parser.set_defaults(func=cmd_extract)

    train_parser = subparsers.add_parser("train", help="Train and evaluate models")
    train_parser.set_defaults(func=cmd_train)

    demo_parser = subparsers.add_parser("demo", help="Launch Gradio demo")
    demo_parser.set_defaults(func=cmd_demo)

    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--yes", "-y", action="store_true",
                           help="Skip confirmation prompts")
    full_parser.add_argument("--skip-demo", action="store_true",
                           help="Skip launching demo at end")
    full_parser.set_defaults(func=cmd_full)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if hasattr(args, 'func'):
        args.func(args)


if __name__ == "__main__":
    main()
