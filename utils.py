"""
Utility functions for AI Image Detector project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import LOG_LEVEL, LOG_FORMAT, OUTPUTS_DIR, LOGS_DIR


def setup_logging(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """Setup logging for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    os.makedirs(LOGS_DIR, exist_ok=True)

    fh = logging.FileHandler(LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(getattr(logging, level))

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))

    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_json(data: Dict, path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Dict:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_metrics(metrics: Dict, name: str) -> str:
    """Save metrics and return path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"{name}_{timestamp}.json"

    save_json(metrics, str(path))
    print(f"[✓] Metrics saved: {path}")

    return str(path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Confusion Matrix") -> plt.Figure:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'AI'],
                yticklabels=['Real', 'AI'])

    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)

    return fig


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                  title: str = "ROC Curve", label: str = None) -> plt.Figure:
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='#2ecc71', lw=2,
           label=f'ROC curve (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return fig


def plot_feature_importance(coefficients: np.ndarray,
                           feature_names: List[str],
                           title: str = "Feature Importance") -> plt.Figure:
    """Plot feature importance."""
    indices = np.argsort(np.abs(coefficients))[-15:]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(range(len(indices)), np.abs(coefficients[indices]))

    selected_names = [feature_names[i] for i in indices]
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(selected_names)
    ax.set_xlabel('Absolute Coefficient Value', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    return fig


def print_summary_table(results: Dict[str, Dict]) -> None:
    """Print results summary as formatted table."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    print(f"{'Model':<20} {'Accuracy':<12} {'AUC-ROC':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 90)

    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics.get('accuracy', 0):<12.3f} "
              f"{metrics.get('auc', 0):<12.3f} "
              f"{metrics.get('precision', 0):<12.3f} "
              f"{metrics.get('recall', 0):<12.3f} "
              f"{metrics.get('f1', 0):<12.3f}")

    print("=" * 90 + "\n")


def check_data_availability() -> Tuple[bool, Dict[str, int]]:
    """Check available data."""
    from config import RAW_DATA_DIR, GENERATORS, REAL_LABEL

    availability = {}

    for gen_name in list(GENERATORS.keys()) + [REAL_LABEL]:
        gen_dir = RAW_DATA_DIR / gen_name
        num_images = len(list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg")))
        availability[gen_name] = num_images

    all_available = all(count >= 500 for count in availability.values())

    return all_available, availability


def print_data_status() -> None:
    """Print data availability status."""
    available, counts = check_data_availability()

    print("\n" + "=" * 70)
    print("Data Availability Status")
    print("=" * 70)

    from config import GENERATORS, REAL_LABEL

    all_gens = list(GENERATORS.keys()) + [REAL_LABEL]

    for gen in all_gens:
        count = counts.get(gen, 0)
        status = "✓" if count >= 500 else "✗"
        gen_label = GENERATORS.get(gen, {}).get("name", gen).upper()
        print(f"  {status} {gen_label:<20} {count:>4} images")

    print("\n" + ("✓" if available else "✗") + " Ready to proceed: " +
          ("Yes" if available else "Need more data"))
    print("=" * 70 + "\n")


def validate_environment() -> bool:
    """Validate required environment variables and dependencies."""
    import importlib

    print("\n" + "=" * 70)
    print("Environment Validation")
    print("=" * 70)

    # Check Python packages
    required_packages = [
        'torch', 'torchvision', 'timm', 'cv2', 'numpy',
        'sklearn', 'skimage', 'PIL', 'gradio'
    ]

    all_good = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            all_good = False

    # Check API keys (optional - only needed for data collection)
    from config import OPENAI_API_KEY, REPLICATE_API_TOKEN, IDEOGRAM_API_KEY

    print("\nAPI Keys (optional - only needed for data collection):")
    print(f"{'✓' if OPENAI_API_KEY else '✗'} OpenAI API Key")
    print(f"{'✓' if REPLICATE_API_TOKEN else '✗'} Replicate API Token")
    print(f"{'✓' if IDEOGRAM_API_KEY else '✗'} Ideogram API Key")

    print("\n" + ("✓" if all_good else "✗") + " Environment validation: " +
          ("PASS" if all_good else "FAIL"))
    print("=" * 70 + "\n")

    return all_good


def get_feature_names(include_components: List[str] = None) -> List[str]:
    """Get feature names for interpretability."""
    if include_components is None:
        include_components = ['dct', 'glcm', 'cnn']

    names = []

    if 'dct' in include_components:
        names.extend([
            "DCT_mean", "DCT_std", "DCT_max", "DCT_sum",
            "DCT_p75", "DCT_p90", "DCT_p95"
        ])

    if 'glcm' in include_components:
        angles = ["0°", "45°", "90°", "135°"]
        properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

        for angle in angles:
            for prop in properties:
                names.append(f"GLCM_{angle}_{prop}")

    if 'cnn' in include_components:
        names.extend([f"CNN_feat_{i}" for i in range(1280)])

    return names


def format_results_for_display(results: Dict) -> str:
    """Format results for human-readable display."""
    output = []

    output.append("=" * 70)
    output.append("COMPREHENSIVE RESULTS SUMMARY")
    output.append("=" * 70)

    for model_type, folds in results.items():
        output.append(f"\n{model_type.upper()}")
        output.append("-" * 70)

        accuracies = [f['accuracy'] for f in folds]
        aucs = [f['auc'] for f in folds]

        output.append(f"Accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
        output.append(f"AUC-ROC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

        for i, fold in enumerate(folds, 1):
            output.append(f"  Fold {i} ({fold['test_generator']:<20}): "
                        f"Acc={fold['accuracy']:.1%}, AUC={fold['auc']:.3f}")

    output.append("\n" + "=" * 70)

    return "\n".join(output)


if __name__ == "__main__":
    print("Testing utilities...")

    validate_environment()
    print_data_status()

    logger = setup_logging("test")
    logger.info("Logging system working")

    print("\n✓ All utilities functional")
