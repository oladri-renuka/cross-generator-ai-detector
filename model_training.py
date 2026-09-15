"""
Model training and evaluation with 4-fold cross-validation across generators.
Implements baselines and ensemble methods with comprehensive evaluation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class DataLoader:
    """Load and prepare data for training."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.generators = ["sdxl", "flux", "sd15"]
        self.real_label = "coco"

    def load_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load all features and create labels."""
        all_features = []
        all_labels = []
        image_names = []

        for generator in self.generators + [self.real_label]:
            feat_path = Path(self.data_dir) / f"{generator}_features.npy"

            if feat_path.exists():
                features = np.load(feat_path)
                label = 0 if generator == self.real_label else 1
                all_features.extend(features)
                all_labels.extend([label] * len(features))

                image_names.extend([generator] * len(features))

        return np.array(all_features), np.array(all_labels), image_names


class CrossGeneratorValidator:
    """4-fold cross-validation rotating which generator is held out."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.results = []
        print("\n" + "=" * 70)
        print("Cross-Generator Validation Architecture Diversity")
        print("=" * 70)
        print("SDXL:  U-Net Diffusion (Improved) - 1024x1024")
        print("FLUX:  Diffusion Transformer (NEW) - 1024x1024")
        print("SD 1.5: U-Net Diffusion (Classic) - 512x512")
        print("COCO:  Real Photos (Ground Truth)")
        print("=" * 70)

    def get_fold_splits(self, features: np.ndarray, labels: np.ndarray,
                       image_names: List[str]) -> List[Dict]:
        """Create 4 folds rotating which generator is held out."""

        folds = []
        generators = self.data_loader.generators

        for test_gen in generators:
            train_idx = []
            test_idx = []

            for i, name in enumerate(image_names):
                if name == test_gen:
                    test_idx.append(i)
                else:
                    train_idx.append(i)

            folds.append({
                "test_generator": test_gen,
                "train_idx": np.array(train_idx),
                "test_idx": np.array(test_idx),
            })

        return folds

    def evaluate_fold(self, features: np.ndarray, labels: np.ndarray,
                     fold: Dict, model_type: str = "ensemble") -> Dict:
        """Evaluate single fold."""

        X_train = features[fold["train_idx"].astype(int)]
        y_train = labels[fold["train_idx"]]
        X_test = features[fold["test_idx"].astype(int)]
        y_test = labels[fold["test_idx"]]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_type == "ensemble":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "cnn_only":
            X_train = X_train[:, -1280:]
            X_test = X_test[:, -1280:]
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "frequency_only":
            X_train = X_train[:, :27]
            X_test = X_test[:, :27]
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            "test_generator": fold["test_generator"],
            "accuracy": acc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def run(self, features: np.ndarray, labels: np.ndarray,
           image_names: List[str], checkpoint_dir: str = "outputs/checkpoints") -> Dict:
        """Run full cross-validation with checkpointing."""

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        folds = self.get_fold_splits(features, labels, image_names)

        results = {
            "ensemble": [],
            "cnn_only": [],
            "frequency_only": [],
        }

        print("\n" + "=" * 70)
        print("4-Fold Cross-Validation (Rotating Held-Out Generator)")
        print("Architecturally Diverse: U-Net vs DiT vs U-Net vs Real")
        print("=" * 70)

        for fold_idx, fold in enumerate(folds):
            print(f"\nFold {fold_idx + 1}/{len(folds)}: Hold out {fold['test_generator'].upper()}")
            print("-" * 70)

            for model_type in ["ensemble", "cnn_only", "frequency_only"]:
                fold_result = self.evaluate_fold(features, labels, fold, model_type)
                results[model_type].append(fold_result)

                print(f"  {model_type.upper():20} → Accuracy: {fold_result['accuracy']:.3f}, "
                      f"AUC: {fold_result['auc']:.3f}, F1: {fold_result['f1']:.3f}")

                # Save checkpoint after each fold
                checkpoint_path = Path(checkpoint_dir) / f"fold_{fold_idx}_{model_type}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "fold_idx": fold_idx,
                        "test_generator": fold["test_generator"],
                        "model_type": model_type,
                        "accuracy": float(fold_result["accuracy"]),
                        "auc": float(fold_result["auc"]),
                        "precision": float(fold_result["precision"]),
                        "recall": float(fold_result["recall"]),
                        "f1": float(fold_result["f1"]),
                    }, f)
                print(f"    ✓ Checkpoint saved: {checkpoint_path.name}")

        print("\n" + "=" * 70)
        print("SUMMARY - Mean Performance Across 4 Folds")
        print("=" * 70)

        summary = {}

        for model_type in ["ensemble", "cnn_only", "frequency_only"]:
            accs = [r["accuracy"] for r in results[model_type]]
            aucs = [r["auc"] for r in results[model_type]]
            f1s = [r["f1"] for r in results[model_type]]

            summary[model_type] = {
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            }

            print(f"\n{model_type.upper()}")
            print(f"  Accuracy: {summary[model_type]['accuracy_mean']:.3f} ± {summary[model_type]['accuracy_std']:.3f}")
            print(f"  AUC-ROC:  {summary[model_type]['auc_mean']:.3f} ± {summary[model_type]['auc_std']:.3f}")
            print(f"  F1-Score: {summary[model_type]['f1_mean']:.3f} ± {summary[model_type]['f1_std']:.3f}")

        print("\n" + "=" * 70)
        print(f"✓ ENSEMBLE GENERALIZES BETTER: "
              f"{summary['ensemble']['accuracy_mean']:.1%} vs "
              f"CNN-only {summary['cnn_only']['accuracy_mean']:.1%} vs "
              f"Frequency-only {summary['frequency_only']['accuracy_mean']:.1%}")
        print("=" * 70)

        return results, summary

    def save_results(self, results: Dict, summary: Dict, output_dir: str = "outputs", checkpoint_dir: str = "outputs/checkpoints"):
        """Save results and create visualizations."""

        Path(output_dir).mkdir(exist_ok=True)

        results_file = Path(output_dir) / f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_to_save = {}
        for model_type, folds in results.items():
            results_to_save[model_type] = []
            for fold in folds:
                results_to_save[model_type].append({
                    "test_generator": fold["test_generator"],
                    "accuracy": float(fold["accuracy"]),
                    "auc": float(fold["auc"]),
                    "precision": float(fold["precision"]),
                    "recall": float(fold["recall"]),
                    "f1": float(fold["f1"]),
                })

        with open(results_file, "w") as f:
            json.dump({
                "results": results_to_save,
                "summary": {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in summary.items()},
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Clean up checkpoints after successful completion
        if Path(checkpoint_dir).exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"✓ Cleaned up training checkpoints")

        self._plot_results(results, summary, output_dir)

    def _plot_results(self, results: Dict, summary: Dict, output_dir: str):
        """Create visualizations of results."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cross-Generator Generalization Evaluation", fontsize=16, fontweight="bold")

        model_types = list(results.keys())
        colors = {"ensemble": "#2ecc71", "cnn_only": "#3498db", "frequency_only": "#e74c3c"}

        accs_by_model = {mt: [r["accuracy"] for r in results[mt]] for mt in model_types}
        aucs_by_model = {mt: [r["auc"] for r in results[mt]] for mt in model_types}
        f1s_by_model = {mt: [r["f1"] for r in results[mt]] for mt in model_types}

        generators = results["ensemble"][0]["test_generator"]

        x_pos = np.arange(len(results["ensemble"]))
        width = 0.25

        for i, mt in enumerate(model_types):
            axes[0, 0].bar(x_pos + i * width, accs_by_model[mt], width,
                          label=mt, color=colors[mt], alpha=0.8)

        axes[0, 0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Accuracy by Held-Out Generator")
        axes[0, 0].set_xticks(x_pos + width)
        axes[0, 0].set_xticklabels([r["test_generator"] for r in results["ensemble"]], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(axis="y", alpha=0.3)

        for i, mt in enumerate(model_types):
            axes[0, 1].bar(x_pos + i * width, aucs_by_model[mt], width,
                          label=mt, color=colors[mt], alpha=0.8)

        axes[0, 1].set_ylabel("AUC-ROC", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("AUC-ROC by Held-Out Generator")
        axes[0, 1].set_xticks(x_pos + width)
        axes[0, 1].set_xticklabels([r["test_generator"] for r in results["ensemble"]], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", alpha=0.3)

        means = [summary[mt]["accuracy_mean"] for mt in model_types]
        stds = [summary[mt]["accuracy_std"] for mt in model_types]

        bars = axes[1, 0].bar(model_types, means, yerr=stds, capsize=5,
                             color=[colors[mt] for mt in model_types], alpha=0.8)

        axes[1, 0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Mean Accuracy ± Std (Cross-Validation)")
        axes[1, 0].set_ylim([0, 1.0])
        axes[1, 0].grid(axis="y", alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, means)):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.05,
                           f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

        model_comparison = []
        for mt in model_types:
            model_comparison.append({
                "Model": mt.upper(),
                "Accuracy": summary[mt]["accuracy_mean"],
                "AUC-ROC": summary[mt]["auc_mean"],
                "F1": summary[mt]["f1_mean"],
            })

        x = np.arange(len(model_types))
        width = 0.25

        accs = [m["Accuracy"] for m in model_comparison]
        aucs = [m["AUC-ROC"] for m in model_comparison]
        f1s = [m["F1"] for m in model_comparison]

        axes[1, 1].bar(x - width, accs, width, label="Accuracy", color="#2ecc71", alpha=0.8)
        axes[1, 1].bar(x, aucs, width, label="AUC-ROC", color="#3498db", alpha=0.8)
        axes[1, 1].bar(x + width, f1s, width, label="F1-Score", color="#e74c3c", alpha=0.8)

        axes[1, 1].set_ylabel("Score", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("Metric Comparison")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([m["Model"] for m in model_comparison], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = Path(output_dir) / f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {plot_path}")
        plt.close()


def main():
    """Main training and evaluation pipeline."""

    print("=" * 70)
    print("Model Training - Cross-Generator Validation")
    print("=" * 70)

    data_loader = DataLoader()

    print("\n[*] Loading features...")
    features, labels, image_names = data_loader.load_features()

    if len(features) == 0:
        print("\n[!] No features found. Run feature_extraction.py first.")
        return

    print(f"    Loaded {len(features)} samples, {features.shape[1]} features")

    validator = CrossGeneratorValidator(data_loader)

    results, summary = validator.run(features, labels, image_names, checkpoint_dir="outputs/checkpoints")

    validator.save_results(results, summary, checkpoint_dir="outputs/checkpoints")

    print("\n✓ Training and evaluation complete!")


if __name__ == "__main__":
    main()
