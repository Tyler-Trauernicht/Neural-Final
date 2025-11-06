import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
from ..interpretability.saliency import SaliencyMap
from ..interpretability.gradcam import GradCAM, get_target_layer
from .utils import denormalize_image


class AdversarialInterpretabilityAnalyzer:
    """
    Comprehensive interpretability analysis for adversarial examples.

    This class provides tools to analyze and visualize how adversarial perturbations
    affect model interpretability using saliency maps, Grad-CAM, and other techniques.
    """

    def __init__(self, model: nn.Module, model_name: str, device: str = 'cpu'):
        """
        Initialize interpretability analyzer.

        Args:
            model: Trained neural network model
            model_name: Name of the model (for selecting appropriate layers)
            device: Device to run computations on
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Initialize saliency analyzer
        self.saliency_analyzer = SaliencyMap(model, device)

        # Initialize Grad-CAM analyzer
        try:
            target_layer = get_target_layer(model, model_name)
            self.gradcam_analyzer = GradCAM(model, target_layer, device)
            self.gradcam_available = True
        except Exception as e:
            print(f"Warning: Grad-CAM not available for {model_name}: {e}")
            self.gradcam_analyzer = None
            self.gradcam_available = False

    def compare_clean_vs_adversarial(self,
                                   clean_inputs: torch.Tensor,
                                   adversarial_inputs: torch.Tensor,
                                   true_labels: torch.Tensor,
                                   class_names: List[str],
                                   save_dir: Optional[str] = None) -> Dict[str, any]:
        """
        Compare interpretability between clean and adversarial examples.

        Args:
            clean_inputs: Clean input images
            adversarial_inputs: Adversarial input images
            true_labels: True labels for the inputs
            class_names: List of class names
            save_dir: Directory to save visualizations

        Returns:
            Dictionary containing comparison results
        """
        results = {
            'clean_saliency': [],
            'adversarial_saliency': [],
            'saliency_similarity': [],
            'prediction_changes': [],
            'confidence_changes': []
        }

        if self.gradcam_available:
            results['clean_gradcam'] = []
            results['adversarial_gradcam'] = []
            results['gradcam_similarity'] = []

        batch_size = clean_inputs.size(0)

        # Get predictions for clean and adversarial inputs
        with torch.no_grad():
            clean_outputs = self.model(clean_inputs)
            adv_outputs = self.model(adversarial_inputs)

            clean_probs = F.softmax(clean_outputs, dim=1)
            adv_probs = F.softmax(adv_outputs, dim=1)

            clean_preds = clean_outputs.argmax(dim=1)
            adv_preds = adv_outputs.argmax(dim=1)

            clean_confidence = clean_probs.max(dim=1)[0]
            adv_confidence = adv_probs.max(dim=1)[0]

        # Analyze each sample
        for i in range(batch_size):
            sample_clean = clean_inputs[i:i+1]
            sample_adv = adversarial_inputs[i:i+1]
            true_label = true_labels[i].item()

            # Saliency analysis
            clean_saliency, clean_pred, clean_conf = self.saliency_analyzer.generate(sample_clean)
            adv_saliency, adv_pred, adv_conf = self.saliency_analyzer.generate(sample_adv)

            results['clean_saliency'].append(clean_saliency)
            results['adversarial_saliency'].append(adv_saliency)

            # Calculate saliency similarity
            saliency_similarity = self._calculate_similarity(clean_saliency, adv_saliency)
            results['saliency_similarity'].append(saliency_similarity)

            # Prediction and confidence changes
            results['prediction_changes'].append({
                'clean_pred': clean_pred,
                'adv_pred': adv_pred,
                'true_label': true_label,
                'prediction_changed': clean_pred != adv_pred
            })

            results['confidence_changes'].append({
                'clean_confidence': clean_conf,
                'adv_confidence': adv_conf,
                'confidence_drop': clean_conf - adv_conf
            })

            # Grad-CAM analysis (if available)
            if self.gradcam_available:
                clean_gradcam, _, _ = self.gradcam_analyzer.generate(sample_clean)
                adv_gradcam, _, _ = self.gradcam_analyzer.generate(sample_adv)

                results['clean_gradcam'].append(clean_gradcam)
                results['adversarial_gradcam'].append(adv_gradcam)

                gradcam_similarity = self._calculate_similarity(clean_gradcam, adv_gradcam)
                results['gradcam_similarity'].append(gradcam_similarity)

            # Create visualizations
            if save_dir:
                self._create_comparison_visualization(
                    i, sample_clean, sample_adv, clean_saliency, adv_saliency,
                    clean_gradcam if self.gradcam_available else None,
                    adv_gradcam if self.gradcam_available else None,
                    clean_pred, adv_pred, true_label, class_names, save_dir
                )

        return results

    def analyze_attention_shift(self,
                              clean_inputs: torch.Tensor,
                              adversarial_inputs: torch.Tensor,
                              class_names: List[str]) -> Dict[str, float]:
        """
        Analyze how adversarial perturbations shift model attention.

        Args:
            clean_inputs: Clean input images
            adversarial_inputs: Adversarial input images
            class_names: List of class names

        Returns:
            Dictionary containing attention shift metrics
        """
        metrics = {
            'mean_saliency_correlation': 0.0,
            'attention_shift_magnitude': 0.0,
            'focus_region_overlap': 0.0
        }

        if self.gradcam_available:
            metrics['mean_gradcam_correlation'] = 0.0

        correlations = []
        shift_magnitudes = []
        focus_overlaps = []

        if self.gradcam_available:
            gradcam_correlations = []

        for i in range(clean_inputs.size(0)):
            clean_sample = clean_inputs[i:i+1]
            adv_sample = adversarial_inputs[i:i+1]

            # Saliency analysis
            clean_saliency, _, _ = self.saliency_analyzer.generate(clean_sample)
            adv_saliency, _, _ = self.saliency_analyzer.generate(adv_sample)

            # Calculate correlation
            correlation = np.corrcoef(clean_saliency.flatten(), adv_saliency.flatten())[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)

            # Calculate attention shift magnitude
            diff = np.abs(clean_saliency - adv_saliency)
            shift_magnitude = np.mean(diff)
            shift_magnitudes.append(shift_magnitude)

            # Calculate focus region overlap
            # Define focus regions as top 20% of saliency values
            clean_threshold = np.percentile(clean_saliency, 80)
            adv_threshold = np.percentile(adv_saliency, 80)

            clean_focus = clean_saliency >= clean_threshold
            adv_focus = adv_saliency >= adv_threshold

            intersection = np.logical_and(clean_focus, adv_focus).sum()
            union = np.logical_or(clean_focus, adv_focus).sum()

            if union > 0:
                overlap = intersection / union
                focus_overlaps.append(overlap)

            # Grad-CAM analysis (if available)
            if self.gradcam_available:
                clean_gradcam, _, _ = self.gradcam_analyzer.generate(clean_sample)
                adv_gradcam, _, _ = self.gradcam_analyzer.generate(adv_sample)

                gradcam_correlation = np.corrcoef(clean_gradcam.flatten(), adv_gradcam.flatten())[0, 1]
                if not np.isnan(gradcam_correlation):
                    gradcam_correlations.append(gradcam_correlation)

        # Calculate aggregate metrics
        if correlations:
            metrics['mean_saliency_correlation'] = np.mean(correlations)

        if shift_magnitudes:
            metrics['attention_shift_magnitude'] = np.mean(shift_magnitudes)

        if focus_overlaps:
            metrics['focus_region_overlap'] = np.mean(focus_overlaps)

        if self.gradcam_available and gradcam_correlations:
            metrics['mean_gradcam_correlation'] = np.mean(gradcam_correlations)

        return metrics

    def analyze_robustness_vs_interpretability(self,
                                             inputs: torch.Tensor,
                                             perturbations: torch.Tensor,
                                             true_labels: torch.Tensor) -> Dict[str, any]:
        """
        Analyze relationship between attack success and interpretability changes.

        Args:
            inputs: Original input images
            perturbations: Perturbations added to create adversarial examples
            true_labels: True labels

        Returns:
            Dictionary containing robustness-interpretability analysis
        """
        adversarial_inputs = inputs + perturbations

        # Get predictions
        with torch.no_grad():
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(adversarial_inputs)

            clean_preds = clean_outputs.argmax(dim=1)
            adv_preds = adv_outputs.argmax(dim=1)

            # Determine which attacks were successful
            successful_attacks = (clean_preds != adv_preds)

        # Analyze interpretability for successful vs failed attacks
        successful_indices = torch.where(successful_attacks)[0]
        failed_indices = torch.where(~successful_attacks)[0]

        results = {
            'successful_attacks': {
                'count': len(successful_indices),
                'saliency_changes': [],
                'attention_metrics': {}
            },
            'failed_attacks': {
                'count': len(failed_indices),
                'saliency_changes': [],
                'attention_metrics': {}
            }
        }

        # Analyze successful attacks
        if len(successful_indices) > 0:
            successful_clean = inputs[successful_indices]
            successful_adv = adversarial_inputs[successful_indices]
            successful_labels = true_labels[successful_indices]

            successful_metrics = self.analyze_attention_shift(
                successful_clean, successful_adv, []
            )
            results['successful_attacks']['attention_metrics'] = successful_metrics

        # Analyze failed attacks
        if len(failed_indices) > 0:
            failed_clean = inputs[failed_indices]
            failed_adv = adversarial_inputs[failed_indices]
            failed_labels = true_labels[failed_indices]

            failed_metrics = self.analyze_attention_shift(
                failed_clean, failed_adv, []
            )
            results['failed_attacks']['attention_metrics'] = failed_metrics

        return results

    def _calculate_similarity(self, map1: np.ndarray, map2: np.ndarray) -> float:
        """
        Calculate similarity between two attention maps.

        Args:
            map1: First attention map
            map2: Second attention map

        Returns:
            Similarity score (correlation coefficient)
        """
        correlation = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _create_comparison_visualization(self,
                                       sample_idx: int,
                                       clean_input: torch.Tensor,
                                       adv_input: torch.Tensor,
                                       clean_saliency: np.ndarray,
                                       adv_saliency: np.ndarray,
                                       clean_gradcam: Optional[np.ndarray],
                                       adv_gradcam: Optional[np.ndarray],
                                       clean_pred: int,
                                       adv_pred: int,
                                       true_label: int,
                                       class_names: List[str],
                                       save_dir: str) -> None:
        """
        Create comprehensive comparison visualization.
        """
        # Determine number of rows based on available methods
        n_rows = 3 if self.gradcam_available else 2
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))

        # Denormalize images for visualization
        clean_denorm = denormalize_image(clean_input).squeeze().permute(1, 2, 0).cpu().numpy()
        adv_denorm = denormalize_image(adv_input).squeeze().permute(1, 2, 0).cpu().numpy()
        perturbation = adv_denorm - clean_denorm

        # Row 1: Images
        axes[0, 0].imshow(clean_denorm)
        axes[0, 0].set_title(f'Clean\nTrue: {class_names[true_label]}\nPred: {class_names[clean_pred]}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(adv_denorm)
        axes[0, 1].set_title(f'Adversarial\nPred: {class_names[adv_pred]}')
        axes[0, 1].axis('off')

        # Visualize perturbation
        pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        axes[0, 2].imshow(pert_vis)
        axes[0, 2].set_title('Perturbation')
        axes[0, 2].axis('off')

        # Perturbation magnitude
        pert_mag = np.linalg.norm(perturbation, axis=2)
        im = axes[0, 3].imshow(pert_mag, cmap='hot')
        axes[0, 3].set_title('Perturbation Magnitude')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

        axes[0, 4].axis('off')  # Empty

        # Row 2: Saliency maps
        axes[1, 0].imshow(clean_saliency, cmap='hot')
        axes[1, 0].set_title('Clean Saliency')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(adv_saliency, cmap='hot')
        axes[1, 1].set_title('Adversarial Saliency')
        axes[1, 1].axis('off')

        # Saliency difference
        saliency_diff = np.abs(clean_saliency - adv_saliency)
        im = axes[1, 2].imshow(saliency_diff, cmap='hot')
        axes[1, 2].set_title('Saliency Difference')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Overlay saliency on images
        axes[1, 3].imshow(clean_denorm)
        axes[1, 3].imshow(clean_saliency, cmap='hot', alpha=0.5)
        axes[1, 3].set_title('Clean + Saliency')
        axes[1, 3].axis('off')

        axes[1, 4].imshow(adv_denorm)
        axes[1, 4].imshow(adv_saliency, cmap='hot', alpha=0.5)
        axes[1, 4].set_title('Adversarial + Saliency')
        axes[1, 4].axis('off')

        # Row 3: Grad-CAM (if available)
        if self.gradcam_available and clean_gradcam is not None:
            axes[2, 0].imshow(clean_gradcam, cmap='jet')
            axes[2, 0].set_title('Clean Grad-CAM')
            axes[2, 0].axis('off')

            axes[2, 1].imshow(adv_gradcam, cmap='jet')
            axes[2, 1].set_title('Adversarial Grad-CAM')
            axes[2, 1].axis('off')

            # Grad-CAM difference
            gradcam_diff = np.abs(clean_gradcam - adv_gradcam)
            im = axes[2, 2].imshow(gradcam_diff, cmap='hot')
            axes[2, 2].set_title('Grad-CAM Difference')
            axes[2, 2].axis('off')
            plt.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)

            # Overlay Grad-CAM on images
            axes[2, 3].imshow(clean_denorm)
            axes[2, 3].imshow(clean_gradcam, cmap='jet', alpha=0.4)
            axes[2, 3].set_title('Clean + Grad-CAM')
            axes[2, 3].axis('off')

            axes[2, 4].imshow(adv_denorm)
            axes[2, 4].imshow(adv_gradcam, cmap='jet', alpha=0.4)
            axes[2, 4].set_title('Adversarial + Grad-CAM')
            axes[2, 4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_sample_{sample_idx:03d}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_interpretability_report(self,
                                       analysis_results: Dict,
                                       save_path: str) -> None:
        """
        Generate comprehensive interpretability analysis report.

        Args:
            analysis_results: Results from interpretability analysis
            save_path: Path to save the report
        """
        report = {
            "interpretability_analysis": {
                "model_name": self.model_name,
                "methods_used": ["saliency_maps"],
                "samples_analyzed": len(analysis_results.get('clean_saliency', [])),
                "gradcam_available": self.gradcam_available
            },
            "attention_shift_metrics": {},
            "prediction_analysis": {},
            "key_findings": []
        }

        if self.gradcam_available:
            report["interpretability_analysis"]["methods_used"].append("grad_cam")

        # Calculate aggregate metrics
        if 'saliency_similarity' in analysis_results:
            similarities = [s for s in analysis_results['saliency_similarity'] if not np.isnan(s)]
            if similarities:
                report["attention_shift_metrics"]["mean_saliency_correlation"] = np.mean(similarities)
                report["attention_shift_metrics"]["std_saliency_correlation"] = np.std(similarities)

        if 'gradcam_similarity' in analysis_results:
            similarities = [s for s in analysis_results['gradcam_similarity'] if not np.isnan(s)]
            if similarities:
                report["attention_shift_metrics"]["mean_gradcam_correlation"] = np.mean(similarities)

        # Prediction analysis
        if 'prediction_changes' in analysis_results:
            changes = analysis_results['prediction_changes']
            total_samples = len(changes)
            changed_predictions = sum(1 for c in changes if c['prediction_changed'])

            report["prediction_analysis"]["total_samples"] = total_samples
            report["prediction_analysis"]["predictions_changed"] = changed_predictions
            report["prediction_analysis"]["prediction_change_rate"] = changed_predictions / total_samples if total_samples > 0 else 0

        # Confidence analysis
        if 'confidence_changes' in analysis_results:
            conf_changes = [c['confidence_drop'] for c in analysis_results['confidence_changes']]
            if conf_changes:
                report["prediction_analysis"]["mean_confidence_drop"] = np.mean(conf_changes)
                report["prediction_analysis"]["std_confidence_drop"] = np.std(conf_changes)

        # Key findings
        findings = []

        # Attention preservation
        mean_sal_corr = report["attention_shift_metrics"].get("mean_saliency_correlation", 0)
        if mean_sal_corr > 0.7:
            findings.append(f"High attention preservation: {mean_sal_corr:.3f} saliency correlation")
        elif mean_sal_corr < 0.3:
            findings.append(f"Significant attention disruption: {mean_sal_corr:.3f} saliency correlation")

        # Prediction changes
        change_rate = report["prediction_analysis"].get("prediction_change_rate", 0)
        if change_rate > 0.8:
            findings.append(f"High attack success rate: {change_rate:.1%} predictions changed")

        report["key_findings"] = findings

        # Save report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def test_interpretability_analyzer():
    """Test the interpretability analyzer."""
    import torch.nn as nn

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(64 * 4 * 4, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Test setup
    model = TestModel()
    analyzer = AdversarialInterpretabilityAnalyzer(model, "TestModel", device='cpu')

    # Test data
    clean_inputs = torch.randn(2, 3, 32, 32)
    adversarial_inputs = clean_inputs + 0.1 * torch.randn_like(clean_inputs)
    true_labels = torch.tensor([0, 1])
    class_names = [f"class_{i}" for i in range(10)]

    print("Testing interpretability analyzer...")

    # Run analysis
    results = analyzer.compare_clean_vs_adversarial(
        clean_inputs, adversarial_inputs, true_labels, class_names
    )

    print(f"Analyzed {len(results['clean_saliency'])} samples")
    print(f"Mean saliency correlation: {np.mean(results['saliency_similarity']):.3f}")

    print("Interpretability analyzer test completed!")


if __name__ == "__main__":
    test_interpretability_analyzer()