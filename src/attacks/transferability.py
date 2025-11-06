import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple, Optional
from .fgsm import FGSM
from .pgd import PGD
from .utils import calculate_perturbation_metrics, evaluate_attack_success


class TransferabilityAnalyzer:
    """
    Analyzer for evaluating adversarial transferability across different models.

    This class generates adversarial examples on source models and evaluates
    their effectiveness on target models to understand transferability patterns.
    """

    def __init__(self, models: Dict[str, nn.Module], device: str = 'cpu'):
        """
        Initialize transferability analyzer.

        Args:
            models: Dictionary mapping model names to model instances
            device: Device to run computations on
        """
        self.models = models
        self.device = device

        # Move all models to device and set to eval mode
        for model in self.models.values():
            model.to(device)
            model.eval()

        # Initialize attack objects for each model
        self.fgsm_attacks = {name: FGSM(model, device) for name, model in models.items()}
        self.pgd_attacks = {name: PGD(model, device) for name, model in models.items()}

    def analyze_transferability(self,
                               inputs: torch.Tensor,
                               labels: torch.Tensor,
                               attack_params: Dict[str, Dict],
                               target_class: Optional[int] = None) -> Dict[str, Dict]:
        """
        Perform comprehensive transferability analysis.

        Args:
            inputs: Input images tensor
            labels: True labels
            attack_params: Dictionary containing attack parameters for each attack type
            target_class: Target class for targeted attacks

        Returns:
            Dictionary containing transferability results
        """
        results = {}

        model_names = list(self.models.keys())

        # For each source model
        for source_model in model_names:
            results[source_model] = {}

            print(f"Generating attacks with source model: {source_model}")

            # Generate adversarial examples using this source model
            source_adversarials = self._generate_adversarial_examples(
                source_model, inputs, labels, attack_params, target_class
            )

            # Test these adversarial examples on all target models
            for target_model in model_names:
                print(f"  Testing on target model: {target_model}")

                target_results = self._evaluate_on_target_model(
                    target_model, inputs, labels, source_adversarials, target_class
                )

                results[source_model][target_model] = target_results

        return results

    def _generate_adversarial_examples(self,
                                      source_model: str,
                                      inputs: torch.Tensor,
                                      labels: torch.Tensor,
                                      attack_params: Dict[str, Dict],
                                      target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial examples using the source model.

        Args:
            source_model: Name of source model
            inputs: Input images
            labels: True labels
            attack_params: Attack parameters
            target_class: Target class for targeted attacks

        Returns:
            Dictionary mapping attack types to adversarial examples
        """
        adversarial_examples = {}

        # FGSM attacks
        if 'fgsm' in attack_params:
            fgsm_params = attack_params['fgsm']

            # Untargeted FGSM
            adv_inputs, _ = self.fgsm_attacks[source_model].attack(
                inputs, labels,
                epsilon=fgsm_params.get('epsilon', 0.03),
                targeted=False
            )
            adversarial_examples['fgsm_untargeted'] = adv_inputs

            # Targeted FGSM (if target_class provided)
            if target_class is not None:
                target_labels = torch.full_like(labels, target_class)
                # Filter out samples already in target class
                mask = labels != target_class
                if mask.sum() > 0:
                    filtered_inputs = inputs[mask]
                    filtered_targets = target_labels[mask]

                    adv_inputs, _ = self.fgsm_attacks[source_model].attack(
                        filtered_inputs, labels[mask],
                        epsilon=fgsm_params.get('epsilon', 0.03),
                        targeted=True,
                        target_labels=filtered_targets
                    )

                    # Reconstruct full tensor
                    full_adv = inputs.clone()
                    full_adv[mask] = adv_inputs
                    adversarial_examples['fgsm_targeted'] = full_adv

        # PGD attacks
        if 'pgd' in attack_params:
            pgd_params = attack_params['pgd']

            # Untargeted PGD
            adv_inputs, _ = self.pgd_attacks[source_model].attack(
                inputs, labels,
                epsilon=pgd_params.get('epsilon', 0.03),
                alpha=pgd_params.get('alpha', 0.01),
                steps=pgd_params.get('steps', 40),
                targeted=False
            )
            adversarial_examples['pgd_untargeted'] = adv_inputs

            # Targeted PGD (if target_class provided)
            if target_class is not None:
                target_labels = torch.full_like(labels, target_class)
                # Filter out samples already in target class
                mask = labels != target_class
                if mask.sum() > 0:
                    filtered_inputs = inputs[mask]
                    filtered_targets = target_labels[mask]

                    adv_inputs, _ = self.pgd_attacks[source_model].attack(
                        filtered_inputs, labels[mask],
                        epsilon=pgd_params.get('epsilon', 0.03),
                        alpha=pgd_params.get('alpha', 0.01),
                        steps=pgd_params.get('steps', 40),
                        targeted=True,
                        target_labels=filtered_targets
                    )

                    # Reconstruct full tensor
                    full_adv = inputs.clone()
                    full_adv[mask] = adv_inputs
                    adversarial_examples['pgd_targeted'] = full_adv

        return adversarial_examples

    def _evaluate_on_target_model(self,
                                 target_model: str,
                                 original_inputs: torch.Tensor,
                                 true_labels: torch.Tensor,
                                 adversarial_examples: Dict[str, torch.Tensor],
                                 target_class: Optional[int] = None) -> Dict[str, Dict]:
        """
        Evaluate adversarial examples on target model.

        Args:
            target_model: Name of target model
            original_inputs: Original input images
            true_labels: True labels
            adversarial_examples: Dictionary of adversarial examples
            target_class: Target class for targeted attacks

        Returns:
            Dictionary containing evaluation results for each attack type
        """
        results = {}
        model = self.models[target_model]

        with torch.no_grad():
            # Get predictions on original inputs
            original_outputs = model(original_inputs)
            original_predictions = original_outputs.argmax(dim=1)

            # Evaluate each attack type
            for attack_type, adversarial_inputs in adversarial_examples.items():
                # Get predictions on adversarial inputs
                adversarial_outputs = model(adversarial_inputs)
                adversarial_predictions = adversarial_outputs.argmax(dim=1)

                # Determine if this is a targeted attack
                is_targeted = 'targeted' in attack_type
                target_labels = None
                if is_targeted and target_class is not None:
                    target_labels = torch.full_like(true_labels, target_class)

                # Calculate success metrics
                success_metrics = evaluate_attack_success(
                    original_predictions,
                    adversarial_predictions,
                    true_labels,
                    target_labels,
                    is_targeted
                )

                # Calculate perturbation metrics
                perturbation_metrics = calculate_perturbation_metrics(
                    original_inputs, adversarial_inputs
                )

                # Combine results
                attack_results = {
                    **success_metrics,
                    **perturbation_metrics,
                    'attack_type': attack_type,
                    'target_model': target_model,
                    'is_targeted': is_targeted
                }

                if is_targeted:
                    attack_results['target_class'] = target_class

                results[attack_type] = attack_results

        return results

    def create_transferability_matrix(self,
                                     results: Dict[str, Dict],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create transferability matrix visualization.

        Args:
            results: Results from analyze_transferability
            save_path: Path to save the visualization

        Returns:
            Matplotlib figure
        """
        model_names = list(self.models.keys())
        attack_types = set()

        # Collect all attack types
        for source_results in results.values():
            for target_results in source_results.values():
                attack_types.update(target_results.keys())

        attack_types = sorted(list(attack_types))
        n_attacks = len(attack_types)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, attack_type in enumerate(attack_types):
            if i >= 4:  # Limit to 4 subplots
                break

            # Create matrix for this attack type
            matrix = np.zeros((len(model_names), len(model_names)))

            for j, source_model in enumerate(model_names):
                for k, target_model in enumerate(model_names):
                    if (source_model in results and
                        target_model in results[source_model] and
                        attack_type in results[source_model][target_model]):
                        success_rate = results[source_model][target_model][attack_type]['attack_success_rate']
                        matrix[j, k] = success_rate * 100  # Convert to percentage

            # Create heatmap
            sns.heatmap(matrix,
                       annot=True,
                       fmt='.1f',
                       cmap='Reds',
                       xticklabels=model_names,
                       yticklabels=model_names,
                       ax=axes[i],
                       vmin=0,
                       vmax=100,
                       cbar_kws={'label': 'Success Rate (%)'})

            axes[i].set_title(f'{attack_type.upper()} Transferability')
            axes[i].set_xlabel('Target Model')
            axes[i].set_ylabel('Source Model')

        # Hide unused subplots
        for i in range(len(attack_types), 4):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def calculate_transferability_metrics(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate aggregate transferability metrics.

        Args:
            results: Results from analyze_transferability

        Returns:
            Dictionary containing transferability metrics
        """
        metrics = {}
        model_names = list(self.models.keys())

        # Collect success rates for different scenarios
        same_model_rates = []  # Source and target are the same model
        cross_model_rates = []  # Source and target are different models
        all_rates = []

        attack_type_rates = {}  # Rates by attack type

        for source_model in model_names:
            for target_model in model_names:
                if (source_model in results and
                    target_model in results[source_model]):

                    target_results = results[source_model][target_model]

                    for attack_type, attack_results in target_results.items():
                        success_rate = attack_results['attack_success_rate']
                        all_rates.append(success_rate)

                        # Track by attack type
                        if attack_type not in attack_type_rates:
                            attack_type_rates[attack_type] = []
                        attack_type_rates[attack_type].append(success_rate)

                        # Same vs cross model
                        if source_model == target_model:
                            same_model_rates.append(success_rate)
                        else:
                            cross_model_rates.append(success_rate)

        # Calculate metrics
        if all_rates:
            metrics['overall_mean_success_rate'] = np.mean(all_rates)
            metrics['overall_std_success_rate'] = np.std(all_rates)

        if same_model_rates:
            metrics['same_model_mean_success_rate'] = np.mean(same_model_rates)

        if cross_model_rates:
            metrics['cross_model_mean_success_rate'] = np.mean(cross_model_rates)
            metrics['transferability_ratio'] = np.mean(cross_model_rates) / np.mean(same_model_rates) if same_model_rates else 0.0

        # Per attack type metrics
        for attack_type, rates in attack_type_rates.items():
            metrics[f'{attack_type}_mean_success_rate'] = np.mean(rates)
            metrics[f'{attack_type}_std_success_rate'] = np.std(rates)

        return metrics

    def save_results(self, results: Dict[str, Dict], save_dir: str) -> None:
        """
        Save transferability analysis results.

        Args:
            results: Results from analyze_transferability
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save raw results
        results_path = os.path.join(save_dir, 'transferability_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

        # Calculate and save metrics
        metrics = self.calculate_transferability_metrics(results)
        metrics_path = os.path.join(save_dir, 'transferability_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Create and save visualization
        fig = self.create_transferability_matrix(results,
                                                os.path.join(save_dir, 'transferability_matrix.png'))
        plt.close(fig)

        print(f"Transferability results saved to {save_dir}")

    def compare_attack_transferability(self,
                                      inputs: torch.Tensor,
                                      labels: torch.Tensor,
                                      attack_configs: List[Dict]) -> Dict[str, Dict]:
        """
        Compare transferability across different attack configurations.

        Args:
            inputs: Input images
            labels: True labels
            attack_configs: List of attack configuration dictionaries

        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}

        for i, config in enumerate(attack_configs):
            config_name = config.get('name', f'config_{i}')
            attack_params = config['params']
            target_class = config.get('target_class', None)

            print(f"Running transferability analysis for {config_name}")

            results = self.analyze_transferability(inputs, labels, attack_params, target_class)
            metrics = self.calculate_transferability_metrics(results)

            comparison_results[config_name] = {
                'config': config,
                'results': results,
                'metrics': metrics
            }

        return comparison_results


def test_transferability():
    """Test transferability analyzer with dummy models."""
    import torch.nn as nn
    import torch.nn.functional as F

    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
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

    # Create models
    models = {
        'model_A': DummyModel('A'),
        'model_B': DummyModel('B')
    }

    # Test data
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))

    # Attack parameters
    attack_params = {
        'fgsm': {'epsilon': 0.03},
        'pgd': {'epsilon': 0.03, 'alpha': 0.01, 'steps': 10}
    }

    # Create analyzer
    analyzer = TransferabilityAnalyzer(models, device='cpu')

    print("Testing transferability analysis...")

    # Run analysis
    results = analyzer.analyze_transferability(inputs, labels, attack_params, target_class=5)

    # Calculate metrics
    metrics = analyzer.calculate_transferability_metrics(results)

    print("Transferability metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    print("Transferability analysis test completed!")


if __name__ == "__main__":
    test_transferability()