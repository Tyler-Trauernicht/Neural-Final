"""
Adversarial Robustness Analysis for Pruned Models

This module implements adversarial attack methods and robustness evaluation
for analyzing how pruning affects model vulnerability to adversarial examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) adversarial attack."""

    def __init__(self, epsilon: float = 0.03):
        """
        Args:
            epsilon: Perturbation magnitude
        """
        self.epsilon = epsilon

    def generate(self, model: nn.Module, data: torch.Tensor,
                target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate FGSM adversarial examples.

        Args:
            model: Target model
            data: Input data
            target: True labels
            device: Device to run on

        Returns:
            Adversarial examples
        """
        model.eval()
        data = data.to(device)
        target = target.to(device)
        data.requires_grad = True

        output = model(data)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        data_grad = data.grad.data
        perturbed_data = data + self.epsilon * data_grad.sign()

        # Clamp to valid image range [0, 1] after normalization
        perturbed_data = torch.clamp(perturbed_data, -2.5, 2.5)  # Approximate range after normalization

        return perturbed_data.detach()


class PGDAttack:
    """Projected Gradient Descent (PGD) adversarial attack."""

    def __init__(self, epsilon: float = 0.03, alpha: float = 0.01, num_iter: int = 10):
        """
        Args:
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def generate(self, model: nn.Module, data: torch.Tensor,
                target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate PGD adversarial examples.

        Args:
            model: Target model
            data: Input data
            target: True labels
            device: Device to run on

        Returns:
            Adversarial examples
        """
        model.eval()
        data = data.to(device)
        target = target.to(device)

        # Start with random perturbation
        perturbed_data = data + torch.empty_like(data).uniform_(-self.epsilon, self.epsilon)
        perturbed_data = torch.clamp(perturbed_data, -2.5, 2.5)

        for _ in range(self.num_iter):
            perturbed_data.requires_grad = True

            output = model(perturbed_data)
            loss = F.cross_entropy(output, target)

            model.zero_grad()
            loss.backward()

            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + self.alpha * data_grad.sign()

            # Project back to epsilon ball
            eta = torch.clamp(perturbed_data - data, -self.epsilon, self.epsilon)
            perturbed_data = torch.clamp(data + eta, -2.5, 2.5).detach()

        return perturbed_data


class CWAttack:
    """Carlini & Wagner (C&W) adversarial attack (simplified version)."""

    def __init__(self, c: float = 1.0, kappa: float = 0.0, num_iter: int = 10):
        """
        Args:
            c: Confidence parameter
            kappa: Margin parameter
            num_iter: Number of iterations
        """
        self.c = c
        self.kappa = kappa
        self.num_iter = num_iter

    def generate(self, model: nn.Module, data: torch.Tensor,
                target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate simplified C&W adversarial examples.

        Args:
            model: Target model
            data: Input data
            target: True labels
            device: Device to run on

        Returns:
            Adversarial examples
        """
        model.eval()
        data = data.to(device)
        target = target.to(device)

        # Initialize perturbation
        w = torch.zeros_like(data, requires_grad=True)

        optimizer = torch.optim.Adam([w], lr=0.01)

        for _ in range(self.num_iter):
            adv_data = torch.tanh(w) * 0.5 + data

            output = model(adv_data)

            # L2 distance loss
            l2_loss = torch.norm(adv_data - data, p=2, dim=(1, 2, 3))

            # Classification loss (simplified)
            f_loss = F.cross_entropy(output, target, reduction='none')

            # Total loss
            loss = l2_loss + self.c * f_loss
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adv_data = torch.tanh(w) * 0.5 + data
        return adv_data.detach()


def evaluate_adversarial_robustness(model: nn.Module, test_loader: DataLoader,
                                  device: torch.device, attacks: List = None,
                                  max_samples: int = 100) -> Dict[str, float]:
    """
    Evaluate model robustness against adversarial attacks.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        attacks: List of attack methods to use
        max_samples: Maximum number of samples to test

    Returns:
        Dictionary with robustness metrics
    """
    if attacks is None:
        attacks = [
            ('FGSM', FGSMAttack(epsilon=0.03)),
            ('PGD', PGDAttack(epsilon=0.03, alpha=0.01, num_iter=10)),
            ('CW', CWAttack(c=1.0, num_iter=5))  # Reduced iterations for speed
        ]

    model.to(device)
    model.eval()

    results = {}

    # Evaluate clean accuracy first
    clean_correct = 0
    total_samples = 0

    # Track per-attack performance
    attack_results = {name: {'correct': 0, 'total': 0} for name, _ in attacks}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if total_samples >= max_samples:
                break

            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Clean accuracy
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += (clean_pred == target).sum().item()

            # Test each attack
            for attack_name, attack_method in attacks:
                try:
                    # Generate adversarial examples
                    adv_data = attack_method.generate(model, data, target, device)

                    # Evaluate on adversarial examples
                    with torch.no_grad():
                        adv_output = model(adv_data)
                        adv_pred = adv_output.argmax(dim=1)
                        adv_correct = (adv_pred == target).sum().item()

                    attack_results[attack_name]['correct'] += adv_correct
                    attack_results[attack_name]['total'] += batch_size

                except Exception as e:
                    print(f"Error in {attack_name} attack: {e}")
                    attack_results[attack_name]['total'] += batch_size

            total_samples += batch_size

    # Calculate final metrics
    clean_accuracy = clean_correct / total_samples * 100

    results['clean_accuracy'] = clean_accuracy
    results['total_samples'] = total_samples

    for attack_name, attack_data in attack_results.items():
        if attack_data['total'] > 0:
            adv_accuracy = attack_data['correct'] / attack_data['total'] * 100
            attack_success_rate = 100 - adv_accuracy  # Success rate = 100 - accuracy
        else:
            adv_accuracy = 0.0
            attack_success_rate = 100.0

        results[f'{attack_name.lower()}_accuracy'] = adv_accuracy
        results[f'{attack_name.lower()}_success_rate'] = attack_success_rate

    return results


def compare_robustness_across_sparsity(model_configs: Dict, test_loader: DataLoader,
                                     device: torch.device, checkpoints_dir: str = "checkpoints",
                                     max_samples: int = 50) -> Dict[str, Dict]:
    """
    Compare adversarial robustness across different sparsity levels.

    Args:
        model_configs: Dictionary of model configurations
        test_loader: Test data loader
        device: Device to run on
        checkpoints_dir: Directory containing model checkpoints
        max_samples: Maximum samples to test for speed

    Returns:
        Dictionary with robustness comparison results
    """
    print("\n" + "="*60)
    print("ADVERSARIAL ROBUSTNESS ANALYSIS")
    print("="*60)

    robustness_results = {}

    for model_name, config in model_configs.items():
        print(f"\nAnalyzing {model_name}...")
        model_results = {}

        # Test original model
        try:
            checkpoint_path = f"{checkpoints_dir}/{model_name}-original.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = config['class'](**config['args'])
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"  Testing original model...")
            original_robustness = evaluate_adversarial_robustness(
                model, test_loader, device, max_samples=max_samples
            )
            model_results['original'] = original_robustness

        except Exception as e:
            print(f"  Error loading original model: {e}")
            continue

        # Test pruned models
        sparsity_levels = [0.2, 0.5, 0.8]
        for sparsity in sparsity_levels:
            try:
                pruned_path = f"{checkpoints_dir}/{model_name}-pruned-{sparsity:.0%}.pt"
                if not os.path.exists(pruned_path):
                    print(f"  Skipping {sparsity:.0%} pruned model (not found)")
                    continue

                checkpoint = torch.load(pruned_path, map_location='cpu')
                pruned_model = config['class'](**config['args'])
                pruned_model.load_state_dict(checkpoint['model_state_dict'])

                print(f"  Testing {sparsity:.0%} pruned model...")
                pruned_robustness = evaluate_adversarial_robustness(
                    pruned_model, test_loader, device, max_samples=max_samples
                )
                model_results[f'pruned_{sparsity:.0%}'] = pruned_robustness

            except Exception as e:
                print(f"  Error loading {sparsity:.0%} pruned model: {e}")
                continue

        robustness_results[model_name] = model_results

    return robustness_results


def analyze_robustness_trends(robustness_results: Dict) -> Dict[str, Any]:
    """
    Analyze trends in adversarial robustness across sparsity levels.

    Args:
        robustness_results: Results from robustness comparison

    Returns:
        Dictionary with trend analysis
    """
    trends = {}

    for model_name, model_results in robustness_results.items():
        if 'original' not in model_results:
            continue

        model_trends = {}

        # Track how robustness changes with sparsity
        attacks = ['fgsm', 'pgd', 'cw']

        for attack in attacks:
            success_key = f'{attack}_success_rate'
            if success_key in model_results['original']:
                original_success = model_results['original'][success_key]
                sparsity_changes = []

                for sparsity in [0.2, 0.5, 0.8]:
                    key = f'pruned_{sparsity:.0%}'
                    if key in model_results and success_key in model_results[key]:
                        pruned_success = model_results[key][success_key]
                        change = pruned_success - original_success
                        sparsity_changes.append((sparsity, change))

                model_trends[attack] = {
                    'original_success_rate': original_success,
                    'sparsity_changes': sparsity_changes
                }

        trends[model_name] = model_trends

    return trends


# Import os for file operations
import os


def create_robustness_report(robustness_results: Dict, trends: Dict,
                           save_path: str = "results/problem_c/adversarial_robustness_report.txt"):
    """
    Create a comprehensive robustness analysis report.

    Args:
        robustness_results: Robustness evaluation results
        trends: Trend analysis results
        save_path: Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("ADVERSARIAL ROBUSTNESS ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")

        f.write("This report analyzes how model pruning affects adversarial robustness.\n")
        f.write("Lower success rates indicate better robustness.\n\n")

        for model_name, model_results in robustness_results.items():
            f.write(f"{model_name.upper()} ROBUSTNESS ANALYSIS\n")
            f.write("-" * 40 + "\n\n")

            if 'original' not in model_results:
                f.write("No results available for this model.\n\n")
                continue

            # Original model results
            orig = model_results['original']
            f.write(f"Original Model:\n")
            f.write(f"  Clean Accuracy: {orig['clean_accuracy']:.2f}%\n")
            f.write(f"  FGSM Success Rate: {orig.get('fgsm_success_rate', 'N/A'):.2f}%\n")
            f.write(f"  PGD Success Rate: {orig.get('pgd_success_rate', 'N/A'):.2f}%\n")
            f.write(f"  C&W Success Rate: {orig.get('cw_success_rate', 'N/A'):.2f}%\n\n")

            # Pruned model results
            for sparsity in [0.2, 0.5, 0.8]:
                key = f'pruned_{sparsity:.0%}'
                if key in model_results:
                    pruned = model_results[key]
                    f.write(f"{sparsity:.0%} Pruned Model:\n")
                    f.write(f"  Clean Accuracy: {pruned['clean_accuracy']:.2f}%\n")
                    f.write(f"  FGSM Success Rate: {pruned.get('fgsm_success_rate', 'N/A'):.2f}%\n")
                    f.write(f"  PGD Success Rate: {pruned.get('pgd_success_rate', 'N/A'):.2f}%\n")
                    f.write(f"  C&W Success Rate: {pruned.get('cw_success_rate', 'N/A'):.2f}%\n\n")

            f.write("\n")

        # Trend analysis
        f.write("ROBUSTNESS TREND ANALYSIS\n")
        f.write("="*30 + "\n\n")

        for model_name, model_trends in trends.items():
            f.write(f"{model_name.upper()}:\n")
            for attack, trend_data in model_trends.items():
                f.write(f"  {attack.upper()}:\n")
                f.write(f"    Original success rate: {trend_data['original_success_rate']:.2f}%\n")
                f.write(f"    Changes with pruning:\n")
                for sparsity, change in trend_data['sparsity_changes']:
                    direction = "increased" if change > 0 else "decreased"
                    f.write(f"      {sparsity:.0%}: {direction} by {abs(change):.2f}%\n")
                f.write("\n")

        f.write("\nKEY FINDINGS:\n")
        f.write("- Pruning generally affects adversarial robustness\n")
        f.write("- Higher sparsity levels may increase vulnerability to some attacks\n")
        f.write("- Different attacks may be affected differently by pruning\n")
        f.write("- Consider robustness vs efficiency trade-offs when choosing sparsity levels\n")

    print(f"Robustness report saved to: {save_path}")


def mock_adversarial_analysis(model_configs: Dict) -> Tuple[Dict, Dict]:
    """
    Create mock adversarial robustness analysis for demonstration.

    This function simulates what a full adversarial analysis would look like
    when actual adversarial examples from Problem B are not available.
    """
    print("Creating mock adversarial robustness analysis...")

    # Mock robustness results
    robustness_results = {}

    for model_name in model_configs.keys():
        model_results = {}

        # Original model - generally more robust
        model_results['original'] = {
            'clean_accuracy': 78.5 + np.random.random() * 5,
            'fgsm_success_rate': 65.0 + np.random.random() * 10,
            'pgd_success_rate': 75.0 + np.random.random() * 10,
            'cw_success_rate': 45.0 + np.random.random() * 10,
            'total_samples': 50
        }

        # Pruned models - generally less robust as sparsity increases
        base_clean = model_results['original']['clean_accuracy']
        base_fgsm = model_results['original']['fgsm_success_rate']
        base_pgd = model_results['original']['pgd_success_rate']
        base_cw = model_results['original']['cw_success_rate']

        for sparsity in [0.2, 0.5, 0.8]:
            # Accuracy slightly decreases with pruning
            clean_acc = base_clean - sparsity * 8 + np.random.random() * 3

            # Attack success rates generally increase (worse robustness) with pruning
            fgsm_success = base_fgsm + sparsity * 15 + np.random.random() * 5
            pgd_success = base_pgd + sparsity * 12 + np.random.random() * 5
            cw_success = base_cw + sparsity * 20 + np.random.random() * 8

            model_results[f'pruned_{sparsity:.0%}'] = {
                'clean_accuracy': max(clean_acc, 10),  # Minimum 10% accuracy
                'fgsm_success_rate': min(fgsm_success, 95),  # Maximum 95% success
                'pgd_success_rate': min(pgd_success, 95),
                'cw_success_rate': min(cw_success, 95),
                'total_samples': 50
            }

        robustness_results[model_name] = model_results

    # Analyze trends
    trends = analyze_robustness_trends(robustness_results)

    return robustness_results, trends