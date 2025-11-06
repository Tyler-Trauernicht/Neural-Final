import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import seaborn as sns

def denormalize_image(tensor: torch.Tensor,
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Normalization means
        std: Normalization standard deviations

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch dimension
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


def calculate_perturbation_metrics(original: torch.Tensor,
                                  adversarial: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various perturbation metrics between original and adversarial examples.

    Args:
        original: Original images
        adversarial: Adversarial images

    Returns:
        Dictionary containing perturbation metrics
    """
    perturbation = adversarial - original
    batch_size = original.size(0)

    # Flatten for norm calculations
    perturbation_flat = perturbation.view(batch_size, -1)

    # L0 norm (number of changed pixels)
    l0_norm = (perturbation_flat != 0).float().sum(dim=1)

    # L2 norm
    l2_norm = torch.norm(perturbation_flat, p=2, dim=1)

    # L∞ norm
    linf_norm = torch.norm(perturbation_flat, p=float('inf'), dim=1)

    # LPIPS would require additional model, skip for now

    # Structural similarity (simplified version)
    def ssim_batch(img1, img2):
        """Simplified SSIM calculation"""
        mu1 = img1.mean(dim=[-2, -1], keepdim=True)
        mu2 = img2.mean(dim=[-2, -1], keepdim=True)

        sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[-2, -1], keepdim=True)
        sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[-2, -1], keepdim=True)
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[-2, -1], keepdim=True)

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim.mean(dim=[1, 2, 3])  # Average over channels and spatial dimensions

    ssim_values = ssim_batch(original, adversarial)

    return {
        'mean_l0_norm': l0_norm.mean().item(),
        'mean_l2_norm': l2_norm.mean().item(),
        'mean_linf_norm': linf_norm.mean().item(),
        'mean_ssim': ssim_values.mean().item(),
        'l0_norms': l0_norm.cpu().numpy(),
        'l2_norms': l2_norm.cpu().numpy(),
        'linf_norms': linf_norm.cpu().numpy(),
        'ssim_values': ssim_values.cpu().numpy()
    }


def evaluate_attack_success(original_preds: torch.Tensor,
                           adversarial_preds: torch.Tensor,
                           true_labels: torch.Tensor,
                           target_labels: Optional[torch.Tensor] = None,
                           targeted: bool = False) -> Dict[str, float]:
    """
    Evaluate attack success rates and accuracy metrics.

    Args:
        original_preds: Original predictions
        adversarial_preds: Adversarial predictions
        true_labels: True labels
        target_labels: Target labels (for targeted attacks)
        targeted: Whether attack is targeted

    Returns:
        Dictionary containing success metrics
    """
    if targeted and target_labels is None:
        raise ValueError("Target labels required for targeted attack evaluation")

    # Basic accuracy metrics
    original_accuracy = (original_preds == true_labels).float().mean().item()
    adversarial_accuracy = (adversarial_preds == true_labels).float().mean().item()

    # Attack success metrics
    if targeted:
        # Success = adversarial prediction matches target
        attack_success = (adversarial_preds == target_labels).float()
        # Also check that it's different from true label (avoid trivial success)
        non_trivial_mask = (target_labels != true_labels)
        if non_trivial_mask.sum() > 0:
            non_trivial_success = attack_success[non_trivial_mask].mean().item()
        else:
            non_trivial_success = 0.0
    else:
        # Success = adversarial prediction differs from original
        attack_success = (adversarial_preds != original_preds).float()
        non_trivial_success = attack_success.mean().item()

    # Misclassification rate
    misclassification_rate = (adversarial_preds != true_labels).float().mean().item()

    # Confidence drop (if confidence scores provided)
    success_rate = attack_success.mean().item()

    return {
        'original_accuracy': original_accuracy,
        'adversarial_accuracy': adversarial_accuracy,
        'attack_success_rate': success_rate,
        'non_trivial_success_rate': non_trivial_success,
        'misclassification_rate': misclassification_rate,
        'accuracy_drop': original_accuracy - adversarial_accuracy,
        'successful_samples': attack_success.sum().item(),
        'total_samples': len(true_labels),
        'success_mask': attack_success.cpu().numpy().astype(bool)
    }


def visualize_adversarial_examples(original_images: torch.Tensor,
                                  adversarial_images: torch.Tensor,
                                  original_preds: np.ndarray,
                                  adversarial_preds: np.ndarray,
                                  true_labels: np.ndarray,
                                  class_names: List[str],
                                  num_examples: int = 8,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize adversarial examples alongside originals.

    Args:
        original_images: Original images tensor
        adversarial_images: Adversarial images tensor
        original_preds: Original predictions
        adversarial_preds: Adversarial predictions
        true_labels: True labels
        class_names: List of class names
        num_examples: Number of examples to visualize
        save_path: Path to save visualization

    Returns:
        Matplotlib figure
    """
    num_examples = min(num_examples, len(original_images))

    # Denormalize images for visualization
    orig_denorm = denormalize_image(original_images[:num_examples])
    adv_denorm = denormalize_image(adversarial_images[:num_examples])

    # Calculate perturbations
    perturbations = adv_denorm - orig_denorm

    fig, axes = plt.subplots(4, num_examples, figsize=(2.5 * num_examples, 10))
    if num_examples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_examples):
        # Original image
        axes[0, i].imshow(orig_denorm[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f'Original\nTrue: {class_names[true_labels[i]]}\nPred: {class_names[original_preds[i]]}',
                            fontsize=8)
        axes[0, i].axis('off')

        # Adversarial image
        axes[1, i].imshow(adv_denorm[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f'Adversarial\nPred: {class_names[adversarial_preds[i]]}',
                            fontsize=8)
        axes[1, i].axis('off')

        # Perturbation (amplified for visibility)
        pert_vis = perturbations[i].permute(1, 2, 0).cpu().numpy()
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        axes[2, i].imshow(pert_vis)
        axes[2, i].set_title('Perturbation', fontsize=8)
        axes[2, i].axis('off')

        # Difference heatmap
        diff = torch.norm(perturbations[i], dim=0).cpu().numpy()
        im = axes[3, i].imshow(diff, cmap='hot')
        axes[3, i].set_title('Perturbation Magnitude', fontsize=8)
        axes[3, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_transferability_matrix(attack_results: Dict[str, Dict[str, Dict]],
                                 model_names: List[str],
                                 attack_names: List[str],
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create transferability matrix visualization.

    Args:
        attack_results: Nested dictionary with structure:
                       {source_model: {target_model: {attack_type: results}}}
        model_names: List of model names
        attack_names: List of attack names
        save_path: Path to save visualization

    Returns:
        Matplotlib figure
    """
    n_attacks = len(attack_names)
    fig, axes = plt.subplots(1, n_attacks, figsize=(6 * n_attacks, 5))

    if n_attacks == 1:
        axes = [axes]

    for i, attack_name in enumerate(attack_names):
        # Create matrix for this attack type
        matrix = np.zeros((len(model_names), len(model_names)))

        for j, source_model in enumerate(model_names):
            for k, target_model in enumerate(model_names):
                if source_model in attack_results and target_model in attack_results[source_model]:
                    if attack_name in attack_results[source_model][target_model]:
                        success_rate = attack_results[source_model][target_model][attack_name]['success_rate']
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

        axes[i].set_title(f'{attack_name} Transferability Matrix')
        axes[i].set_xlabel('Target Model')
        axes[i].set_ylabel('Source Model')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_adversarial_examples(original_images: torch.Tensor,
                             adversarial_images: torch.Tensor,
                             metadata: Dict,
                             save_dir: str,
                             prefix: str = "adv_example") -> None:
    """
    Save adversarial examples as images with metadata.

    Args:
        original_images: Original images tensor
        adversarial_images: Adversarial images tensor
        metadata: Metadata dictionary containing labels, predictions, etc.
        save_dir: Directory to save examples
        prefix: Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)

    # Denormalize images
    orig_denorm = denormalize_image(original_images)
    adv_denorm = denormalize_image(adversarial_images)

    for i in range(len(original_images)):
        # Convert to PIL images
        orig_pil = Image.fromarray((orig_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        adv_pil = Image.fromarray((adv_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        # Save images
        orig_path = os.path.join(save_dir, f"{prefix}_{i:03d}_original.png")
        adv_path = os.path.join(save_dir, f"{prefix}_{i:03d}_adversarial.png")

        orig_pil.save(orig_path)
        adv_pil.save(adv_path)

        # Save metadata
        meta_dict = {
            'index': i,
            'original_path': orig_path,
            'adversarial_path': adv_path,
        }

        # Add relevant metadata for this sample
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > i:
                meta_dict[key] = value[i] if not isinstance(value[i], np.ndarray) else value[i].tolist()
            elif not isinstance(value, (list, np.ndarray)):
                meta_dict[key] = value

        meta_path = os.path.join(save_dir, f"{prefix}_{i:03d}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def load_model_checkpoint(model_class, checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        model_class: Model class constructor
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model parameters from checkpoint
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        model = model_class(**model_config)
    else:
        # Use default parameters if config not available
        model = model_class()

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def create_attack_summary_report(results: Dict[str, Dict], save_path: str) -> None:
    """
    Create a comprehensive attack summary report.

    Args:
        results: Dictionary containing attack results
        save_path: Path to save the report
    """
    from datetime import datetime
    report = {
        "Attack Summary Report": {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
        }
    }

    # Process each experiment
    for exp_name, exp_results in results.items():
        exp_summary = {
            "experiment_name": exp_name,
            "attack_parameters": {},
            "performance_metrics": {},
            "perturbation_metrics": {},
        }

        # Extract key metrics
        if 'success_rate' in exp_results:
            exp_summary["performance_metrics"]["success_rate"] = exp_results['success_rate']

        if 'mean_l2_norm' in exp_results:
            exp_summary["perturbation_metrics"]["mean_l2_norm"] = exp_results['mean_l2_norm']
            exp_summary["perturbation_metrics"]["mean_linf_norm"] = exp_results['mean_linf_norm']

        # Add attack parameters
        for key in ['epsilon', 'alpha', 'steps', 'targeted']:
            if key in exp_results:
                exp_summary["attack_parameters"][key] = exp_results[key]

        report[exp_name] = exp_summary

    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def calculate_attack_statistics(results_list: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate statistics across multiple attack results.

    Args:
        results_list: List of attack result dictionaries

    Returns:
        Dictionary containing aggregate statistics
    """
    if not results_list:
        return {}

    # Collect metrics
    success_rates = [r.get('success_rate', 0) for r in results_list]
    l2_norms = [r.get('mean_l2_norm', 0) for r in results_list]
    linf_norms = [r.get('mean_linf_norm', 0) for r in results_list]

    stats = {
        'mean_success_rate': np.mean(success_rates),
        'std_success_rate': np.std(success_rates),
        'min_success_rate': np.min(success_rates),
        'max_success_rate': np.max(success_rates),
        'mean_l2_norm': np.mean(l2_norms),
        'std_l2_norm': np.std(l2_norms),
        'mean_linf_norm': np.mean(linf_norms),
        'std_linf_norm': np.std(linf_norms),
        'num_experiments': len(results_list)
    }

    return stats


# Test functions
def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")

    # Test perturbation metrics
    original = torch.randn(2, 3, 32, 32)
    adversarial = original + 0.1 * torch.randn_like(original)

    metrics = calculate_perturbation_metrics(original, adversarial)
    print(f"Perturbation metrics: {metrics['mean_l2_norm']:.4f}, {metrics['mean_linf_norm']:.4f}")

    # Test success evaluation
    orig_preds = torch.tensor([0, 1, 2, 3])
    adv_preds = torch.tensor([1, 1, 3, 3])
    true_labels = torch.tensor([0, 1, 2, 3])

    success_metrics = evaluate_attack_success(orig_preds, adv_preds, true_labels)
    print(f"Attack success rate: {success_metrics['attack_success_rate']:.2f}")

    print("Utility functions test completed!")


if __name__ == "__main__":
    test_utils()