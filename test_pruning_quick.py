"""
Quick test of the pruning implementation without full fine-tuning.
This demonstrates the core functionality of Problem C.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Project imports
from src.models.simple_cnn import create_simple_cnn
from src.models.resnet_small import create_resnet_small
from src.dataset.sports_dataset import SportsDataset
from src.pruning.unstructured import (
    prune_model, evaluate_pruned_model, count_parameters, get_model_size_mb
)


def quick_test_pruning():
    """Quick test of pruning functionality."""
    print("=" * 60)
    print("QUICK PRUNING TEST - PROBLEM C")
    print("=" * 60)

    device = torch.device("cpu")  # Use CPU for quick testing

    # Setup data loader with small subset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load small test dataset
    test_dataset = SportsDataset(root_dir="data", split="valid", transform=transform)

    # Use only first 20 samples for quick testing
    small_subset = Subset(test_dataset, range(min(20, len(test_dataset))))
    test_loader = DataLoader(small_subset, batch_size=4, shuffle=False)

    print(f"Using {len(small_subset)} samples for quick testing")

    # Test models
    models_config = {
        'simple_cnn': {
            'class': create_simple_cnn,
            'args': {'num_classes': 10, 'input_size': 32, 'dropout_rate': 0.5}
        },
        'resnet_small': {
            'class': create_resnet_small,
            'args': {'num_classes': 10, 'input_size': 32}
        }
    }

    results = {}

    for model_name, config in models_config.items():
        print(f"\n{'-'*50}")
        print(f"TESTING: {model_name.upper()}")
        print(f"{'-'*50}")

        # Load original model
        checkpoint_path = f"checkpoints/{model_name}-original.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = config['class'](**config['args'])
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {model_name} from checkpoint")
        else:
            model = config['class'](**config['args'])
            print(f"Created fresh {model_name} model")

        # Evaluate original
        original_results = evaluate_pruned_model(
            model, test_loader, device, f"{model_name}-original"
        )

        model_results = {'original': original_results}

        # Test different sparsity levels
        sparsity_levels = [0.2, 0.5, 0.8]

        for sparsity in sparsity_levels:
            print(f"\nTesting {sparsity:.0%} pruning...")

            # Prune model
            pruned_model = prune_model(model, sparsity)

            # Evaluate pruned model (without fine-tuning for speed)
            pruned_results = evaluate_pruned_model(
                pruned_model, test_loader, device,
                f"{model_name}-pruned-{sparsity:.0%}"
            )

            model_results[f'pruned_{sparsity:.0%}'] = pruned_results

        results[model_name] = model_results

    # Create summary
    print("\n" + "=" * 60)
    print("PRUNING RESULTS SUMMARY")
    print("=" * 60)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 30)

        orig = model_results['original']
        print(f"Original: {orig['accuracy']:.1f}% accuracy, "
              f"{orig['nonzero_params']:,} params, "
              f"{orig['model_size_mb']:.2f} MB")

        for sparsity in [0.2, 0.5, 0.8]:
            key = f'pruned_{sparsity:.0%}'
            if key in model_results:
                res = model_results[key]
                accuracy_drop = orig['accuracy'] - res['accuracy']
                size_reduction = ((orig['model_size_mb'] - res['model_size_mb']) /
                                orig['model_size_mb']) * 100
                param_reduction = ((orig['nonzero_params'] - res['nonzero_params']) /
                                 orig['nonzero_params']) * 100

                print(f"  {sparsity:.0%} pruned: {res['accuracy']:.1f}% accuracy "
                      f"(-{accuracy_drop:.1f}%), "
                      f"{res['nonzero_params']:,} params (-{param_reduction:.1f}%), "
                      f"{res['model_size_mb']:.2f} MB (-{size_reduction:.1f}%)")

    # Create quick visualization
    create_quick_visualization(results)

    return results


def create_quick_visualization(results):
    """Create a quick visualization of the results."""
    print("\nCreating visualization...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    sparsities = [0.0, 0.2, 0.5, 0.8]

    for model_name, model_results in results.items():
        # Extract data
        accuracies = [model_results['original']['accuracy']]
        sizes = [model_results['original']['model_size_mb']]
        params = [model_results['original']['nonzero_params']]
        times = [model_results['original']['inference_time_batch1']['mean_ms']]

        for sparsity in [0.2, 0.5, 0.8]:
            key = f'pruned_{sparsity:.0%}'
            if key in model_results:
                res = model_results[key]
                accuracies.append(res['accuracy'])
                sizes.append(res['model_size_mb'])
                params.append(res['nonzero_params'])
                times.append(res['inference_time_batch1']['mean_ms'])

        # Plot accuracy vs sparsity
        ax1.plot(sparsities[:len(accuracies)], accuracies, 'o-',
                label=model_name, linewidth=2, markersize=8)

        # Plot model size vs sparsity
        ax2.plot(sparsities[:len(sizes)], sizes, 'o-',
                label=model_name, linewidth=2, markersize=8)

        # Plot parameter count vs sparsity
        ax3.plot(sparsities[:len(params)], params, 'o-',
                label=model_name, linewidth=2, markersize=8)

        # Plot inference time vs sparsity
        ax4.plot(sparsities[:len(times)], times, 'o-',
                label=model_name, linewidth=2, markersize=8)

    # Format plots
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Sparsity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Sparsity')
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size vs Sparsity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('Sparsity')
    ax3.set_ylabel('Parameters')
    ax3.set_title('Parameter Count vs Sparsity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('Sparsity')
    ax4.set_ylabel('Inference Time (ms)')
    ax4.set_title('Inference Time vs Sparsity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs("results/problem_c", exist_ok=True)
    plt.savefig("results/problem_c/quick_pruning_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Visualization saved to: results/problem_c/quick_pruning_results.png")


if __name__ == "__main__":
    results = quick_test_pruning()
    print("\nQuick pruning test completed successfully!")