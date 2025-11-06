"""
Basic test of pruning functionality.
"""

import torch
import torch.nn as nn
from src.models.simple_cnn import create_simple_cnn
from src.pruning.unstructured import prune_model, count_parameters, get_model_size_mb

def basic_test():
    print("Testing basic pruning functionality...")

    # Create a simple model
    model = create_simple_cnn(num_classes=10, input_size=32, dropout_rate=0.5)

    print(f"Original model:")
    orig_params = count_parameters(model)
    orig_size = get_model_size_mb(model)
    print(f"  Parameters: {orig_params['total']:,} total, {orig_params['nonzero']:,} non-zero")
    print(f"  Sparsity: {orig_params['sparsity']:.1%}")
    print(f"  Size: {orig_size:.2f} MB")

    # Test pruning at different levels
    sparsity_levels = [0.2, 0.5, 0.8]

    for sparsity in sparsity_levels:
        print(f"\nPruning at {sparsity:.0%} sparsity:")

        # Prune model
        pruned_model = prune_model(model, sparsity)

        # Check results
        pruned_params = count_parameters(pruned_model)
        pruned_size = get_model_size_mb(pruned_model)

        print(f"  Parameters: {pruned_params['total']:,} total, {pruned_params['nonzero']:,} non-zero")
        print(f"  Sparsity: {pruned_params['sparsity']:.1%}")
        print(f"  Size: {pruned_size:.2f} MB")

        # Calculate reductions
        param_reduction = (orig_params['nonzero'] - pruned_params['nonzero']) / orig_params['nonzero'] * 100
        size_reduction = (orig_size - pruned_size) / orig_size * 100

        print(f"  Parameter reduction: {param_reduction:.1f}%")
        print(f"  Size reduction: {size_reduction:.1f}%")

    print("\nBasic pruning test completed successfully!")

if __name__ == "__main__":
    basic_test()