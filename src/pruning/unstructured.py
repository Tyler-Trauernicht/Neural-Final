"""
Unstructured Pruning Implementation for Problem C

This module implements magnitude-based unstructured pruning using PyTorch's pruning utilities.
Supports multiple sparsity levels with fine-tuning capability.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def get_pruneable_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """
    Get all Conv2d and Linear modules that can be pruned.

    Args:
        model: PyTorch model

    Returns:
        List of (module, parameter_name) tuples for pruning
    """
    modules_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))

    return modules_to_prune


def calculate_model_sparsity(model: nn.Module) -> float:
    """
    Calculate the overall sparsity of a model.

    Args:
        model: PyTorch model

    Returns:
        Sparsity ratio (0.0 to 1.0)
    """
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                # Pruned module
                weight = module.weight_mask * module.weight_orig
            else:
                # Unpruned module
                weight = module.weight

            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()

    return zero_params / total_params if total_params > 0 else 0.0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and non-zero parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    nonzero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param).item()

    return {
        'total': total_params,
        'nonzero': nonzero_params,
        'sparsity': 1.0 - (nonzero_params / total_params)
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def prune_model(model: nn.Module, sparsity_ratio: float) -> nn.Module:
    """
    Apply magnitude-based unstructured pruning to a model.

    Args:
        model: PyTorch model to prune
        sparsity_ratio: Target sparsity ratio (0.0 to 1.0)

    Returns:
        Pruned model
    """
    # Create a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)

    # Get modules to prune (Conv2d and Linear only)
    modules_to_prune = get_pruneable_modules(pruned_model)

    if not modules_to_prune:
        print("Warning: No pruneable modules found in the model")
        return pruned_model

    # Apply global magnitude pruning
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_ratio,
    )

    # Make pruning permanent by removing the masks and keeping only the pruned weights
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)

    # Verify actual sparsity
    actual_sparsity = calculate_model_sparsity(pruned_model)
    print(f"Target sparsity: {sparsity_ratio:.1%}, Actual sparsity: {actual_sparsity:.1%}")

    return pruned_model


def fine_tune_model(model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, device: torch.device,
                   num_epochs: int = 10, learning_rate: float = 1e-4,
                   verbose: bool = True) -> Dict[str, List[float]]:
    """
    Fine-tune a pruned model to recover performance.

    Args:
        model: Pruned model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        verbose: Whether to print progress

    Returns:
        Dictionary with training history
    """
    model.to(device)
    model.train()

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Handle case when val_loader is empty
        if val_total > 0:
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
        else:
            val_acc = 0.0
            avg_val_loss = 0.0

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            if val_total > 0:
                print(f'Epoch {epoch+1}/{num_epochs}: '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}: '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    return history


def evaluate_model_accuracy(model: nn.Module, test_loader: DataLoader,
                          device: torch.device) -> float:
    """
    Evaluate model accuracy on test set.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Test accuracy as percentage
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def measure_inference_time(model: nn.Module, test_loader: DataLoader,
                         device: torch.device, num_runs: int = 100,
                         warmup_runs: int = 10) -> Dict[str, float]:
    """
    Measure model inference time with proper methodology.

    Args:
        model: Model to benchmark
        test_loader: Test data loader
        device: Device to run on
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs to discard

    Returns:
        Dictionary with timing statistics
    """
    model.to(device)
    model.eval()

    # Get a batch for timing
    data_iter = iter(test_loader)
    data, _ = next(data_iter)
    data = data.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'batch_size': data.size(0)
    }


def evaluate_pruned_model(model: nn.Module, test_loader: DataLoader,
                         device: torch.device, model_name: str = "model") -> Dict[str, Any]:
    """
    Comprehensive evaluation of a pruned model.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        model_name: Name for logging

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")

    # Basic metrics
    param_counts = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    accuracy = evaluate_model_accuracy(model, test_loader, device)

    # Timing for different batch sizes
    timing_batch1 = measure_inference_time(model, test_loader, device)

    # Create batch size 16 loader if needed
    if test_loader.batch_size != 16:
        from torch.utils.data import DataLoader
        batch16_loader = DataLoader(
            test_loader.dataset,
            batch_size=16,
            shuffle=False,
            num_workers=test_loader.num_workers
        )
        timing_batch16 = measure_inference_time(model, batch16_loader, device)
    else:
        timing_batch16 = timing_batch1

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'total_params': param_counts['total'],
        'nonzero_params': param_counts['nonzero'],
        'sparsity': param_counts['sparsity'],
        'model_size_mb': model_size_mb,
        'inference_time_batch1': timing_batch1,
        'inference_time_batch16': timing_batch16
    }

    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Parameters: {param_counts['nonzero']:,}/{param_counts['total']:,} "
          f"(sparsity: {param_counts['sparsity']:.1%})")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print(f"  Inference time (batch=1): {timing_batch1['mean_ms']:.2f}±{timing_batch1['std_ms']:.2f} ms")
    print(f"  Inference time (batch=16): {timing_batch16['mean_ms']:.2f}±{timing_batch16['std_ms']:.2f} ms")

    return results


def save_pruned_model(model: nn.Module, filepath: str, metadata: Dict[str, Any] = None):
    """
    Save a pruned model with metadata.

    Args:
        model: Model to save
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if metadata:
        save_dict.update(metadata)

    torch.save(save_dict, filepath)
    print(f"Saved pruned model to {filepath}")


def load_pruned_model(filepath: str, model_class, *model_args, **model_kwargs) -> Tuple[nn.Module, Dict]:
    """
    Load a pruned model with metadata.

    Args:
        filepath: Path to the saved model
        model_class: Model class to instantiate
        *model_args: Arguments for model class
        **model_kwargs: Keyword arguments for model class

    Returns:
        Tuple of (model, metadata)
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    # Create model instance
    model = model_class(*model_args, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract metadata
    metadata = {k: v for k, v in checkpoint.items()
                if k not in ['model_state_dict', 'model_class']}

    return model, metadata