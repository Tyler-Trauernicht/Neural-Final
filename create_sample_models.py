"""
Create sample pre-trained models for Problem C testing.

This script creates mock trained models that can be used to test the pruning pipeline
when Problem A checkpoints are not available.
"""

import torch
import torch.nn as nn
import os
from src.models.simple_cnn import create_simple_cnn
from src.models.resnet_small import create_resnet_small
from src.dataset.sports_dataset import SportsDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def create_mock_trained_model(model_class, model_args, model_name, checkpoints_dir):
    """
    Create a mock trained model with reasonable accuracy.

    Args:
        model_class: Model class to create
        model_args: Arguments for model creation
        model_name: Name of the model
        checkpoints_dir: Directory to save checkpoint
    """
    print(f"Creating mock trained model: {model_name}")

    # Create model
    model = model_class(**model_args)

    # Initialize with better weights (not totally random)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'accuracy': 75.0 + torch.rand(1).item() * 10,  # Mock accuracy between 75-85%
        'epoch': 50,
        'optimizer_state_dict': None,  # Not needed for inference
    }

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}-original.pt")
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    print(f"Mock model saved to: {checkpoint_path}")
    print(f"Mock accuracy: {checkpoint['accuracy']:.2f}%")

    return model


def quick_test_model(model, data_dir):
    """Quick test to make sure model works with data."""
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load small subset of data
    try:
        test_dataset = SportsDataset(
            root_dir=data_dir,
            split='valid',
            transform=transform
        )

        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                print(f"Test batch {batch_idx}: Input {data.shape} -> Output {output.shape}")
                break  # Just test one batch

        print(f"Model test successful! Dataset has {len(test_dataset)} samples")

    except Exception as e:
        print(f"Model test failed: {e}")


def main():
    """Create sample models for testing."""
    print("Creating sample pre-trained models for Problem C testing...")

    checkpoints_dir = "checkpoints"
    data_dir = "data"

    # Model configurations
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

    # Create models
    for model_name, config in models_config.items():
        model = create_mock_trained_model(
            config['class'],
            config['args'],
            model_name,
            checkpoints_dir
        )

        # Quick test
        quick_test_model(model, data_dir)
        print("-" * 50)

    print("Sample models created successfully!")


if __name__ == "__main__":
    main()