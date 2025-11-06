import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class GradCAM:
    """Grad-CAM implementation for CNN interpretability"""

    def __init__(self, model, target_layer, device='cpu'):
        """
        Args:
            model: Trained PyTorch model
            target_layer: Target layer for Grad-CAM (e.g., model.features[-3])
            device: Device to run computations on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class

        Returns:
            heatmap: Grad-CAM heatmap
            prediction: Model prediction
            confidence: Prediction confidence
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1).max().item()

        # Use target class or predicted class
        if target_class is None:
            target_class = prediction

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # Apply ReLU to heatmap
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize heatmap to input size
        input_size = input_tensor.shape[-2:]
        heatmap = cv2.resize(heatmap, (input_size[1], input_size[0]))

        return heatmap, prediction, confidence

    def visualize(self, input_tensor, target_class=None, class_names=None, save_path=None):
        """
        Visualize Grad-CAM heatmap alongside original image

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            class_names: List of class names
            save_path: Path to save the visualization

        Returns:
            fig: Matplotlib figure
        """
        # Generate Grad-CAM
        heatmap, prediction, confidence = self.generate(input_tensor, target_class)

        # Prepare original image for visualization
        original_image = input_tensor.squeeze().cpu()

        # Denormalize image (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = original_image * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        original_image = original_image.permute(1, 2, 0).numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        # Add prediction information
        pred_text = f'Prediction: {prediction}'
        if class_names:
            pred_text = f'Prediction: {class_names[prediction]}'
        pred_text += f'\nConfidence: {confidence:.3f}'
        if target_class is not None and target_class != prediction:
            target_text = f'\nTarget: {target_class}'
            if class_names:
                target_text = f'\nTarget: {class_names[target_class]}'
            pred_text += target_text

        fig.suptitle(pred_text, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_batch(self, data_loader, num_samples=10, class_names=None, save_dir=None):
        """
        Generate Grad-CAM heatmaps for multiple samples

        Args:
            data_loader: Data loader
            num_samples: Number of samples to process
            class_names: List of class names
            save_dir: Directory to save visualizations

        Returns:
            results: List of results
        """
        results = []
        sample_count = 0

        for batch_idx, (images, labels) in enumerate(data_loader):
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break

                image = images[i:i+1]  # Keep batch dimension
                label = labels[i].item()

                # Generate Grad-CAM
                heatmap, prediction, confidence = self.generate(image)

                result = {
                    'image_idx': sample_count,
                    'true_label': label,
                    'predicted_label': prediction,
                    'confidence': confidence,
                    'correct': label == prediction,
                    'heatmap': heatmap
                }

                # Visualize and save
                if save_dir:
                    save_path = f"{save_dir}/gradcam_sample_{sample_count:03d}.png"
                    fig = self.visualize(image, class_names=class_names, save_path=save_path)
                    plt.close(fig)

                results.append(result)
                sample_count += 1

            if sample_count >= num_samples:
                break

        return results

    def compare_classes(self, input_tensor, class_indices, class_names=None, save_path=None):
        """
        Compare Grad-CAM heatmaps for different target classes

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_indices: List of class indices to compare
            class_names: List of class names
            save_path: Path to save the visualization

        Returns:
            fig: Matplotlib figure
        """
        num_classes = len(class_indices)
        fig, axes = plt.subplots(2, num_classes + 1, figsize=(4 * (num_classes + 1), 8))

        # Prepare original image
        original_image = input_tensor.squeeze().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = original_image * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        original_image = original_image.permute(1, 2, 0).numpy()

        # Show original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')

        # Generate heatmaps for each class
        for i, class_idx in enumerate(class_indices):
            heatmap, prediction, confidence = self.generate(input_tensor, class_idx)

            # Heatmap
            im = axes[0, i + 1].imshow(heatmap, cmap='jet')
            title = f'Class {class_idx}'
            if class_names:
                title = f'{class_names[class_idx]}'
            axes[0, i + 1].set_title(title)
            axes[0, i + 1].axis('off')

            # Overlay
            axes[1, i + 1].imshow(original_image)
            axes[1, i + 1].imshow(heatmap, cmap='jet', alpha=0.4)
            axes[1, i + 1].set_title(f'Overlay')
            axes[1, i + 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def get_target_layer(model, model_name):
    """
    Get appropriate target layer for Grad-CAM based on model architecture

    Args:
        model: PyTorch model
        model_name: Name of the model ('SimpleCNN' or 'ResNetSmall')

    Returns:
        target_layer: Target layer for Grad-CAM
    """
    if 'SimpleCNN' in model_name or 'simple' in model_name.lower():
        # For SimpleCNN, use the last convolutional layer before adaptive pooling
        return model.features[-2]  # BatchNorm2d before AdaptiveAvgPool2d
    elif 'ResNet' in model_name or 'resnet' in model_name.lower():
        # For ResNet, use the last layer before global average pooling
        return model.layer3[-1]  # Last block in layer3
    else:
        # Default: try to find the last convolutional layer
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))

        if conv_layers:
            return conv_layers[-1][1]  # Return the last conv layer
        else:
            raise ValueError(f"Cannot find suitable target layer for model: {model_name}")


if __name__ == "__main__":
    # Test Grad-CAM with dummy model and data
    import torch.nn as nn

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.classifier = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = TestModel()
    target_layer = model.features[-2]  # Last conv layer

    gradcam = GradCAM(model, target_layer)

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    heatmap, pred, conf = gradcam.generate(dummy_input)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Prediction: {pred}, Confidence: {conf:.3f}")
    print("Grad-CAM test completed successfully!")