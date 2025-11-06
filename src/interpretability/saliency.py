import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SaliencyMap:
    """Saliency map generation for model interpretability"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: Trained PyTorch model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(self, input_tensor, target_class=None):
        """
        Generate saliency map for input image

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class

        Returns:
            saliency_map: Numpy array of saliency map
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

        # Get gradients
        gradients = input_tensor.grad.data

        # Generate saliency map
        saliency_map = torch.max(gradients.abs(), dim=1)[0].squeeze()
        saliency_map = saliency_map.cpu().numpy()

        # Normalize to [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        return saliency_map, prediction, confidence

    def visualize(self, input_tensor, target_class=None, class_names=None, save_path=None):
        """
        Visualize saliency map alongside original image

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            class_names: List of class names
            save_path: Path to save the visualization

        Returns:
            fig: Matplotlib figure
        """
        # Generate saliency map
        saliency_map, prediction, confidence = self.generate(input_tensor, target_class)

        # Prepare original image for visualization
        original_image = input_tensor.squeeze().detach().cpu()

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

        # Saliency map
        im1 = axes[1].imshow(saliency_map, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(saliency_map, cmap='hot', alpha=0.5)
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
        Generate saliency maps for multiple samples

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

                # Generate saliency map
                saliency_map, prediction, confidence = self.generate(image)

                result = {
                    'image_idx': sample_count,
                    'true_label': label,
                    'predicted_label': prediction,
                    'confidence': confidence,
                    'correct': label == prediction,
                    'saliency_map': saliency_map
                }

                # Visualize and save
                if save_dir:
                    save_path = f"{save_dir}/saliency_sample_{sample_count:03d}.png"
                    fig = self.visualize(image, class_names=class_names, save_path=save_path)
                    plt.close(fig)

                results.append(result)
                sample_count += 1

            if sample_count >= num_samples:
                break

        return results

    def analyze_misclassifications(self, data_loader, class_names=None, save_dir=None):
        """
        Analyze saliency maps for misclassified samples

        Args:
            data_loader: Data loader
            class_names: List of class names
            save_dir: Directory to save visualizations

        Returns:
            misclassified_results: List of misclassified samples
        """
        misclassified_results = []

        for batch_idx, (images, labels) in enumerate(data_loader):
            for i in range(images.size(0)):
                image = images[i:i+1]
                label = labels[i].item()

                # Generate saliency map
                saliency_map, prediction, confidence = self.generate(image)

                # Only process misclassified samples
                if label != prediction:
                    result = {
                        'image_idx': len(misclassified_results),
                        'true_label': label,
                        'predicted_label': prediction,
                        'confidence': confidence,
                        'saliency_map': saliency_map
                    }

                    # Visualize and save
                    if save_dir:
                        import os
                        os.makedirs(save_dir, exist_ok=True)
                        true_class = class_names[label] if class_names else str(label)
                        pred_class = class_names[prediction] if class_names else str(prediction)
                        save_path = f"{save_dir}/misclassified_{true_class}_as_{pred_class}_{len(misclassified_results):03d}.png"
                        fig = self.visualize(image, class_names=class_names, save_path=save_path)
                        plt.close(fig)

                    misclassified_results.append(result)

        return misclassified_results