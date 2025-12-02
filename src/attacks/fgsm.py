import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple

class FGSM:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack implementation.

    This attack generates adversarial examples by adding perturbations in the direction
    of the gradient sign to maximize the loss function.

    References:
    - Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2014)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize FGSM attack.

        Args:
            model: Target neural network model
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def attack(self,
               inputs: torch.Tensor,
               labels: torch.Tensor,
               epsilon: float = 0.03,
               targeted: bool = False,
               target_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using FGSM.

        Args:
            inputs: Input images tensor (batch_size, channels, height, width)
            labels: True labels for inputs (batch_size,)
            epsilon: Perturbation magnitude
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack (batch_size,)

        Returns:
            Tuple of (adversarial_examples, attack_info)
            - adversarial_examples: Generated adversarial examples
            - attack_info: Dictionary containing attack statistics
        """
        # Clone inputs to avoid modifying the original tensor
        inputs = inputs.clone().detach().to(self.device)
        labels = labels.to(self.device)

        if targeted and target_labels is None:
            raise ValueError("Target labels must be provided for targeted attack")

        if targeted:
            target_labels = target_labels.to(self.device)

        # Create adversarial examples
        inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Calculate loss
        if targeted:
            # For targeted attack, minimize loss w.r.t. target labels
            loss = F.cross_entropy(outputs, target_labels)
            # We want to minimize this loss, so we'll use negative gradient
            loss_multiplier = -1.0
        else:
            # For untargeted attack, maximize loss w.r.t. true labels
            loss = F.cross_entropy(outputs, labels)
            loss_multiplier = 1.0

        # Compute gradients
        self.model.zero_grad()
        loss.backward()

        # Get the sign of gradients
        if inputs.grad is not None:
            grad_sign = inputs.grad.data.sign()
        else:
            # Fallback: use random perturbation direction if gradient is None
            grad_sign = torch.sign(torch.randn_like(inputs))

        # Generate adversarial examples
        adversarial_inputs = inputs.detach() + loss_multiplier * epsilon * grad_sign

        # Clamp to valid pixel range [0, 1] (assuming normalized inputs)
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)

        # Evaluate attack success
        with torch.no_grad():
            original_outputs = self.model(inputs.detach())
            adversarial_outputs = self.model(adversarial_inputs)

            original_predictions = original_outputs.argmax(dim=1)
            adversarial_predictions = adversarial_outputs.argmax(dim=1)

            if targeted:
                # Success if adversarial prediction matches target
                success = (adversarial_predictions == target_labels).float()
            else:
                # Success if adversarial prediction differs from original
                success = (adversarial_predictions != labels).float()

            # Calculate perturbation metrics
            perturbation = adversarial_inputs - inputs.detach()
            l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
            linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1)

            # Confidence scores
            original_confidence = F.softmax(original_outputs, dim=1).max(dim=1)[0]
            adversarial_confidence = F.softmax(adversarial_outputs, dim=1).max(dim=1)[0]

            attack_info = {
                'success_rate': success.mean().item(),
                'successful_samples': success.sum().item(),
                'total_samples': inputs.size(0),
                'epsilon': epsilon,
                'targeted': targeted,
                'mean_l2_norm': l2_norm.mean().item(),
                'mean_linf_norm': linf_norm.mean().item(),
                'original_predictions': original_predictions.cpu().numpy(),
                'adversarial_predictions': adversarial_predictions.cpu().numpy(),
                'original_confidence': original_confidence.cpu().numpy(),
                'adversarial_confidence': adversarial_confidence.cpu().numpy(),
                'l2_norms': l2_norm.cpu().numpy(),
                'linf_norms': linf_norm.cpu().numpy(),
                'success_mask': success.cpu().numpy().astype(bool)
            }

            if targeted:
                attack_info['target_labels'] = target_labels.cpu().numpy()

        return adversarial_inputs.detach(), attack_info

    def attack_single(self,
                     input_tensor: torch.Tensor,
                     true_label: int,
                     epsilon: float = 0.03,
                     targeted: bool = False,
                     target_label: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial example for a single input.

        Args:
            input_tensor: Single input image (1, channels, height, width)
            true_label: True label for the input
            epsilon: Perturbation magnitude
            targeted: Whether to perform targeted attack
            target_label: Target label for targeted attack

        Returns:
            Tuple of (adversarial_example, attack_info)
        """
        labels = torch.tensor([true_label], dtype=torch.long)
        target_labels = torch.tensor([target_label], dtype=torch.long) if target_label is not None else None

        return self.attack(input_tensor, labels, epsilon, targeted, target_labels)

    def evaluate_robustness(self,
                           data_loader,
                           epsilons: list = [0.01, 0.03, 0.05, 0.1],
                           targeted: bool = False,
                           target_class: Optional[int] = None) -> dict:
        """
        Evaluate model robustness across different epsilon values.

        Args:
            data_loader: DataLoader containing test samples
            epsilons: List of epsilon values to test
            targeted: Whether to perform targeted attacks
            target_class: Target class for targeted attacks

        Returns:
            Dictionary containing robustness evaluation results
        """
        results = {
            'epsilons': epsilons,
            'accuracy_clean': 0.0,
            'accuracy_adv': [],
            'success_rates': [],
            'mean_l2_norms': [],
            'mean_linf_norms': []
        }

        all_inputs = []
        all_labels = []

        # Collect all data
        for inputs, labels in data_loader:
            all_inputs.append(inputs)
            all_labels.append(labels)

        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Evaluate clean accuracy
        with torch.no_grad():
            all_inputs = all_inputs.to(self.device)
            all_labels = all_labels.to(self.device)
            clean_outputs = self.model(all_inputs)
            clean_predictions = clean_outputs.argmax(dim=1)
            clean_accuracy = (clean_predictions == all_labels).float().mean().item()
            results['accuracy_clean'] = clean_accuracy

        # Evaluate for each epsilon
        for epsilon in epsilons:
            print(f"Evaluating epsilon: {epsilon}")

            # Prepare target labels for targeted attack
            target_labels = None
            if targeted and target_class is not None:
                target_labels = torch.full_like(all_labels, target_class)

            # Generate adversarial examples
            adversarial_inputs, attack_info = self.attack(
                all_inputs, all_labels, epsilon, targeted, target_labels
            )

            # Evaluate adversarial accuracy
            with torch.no_grad():
                adv_outputs = self.model(adversarial_inputs)
                adv_predictions = adv_outputs.argmax(dim=1)
                adv_accuracy = (adv_predictions == all_labels).float().mean().item()

            results['accuracy_adv'].append(adv_accuracy)
            results['success_rates'].append(attack_info['success_rate'])
            results['mean_l2_norms'].append(attack_info['mean_l2_norm'])
            results['mean_linf_norms'].append(attack_info['mean_linf_norm'])

        return results

    def generate_targeted_examples(self,
                                  inputs: torch.Tensor,
                                  labels: torch.Tensor,
                                  target_class: int,
                                  epsilon: float = 0.03,
                                  num_examples: int = 10) -> Tuple[torch.Tensor, dict]:
        """
        Generate targeted adversarial examples for a specific target class.

        Args:
            inputs: Input images
            labels: True labels
            target_class: Target class to attack towards
            epsilon: Perturbation magnitude
            num_examples: Number of examples to generate

        Returns:
            Tuple of (adversarial_examples, attack_info)
        """
        # Filter out samples that already belong to target class
        mask = labels != target_class
        filtered_inputs = inputs[mask]
        filtered_labels = labels[mask]

        if filtered_inputs.size(0) == 0:
            raise ValueError(f"No samples available for targeting class {target_class}")

        # Select up to num_examples
        if filtered_inputs.size(0) > num_examples:
            indices = torch.randperm(filtered_inputs.size(0))[:num_examples]
            filtered_inputs = filtered_inputs[indices]
            filtered_labels = filtered_labels[indices]

        # Create target labels
        target_labels = torch.full_like(filtered_labels, target_class)

        return self.attack(filtered_inputs, filtered_labels, epsilon,
                          targeted=True, target_labels=target_labels)


def test_fgsm():
    """Test FGSM implementation with dummy data."""
    import torch.nn as nn

    # Create a simple test model
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
            x = self.fc(x)
            return x

    # Test setup
    model = TestModel()
    fgsm = FGSM(model, device='cpu')

    # Test data
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))

    print("Testing FGSM implementation...")

    # Test untargeted attack
    print("\n1. Testing untargeted attack:")
    adv_inputs, attack_info = fgsm.attack(inputs, labels, epsilon=0.03, targeted=False)
    print(f"   Success rate: {attack_info['success_rate']:.2f}")
    print(f"   Mean L2 norm: {attack_info['mean_l2_norm']:.4f}")
    print(f"   Mean L∞ norm: {attack_info['mean_linf_norm']:.4f}")

    # Test targeted attack
    print("\n2. Testing targeted attack:")
    target_labels = torch.full_like(labels, 5)  # Target class 5
    adv_inputs, attack_info = fgsm.attack(inputs, labels, epsilon=0.03,
                                         targeted=True, target_labels=target_labels)
    print(f"   Success rate: {attack_info['success_rate']:.2f}")
    print(f"   Mean L2 norm: {attack_info['mean_l2_norm']:.4f}")
    print(f"   Mean L∞ norm: {attack_info['mean_linf_norm']:.4f}")

    # Test single input
    print("\n3. Testing single input attack:")
    single_input = inputs[0:1]
    single_label = labels[0].item()
    adv_single, attack_info = fgsm.attack_single(single_input, single_label, epsilon=0.03)
    print(f"   Success: {attack_info['success_rate'] > 0}")
    print(f"   L2 norm: {attack_info['mean_l2_norm']:.4f}")

    print("\nFGSM test completed successfully!")


if __name__ == "__main__":
    test_fgsm()