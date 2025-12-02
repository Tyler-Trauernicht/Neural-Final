import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple

class PGD:
    """
    Projected Gradient Descent (PGD) adversarial attack implementation.

    This attack generates adversarial examples through iterative gradient-based optimization
    with projection back to the L∞ ball around the original input.

    References:
    - Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize PGD attack.

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
               alpha: float = 0.01,
               steps: int = 40,
               targeted: bool = False,
               target_labels: Optional[torch.Tensor] = None,
               random_start: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using PGD.

        Args:
            inputs: Input images tensor (batch_size, channels, height, width)
            labels: True labels for inputs (batch_size,)
            epsilon: Maximum perturbation magnitude (L∞ bound)
            alpha: Step size for each iteration
            steps: Number of PGD iterations
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack (batch_size,)
            random_start: Whether to start from random point in epsilon ball

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

        # Initialize adversarial examples
        if random_start:
            # Start from random point within epsilon ball
            delta = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
            adversarial_inputs = torch.clamp(inputs + delta, 0, 1)
        else:
            # Start from original inputs
            adversarial_inputs = inputs.clone()

        # Store iteration statistics
        iteration_stats = {
            'losses': [],
            'success_rates': [],
            'l2_norms': [],
            'linf_norms': []
        }

        # PGD iterations
        for step in range(steps):
            adversarial_inputs.requires_grad_(True)

            # Forward pass
            outputs = self.model(adversarial_inputs)

            # Calculate loss
            if targeted:
                # For targeted attack, minimize loss w.r.t. target labels
                loss = F.cross_entropy(outputs, target_labels)
                loss_multiplier = -1.0
            else:
                # For untargeted attack, maximize loss w.r.t. true labels
                loss = F.cross_entropy(outputs, labels)
                loss_multiplier = 1.0

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Update adversarial examples
            with torch.no_grad():
                # Get gradient and normalize
                if adversarial_inputs.grad is not None:
                    grad = adversarial_inputs.grad.data
                else:
                    # Fallback: use random direction if gradient is None
                    grad = torch.randn_like(adversarial_inputs)

                # Take step in gradient direction
                adversarial_inputs = adversarial_inputs.detach() + loss_multiplier * alpha * grad.sign()

                # Project back to epsilon ball around original input
                delta = adversarial_inputs - inputs
                delta = torch.clamp(delta, -epsilon, epsilon)
                adversarial_inputs = torch.clamp(inputs + delta, 0, 1)

                # Evaluate current iteration
                with torch.no_grad():
                    current_outputs = self.model(adversarial_inputs)
                    current_predictions = current_outputs.argmax(dim=1)

                    if targeted:
                        success = (current_predictions == target_labels).float()
                    else:
                        success = (current_predictions != labels).float()

                    # Calculate perturbation metrics
                    perturbation = adversarial_inputs - inputs
                    l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
                    linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1)

                    iteration_stats['losses'].append(loss.item())
                    iteration_stats['success_rates'].append(success.mean().item())
                    iteration_stats['l2_norms'].append(l2_norm.mean().item())
                    iteration_stats['linf_norms'].append(linf_norm.mean().item())

        # Final evaluation
        with torch.no_grad():
            original_outputs = self.model(inputs)
            adversarial_outputs = self.model(adversarial_inputs)

            original_predictions = original_outputs.argmax(dim=1)
            adversarial_predictions = adversarial_outputs.argmax(dim=1)

            if targeted:
                success = (adversarial_predictions == target_labels).float()
            else:
                success = (adversarial_predictions != labels).float()

            # Calculate perturbation metrics
            perturbation = adversarial_inputs - inputs
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
                'alpha': alpha,
                'steps': steps,
                'targeted': targeted,
                'random_start': random_start,
                'mean_l2_norm': l2_norm.mean().item(),
                'mean_linf_norm': linf_norm.mean().item(),
                'original_predictions': original_predictions.cpu().numpy(),
                'adversarial_predictions': adversarial_predictions.cpu().numpy(),
                'original_confidence': original_confidence.cpu().numpy(),
                'adversarial_confidence': adversarial_confidence.cpu().numpy(),
                'l2_norms': l2_norm.cpu().numpy(),
                'linf_norms': linf_norm.cpu().numpy(),
                'success_mask': success.cpu().numpy().astype(bool),
                'iteration_stats': iteration_stats
            }

            if targeted:
                attack_info['target_labels'] = target_labels.cpu().numpy()

        return adversarial_inputs.detach(), attack_info

    def attack_single(self,
                     input_tensor: torch.Tensor,
                     true_label: int,
                     epsilon: float = 0.03,
                     alpha: float = 0.01,
                     steps: int = 40,
                     targeted: bool = False,
                     target_label: Optional[int] = None,
                     random_start: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial example for a single input.

        Args:
            input_tensor: Single input image (1, channels, height, width)
            true_label: True label for the input
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations
            targeted: Whether to perform targeted attack
            target_label: Target label for targeted attack
            random_start: Whether to start from random point

        Returns:
            Tuple of (adversarial_example, attack_info)
        """
        labels = torch.tensor([true_label], dtype=torch.long)
        target_labels = torch.tensor([target_label], dtype=torch.long) if target_label is not None else None

        return self.attack(input_tensor, labels, epsilon, alpha, steps,
                          targeted, target_labels, random_start)

    def evaluate_robustness(self,
                           data_loader,
                           epsilon: float = 0.03,
                           alpha: float = 0.01,
                           steps: int = 40,
                           targeted: bool = False,
                           target_class: Optional[int] = None) -> dict:
        """
        Evaluate model robustness using PGD attack.

        Args:
            data_loader: DataLoader containing test samples
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations
            targeted: Whether to perform targeted attacks
            target_class: Target class for targeted attacks

        Returns:
            Dictionary containing robustness evaluation results
        """
        results = {
            'parameters': {
                'epsilon': epsilon,
                'alpha': alpha,
                'steps': steps,
                'targeted': targeted
            },
            'accuracy_clean': 0.0,
            'accuracy_adv': 0.0,
            'success_rate': 0.0,
            'mean_l2_norm': 0.0,
            'mean_linf_norm': 0.0,
            'convergence_stats': {
                'final_losses': [],
                'success_rates_per_step': [],
                'l2_norms_per_step': [],
                'linf_norms_per_step': []
            }
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

        # Prepare target labels for targeted attack
        target_labels = None
        if targeted and target_class is not None:
            target_labels = torch.full_like(all_labels, target_class)

        # Generate adversarial examples
        adversarial_inputs, attack_info = self.attack(
            all_inputs, all_labels, epsilon, alpha, steps, targeted, target_labels
        )

        # Evaluate adversarial accuracy
        with torch.no_grad():
            adv_outputs = self.model(adversarial_inputs)
            adv_predictions = adv_outputs.argmax(dim=1)
            adv_accuracy = (adv_predictions == all_labels).float().mean().item()

        results['accuracy_adv'] = adv_accuracy
        results['success_rate'] = attack_info['success_rate']
        results['mean_l2_norm'] = attack_info['mean_l2_norm']
        results['mean_linf_norm'] = attack_info['mean_linf_norm']
        results['convergence_stats'] = attack_info['iteration_stats']

        return results

    def generate_targeted_examples(self,
                                  inputs: torch.Tensor,
                                  labels: torch.Tensor,
                                  target_class: int,
                                  epsilon: float = 0.03,
                                  alpha: float = 0.01,
                                  steps: int = 40,
                                  num_examples: int = 10) -> Tuple[torch.Tensor, dict]:
        """
        Generate targeted adversarial examples for a specific target class.

        Args:
            inputs: Input images
            labels: True labels
            target_class: Target class to attack towards
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations
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

        return self.attack(filtered_inputs, filtered_labels, epsilon, alpha, steps,
                          targeted=True, target_labels=target_labels)

    def adaptive_attack(self,
                       inputs: torch.Tensor,
                       labels: torch.Tensor,
                       epsilon: float = 0.03,
                       max_steps: int = 100,
                       targeted: bool = False,
                       target_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Adaptive PGD attack that adjusts step size and stops early when successful.

        Args:
            inputs: Input images tensor
            labels: True labels for inputs
            epsilon: Maximum perturbation magnitude
            max_steps: Maximum number of iterations
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack

        Returns:
            Tuple of (adversarial_examples, attack_info)
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if targeted and target_labels is None:
            raise ValueError("Target labels must be provided for targeted attack")

        if targeted:
            target_labels = target_labels.to(self.device)

        # Adaptive parameters
        alpha_start = epsilon / 4
        alpha_decay = 0.98
        early_stop_threshold = 0.95  # Stop when 95% of samples are successful

        # Initialize
        delta = torch.zeros_like(inputs)
        alpha = alpha_start

        success_history = []
        step = 0

        while step < max_steps:
            adversarial_inputs = inputs + delta
            adversarial_inputs.requires_grad_(True)

            # Forward pass
            outputs = self.model(adversarial_inputs)

            # Calculate loss
            if targeted:
                loss = F.cross_entropy(outputs, target_labels)
                loss_multiplier = -1.0
            else:
                loss = F.cross_entropy(outputs, labels)
                loss_multiplier = 1.0

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Update
            with torch.no_grad():
                grad = adversarial_inputs.grad.data
                delta = delta + loss_multiplier * alpha * grad.sign()

                # Project back to epsilon ball
                delta = torch.clamp(delta, -epsilon, epsilon)
                adversarial_inputs = torch.clamp(inputs + delta, 0, 1)
                delta = adversarial_inputs - inputs

                # Check success
                current_outputs = self.model(adversarial_inputs)
                current_predictions = current_outputs.argmax(dim=1)

                if targeted:
                    success = (current_predictions == target_labels).float()
                else:
                    success = (current_predictions != labels).float()

                success_rate = success.mean().item()
                success_history.append(success_rate)

                # Early stopping
                if success_rate >= early_stop_threshold:
                    print(f"Early stopping at step {step + 1}, success rate: {success_rate:.3f}")
                    break

                # Decay alpha
                alpha *= alpha_decay

            step += 1

        # Final evaluation
        adversarial_inputs = inputs + delta
        with torch.no_grad():
            adversarial_outputs = self.model(adversarial_inputs)
            adversarial_predictions = adversarial_outputs.argmax(dim=1)

            if targeted:
                final_success = (adversarial_predictions == target_labels).float()
            else:
                final_success = (adversarial_predictions != labels).float()

            perturbation = delta
            l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
            linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1)

            attack_info = {
                'success_rate': final_success.mean().item(),
                'steps_taken': step + 1,
                'mean_l2_norm': l2_norm.mean().item(),
                'mean_linf_norm': linf_norm.mean().item(),
                'success_history': success_history,
                'adaptive': True,
                'epsilon': epsilon,
                'alpha_start': alpha_start,
                'alpha_final': alpha
            }

        return adversarial_inputs.detach(), attack_info


def test_pgd():
    """Test PGD implementation with dummy data."""
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
    pgd = PGD(model, device='cpu')

    # Test data
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))

    print("Testing PGD implementation...")

    # Test untargeted attack
    print("\n1. Testing untargeted attack:")
    adv_inputs, attack_info = pgd.attack(inputs, labels, epsilon=0.03, alpha=0.01,
                                        steps=10, targeted=False)
    print(f"   Success rate: {attack_info['success_rate']:.2f}")
    print(f"   Mean L2 norm: {attack_info['mean_l2_norm']:.4f}")
    print(f"   Mean L∞ norm: {attack_info['mean_linf_norm']:.4f}")
    print(f"   Steps: {attack_info['steps']}")

    # Test targeted attack
    print("\n2. Testing targeted attack:")
    target_labels = torch.full_like(labels, 5)  # Target class 5
    adv_inputs, attack_info = pgd.attack(inputs, labels, epsilon=0.03, alpha=0.01,
                                        steps=10, targeted=True, target_labels=target_labels)
    print(f"   Success rate: {attack_info['success_rate']:.2f}")
    print(f"   Mean L2 norm: {attack_info['mean_l2_norm']:.4f}")
    print(f"   Mean L∞ norm: {attack_info['mean_linf_norm']:.4f}")

    # Test adaptive attack
    print("\n3. Testing adaptive attack:")
    adv_inputs, attack_info = pgd.adaptive_attack(inputs, labels, epsilon=0.03,
                                                 max_steps=20, targeted=False)
    print(f"   Success rate: {attack_info['success_rate']:.2f}")
    print(f"   Steps taken: {attack_info['steps_taken']}")
    print(f"   Mean L2 norm: {attack_info['mean_l2_norm']:.4f}")

    print("\nPGD test completed successfully!")


if __name__ == "__main__":
    test_pgd()