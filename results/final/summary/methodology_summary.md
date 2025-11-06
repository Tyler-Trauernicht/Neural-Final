# Methodology Summary: EE4745 Neural Network Final Project

**Authors:** Tyler Trauernicht, Vinh Le
**Date:** November 05, 2025

---

## Experimental Design Overview

This document summarizes the methodological approach used across all three project components, ensuring reproducibility and providing a reference for future research extensions.

## Dataset Preparation

### Sports Image Dataset
- **Source:** EE4745 project dataset
- **Classes:** 10 sports categories
  - baseball, basketball, football, golf, hockey
  - rugby, swimming, tennis, volleyball, weightlifting
- **Data splits:**
  - Training: ~1,593 images (131-191 per class)
  - Validation: 50 images (5 per class)
  - Test: 50 images (5 per class)

### Preprocessing Pipeline
```python
transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Data Augmentation
- **RandomHorizontalFlip(p=0.5):** Horizontal mirroring
- **RandomRotation(degrees=10):** Small rotational variations
- **ColorJitter(brightness=0.2, contrast=0.2):** Photometric variations

## Problem A: Classification Methodology

### Model Architectures

#### SimpleCNN
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        # Conv Block 1: 3 -> 32 channels
        # Conv Block 2: 32 -> 64 channels
        # Conv Block 3: 64 -> 128 channels
        # Classifier: 2-layer MLP with dropout
```

#### ResNetSmall
```python
class ResNetSmall(nn.Module):
    def __init__(self):
        # Initial conv: 3 -> 64 channels
        # ResNet Layer 1: 64 channels, 2 blocks
        # ResNet Layer 2: 128 channels, 2 blocks
        # ResNet Layer 3: 256 channels, 2 blocks
        # Global average pooling + linear classifier
```

### Training Configuration
- **Optimizer:** Adam(lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR(T_max=50)
- **Loss function:** CrossEntropyLoss()
- **Batch size:** 32
- **Epochs:** 50 with early stopping (patience=10)
- **Device:** CPU (project requirement)

### Evaluation Metrics
- **Accuracy:** Top-1 classification accuracy
- **Per-class accuracy:** Individual class performance
- **Confusion matrix:** Misclassification patterns
- **Training efficiency:** Time and convergence analysis

## Problem B: Adversarial Attack Methodology

### Attack Implementations

#### Fast Gradient Sign Method (FGSM)
```python
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    # Untargeted attack
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)
```

#### Projected Gradient Descent (PGD)
```python
def pgd_attack(model, data, target, epsilon, alpha, iters):
    perturbed_data = data.clone()

    for i in range(iters):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + alpha * data_grad.sign()

        # Project back to epsilon ball
        eta = torch.clamp(perturbed_data - data, -epsilon, epsilon)
        perturbed_data = torch.clamp(data + eta, 0, 1)

    return perturbed_data
```

### Attack Parameters
- **FGSM epsilon values:** [0.01, 0.03, 0.05, 0.1]
- **PGD parameters:** α=0.01, iterations=40, ε=0.03
- **Target class:** Basketball (index 1)
- **Evaluation set:** Full test set (50 images)

### Transferability Testing
```python
# Generate attacks on source model
adversarial_examples = attack_method(source_model, data, labels)

# Test on target model
target_predictions = target_model(adversarial_examples)
transfer_success_rate = calculate_success_rate(target_predictions, labels)
```

## Problem C: Pruning Methodology

### Magnitude-Based Pruning
```python
def magnitude_prune(model, sparsity_level):
    parameters_to_prune = [
        (module, 'weight') for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity_level
    )

    return model
```

### Fine-tuning Protocol
- **Learning rate:** 1e-4 (10x reduction from initial training)
- **Epochs:** 10 post-pruning fine-tuning
- **Optimizer:** Adam with same weight decay
- **Metric tracking:** Accuracy recovery during fine-tuning

### Evaluation Metrics
- **Accuracy retention:** Test accuracy / baseline accuracy
- **Model size:** Memory footprint measurement
- **Inference speed:** Average forward pass time
- **Sparsity verification:** Actual zero-weight percentage

## Interpretability Analysis

### Saliency Maps
```python
def generate_saliency_map(model, input_tensor, target_class):
    input_tensor.requires_grad = True
    output = model(input_tensor)
    loss = output[0, target_class]

    loss.backward()
    saliency = input_tensor.grad.data.abs()
    return saliency
```

### Grad-CAM Implementation
```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradients)

    def generate_cam(self, input_tensor, target_class):
        # Forward pass with hooks
        # Backward pass to get gradients
        # Compute weighted activation maps
        return cam_heatmap
```

## Statistical Analysis

### Performance Comparison
- **Paired t-tests:** Statistical significance of model differences
- **Confidence intervals:** 95% CI for accuracy measurements
- **Effect size calculation:** Cohen's d for practical significance

### Experimental Controls
- **Random seed:** Fixed at 42 for reproducibility
- **Hardware standardization:** All experiments on same CPU
- **Hyperparameter consistency:** Same training configuration across models

## Reproducibility Guidelines

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run experiments
python -m src.models.simple_cnn
python -m src.models.resnet_small
```

### File Organization
```
project_root/
├── src/                    # Source code
├── data/                   # Dataset (symlinked)
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── results/               # Experimental results
│   ├── problem_a/
│   ├── problem_b/
│   ├── problem_c/
│   └── final/            # Compiled results
└── requirements.txt       # Dependencies
```

### Key Dependencies
```
torch==1.9.0
torchvision==0.10.0
numpy==1.21.0
matplotlib==3.4.2
seaborn==0.11.1
pandas==1.3.0
```

## Quality Assurance

### Validation Procedures
1. **Code review:** All implementations peer-reviewed
2. **Unit testing:** Individual components tested
3. **Integration testing:** End-to-end pipeline validation
4. **Results verification:** Cross-validation of key findings

### Error Handling
- **Input validation:** Robust handling of edge cases
- **Checkpoint saving:** Regular model state preservation
- **Logging:** Comprehensive experimental tracking
- **Exception handling:** Graceful failure recovery

## Limitations and Assumptions

### Dataset Limitations
- **Size:** Limited training samples may affect generalization
- **Diversity:** Single dataset domain may not represent all sports
- **Quality:** Image resolution constrained to 32x32 pixels

### Computational Constraints
- **CPU-only training:** Hardware limitation affecting training speed
- **Memory constraints:** May limit batch size and model size
- **Time constraints:** Limited hyperparameter exploration

### Methodological Assumptions
- **Attack model:** Assumes perfect knowledge of model architecture
- **Evaluation metrics:** Standard metrics may not capture all aspects
- **Generalization:** Results may not transfer to other domains

## Future Extensions

### Methodological Improvements
1. **Larger datasets:** Expand to more comprehensive sports databases
2. **Hardware acceleration:** GPU training for larger models
3. **Advanced attacks:** Implement physical-world attacks
4. **Structured pruning:** Channel/filter-level compression

### Evaluation Enhancements
1. **Human evaluation:** User studies for interpretability
2. **Real-world testing:** Deploy models in practical scenarios
3. **Fairness analysis:** Examine bias across different sports
4. **Robustness certification:** Formal verification methods

---

This methodology summary provides the foundation for reproducing and extending the experimental work. All code implementations follow these specifications and are available in the project repository.
