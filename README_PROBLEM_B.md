# Problem B: Adversarial Attacks Implementation

This directory contains a comprehensive implementation of Problem B for EE4745 Neural Networks course, focusing on adversarial attacks against deep neural networks.

## Overview

This implementation provides:

1. **Complete Attack Implementations**: FGSM and PGD attacks with targeted and untargeted variants
2. **Transferability Analysis**: Cross-model attack evaluation framework
3. **Interpretability Analysis**: Saliency maps and Grad-CAM analysis of adversarial examples
4. **Comprehensive Evaluation**: 40+ adversarial examples with detailed metrics
5. **Automated Pipeline**: Single-command execution for complete analysis

## Project Structure

```
├── attack_problem_b.py              # Main execution script
├── src/
│   ├── attacks/
│   │   ├── fgsm.py                 # Fast Gradient Sign Method implementation
│   │   ├── pgd.py                  # Projected Gradient Descent implementation
│   │   ├── transferability.py     # Cross-model transferability analysis
│   │   ├── interpretability.py    # Adversarial interpretability analysis
│   │   ├── utils.py                # Attack utilities and evaluation metrics
│   │   └── __init__.py            # Module initialization
│   ├── dataset/
│   │   └── sports_dataset.py      # Sports dataset implementation
│   ├── models/
│   │   ├── simple_cnn.py         # SimpleCNN model
│   │   └── resnet_small.py       # ResNetSmall model
│   └── interpretability/
│       ├── saliency.py           # Saliency map generation
│       └── gradcam.py            # Grad-CAM visualization
└── results/problem_b/             # Generated results directory
```

## Features Implemented

### 1. Attack Methods

#### FGSM (Fast Gradient Sign Method)
- **Untargeted attacks**: Cause misclassification to any wrong class
- **Targeted attacks**: Force classification to "basketball" class
- **Multiple epsilon values**: [0.01, 0.03, 0.05, 0.1] for robustness evaluation
- **Single-step gradient-based perturbations**

#### PGD (Projected Gradient Descent)
- **Iterative attacks**: 40 steps with α=0.01, ε=0.03
- **Projection constraints**: L∞ ball projection
- **Random initialization**: Optional random start within epsilon ball
- **Adaptive attacks**: Early stopping and parameter adjustment

### 2. Transferability Analysis

- **Cross-model evaluation**: Test adversarial examples across SimpleCNN and ResNetSmall
- **Transfer matrices**: Visualization of attack success rates between model pairs
- **Statistical analysis**: Transferability ratios and cross-model success rates
- **Comprehensive metrics**: Per-attack-type transferability evaluation

### 3. Interpretability Analysis

#### Saliency Maps
- **Gradient-based attribution**: Visualize pixel importance for predictions
- **Clean vs adversarial comparison**: Show attention shift due to perturbations
- **Similarity metrics**: Correlation analysis between clean and adversarial saliency

#### Grad-CAM Visualization
- **Layer-wise activation**: Target layer selection for model architectures
- **Heatmap generation**: Class-specific attention visualization
- **Comparative analysis**: Side-by-side clean vs adversarial attention patterns

### 4. Evaluation Metrics

#### Attack Success Metrics
- **Success rate**: Percentage of successful adversarial examples
- **Perturbation norms**: L2 and L∞ norm measurements
- **Confidence analysis**: Prediction confidence changes
- **Misclassification rate**: Overall robustness assessment

#### Transferability Metrics
- **Same-model success**: Attack success on source model
- **Cross-model success**: Attack success on different target models
- **Transferability ratio**: Cross-model / same-model success ratio
- **Attack-type analysis**: Method-specific transferability patterns

## Usage

### Basic Execution

Run the complete Problem B analysis:

```bash
python3 attack_problem_b.py
```

### Advanced Options

```bash
python3 attack_problem_b.py \
    --data-dir data \
    --results-dir results/problem_b \
    --num-samples 50 \
    --batch-size 10 \
    --device cpu
```

### Parameters

- `--data-dir`: Path to sports dataset directory
- `--results-dir`: Output directory for results
- `--num-samples`: Number of test samples to analyze (default: 50)
- `--batch-size`: Batch size for processing (default: 10)
- `--device`: Computing device (`cpu` or `cuda`)

## Generated Outputs

### 1. Adversarial Examples
- **Location**: `results/problem_b/adversarial_examples/{model_name}/`
- **Content**: Original and adversarial image pairs with metadata
- **Format**: PNG images + JSON metadata files

### 2. Transferability Analysis
- **Location**: `results/problem_b/transferability/`
- **Files**:
  - `transferability_results.json`: Raw transferability data
  - `transferability_metrics.json`: Aggregate metrics
  - `transferability_matrix.png`: Visualization matrix

### 3. Interpretability Analysis
- **Location**: `results/problem_b/interpretability/{model_name}/`
- **Subdirectories**:
  - `saliency/`: Saliency map comparisons
  - `gradcam/`: Grad-CAM visualizations
- **Content**: Clean vs adversarial interpretability comparisons

### 4. Summary Reports
- **Location**: `results/problem_b/`
- **Files**:
  - `experiment_summary.json`: Machine-readable summary
  - `experiment_summary.txt`: Human-readable summary

## Attack Configuration

### Basketball Target Class

The implementation targets the "basketball" class (index 1) for targeted attacks, as specified in the project requirements:

```python
BASKETBALL_CLASS = SportsDataset.CLASSES.index('basketball')  # Index 1
```

### Attack Parameters

- **FGSM Epsilons**: [0.01, 0.03, 0.05, 0.1]
- **PGD Configuration**: ε=0.03, α=0.01, steps=40
- **Normalization**: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- **Perturbation Constraints**: L∞ ball projections

## Model Loading

The script attempts to load trained models from Problem A:

```
checkpoints/simple_cnn-original.pt
checkpoints/resnet_small-original.pt
```

If checkpoints are not found, untrained models are created as fallbacks for demonstration purposes.

## Technical Implementation

### White-box Attacks

All attacks assume full access to model parameters, gradients, and architecture:

- **Gradient computation**: Automatic differentiation with PyTorch
- **Loss functions**: Cross-entropy for classification
- **Optimization**: First-order gradient methods

### Attack Success Criteria

- **Untargeted**: Adversarial prediction ≠ original prediction
- **Targeted**: Adversarial prediction = target class (basketball)
- **Non-trivial**: Target class ≠ original true class

### Transferability Evaluation

Cross-model attack evaluation matrix:

|               | SimpleCNN Target | ResNetSmall Target |
|---------------|------------------|--------------------|
| SimpleCNN Source | Same-model      | Cross-model        |
| ResNetSmall Source | Cross-model    | Same-model         |

## Results Analysis

### Expected Deliverables

1. **40+ Adversarial Examples**: 10 per attack type × 2 models × 2+ attack methods
2. **Transferability Matrix**: Success rates across model pairs
3. **Interpretability Comparisons**: Saliency and Grad-CAM analysis
4. **Statistical Analysis**: Success rates, perturbation norms, transferability ratios

### Key Metrics

- **Attack Success Rates**: Percentage of successful adversarial examples
- **Perturbation Magnitudes**: L2 and L∞ norms of perturbations
- **Transferability Ratios**: Cross-model vs same-model success rates
- **Attention Correlations**: Saliency map similarities between clean/adversarial

## Dependencies

Core requirements:
- PyTorch ≥ 1.9.0
- torchvision
- numpy
- matplotlib
- seaborn
- PIL
- opencv-python

Install via:
```bash
pip install -r requirements.txt
```

## Testing

Individual component tests:

```bash
# Test FGSM implementation
python3 -m src.attacks.fgsm

# Test PGD implementation
python3 -m src.attacks.pgd

# Test transferability analyzer
python3 -m src.attacks.transferability

# Test interpretability analyzer
python3 -m src.attacks.interpretability
```

## Extension Possibilities

### Bonus Defense Implementation (+5 points)

The codebase is designed to support defense mechanisms:

1. **PGD Adversarial Training**: Train models on adversarial examples
2. **Input Preprocessing**: Gaussian smoothing, bit-depth reduction
3. **Certified Defenses**: Randomized smoothing techniques
4. **Detection Methods**: Statistical anomaly detection

### Additional Attack Methods

The modular design supports easy addition of:

- C&W attacks
- AutoAttack
- Semantic attacks
- Physical-world attacks

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `--num-samples` or `--batch-size`
2. **No GPU**: Set `--device cpu` explicitly
3. **Missing checkpoints**: Script creates untrained models as fallbacks
4. **Import errors**: Ensure all dependencies are installed

### Performance Optimization

- Use GPU if available: `--device cuda`
- Increase batch size for faster processing
- Reduce sample count for quick testing

## Academic Integrity

This implementation is provided as educational reference for EE4745 coursework. Students should:

1. Understand the algorithmic details
2. Implement core components independently
3. Use as reference for debugging and verification
4. Acknowledge usage appropriately in submissions

## References

1. Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2014)
2. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
3. Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
4. Simonyan et al. "Deep Inside Convolutional Networks: Visualising Image Classification Models" (2014)

---

**Author**: Claude Code
**Course**: EE4745 Neural Networks
**Problem**: B - Adversarial Attacks
**Implementation Date**: November 2024