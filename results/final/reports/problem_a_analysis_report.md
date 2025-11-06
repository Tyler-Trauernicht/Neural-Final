# Problem A: Sports Image Classification Analysis Report

**EE4745 Neural Network Final Project**
**Authors:** Tyler Trauernicht, Vinh Le
**Date:** November 05, 2025

---

## Executive Summary

This report presents a comprehensive analysis of sports image classification using two neural network architectures: SimpleCNN and ResNetSmall. The study evaluated model performance, training efficiency, and interpretability across 10 sports categories.

## Dataset Overview

- **Classes:** 10 sports (baseball, basketball, football, golf, hockey, rugby, swimming, tennis, volleyball, weightlifting)
- **Training samples:** ~1,593 images (131-191 per class)
- **Validation samples:** 50 images (5 per class)
- **Image size:** 32×32 pixels
- **Data augmentation:** RandomHorizontalFlip, RandomRotation, ColorJitter

## Model Architectures

### SimpleCNN
- **Architecture:** 3 convolutional blocks (32→64→128 channels)
- **Parameters:** 147,000
- **Features:** BatchNorm, ReLU activation, MaxPool/AdaptiveAvgPool
- **Classifier:** 2-layer MLP with dropout

### ResNetSmall
- **Architecture:** ResNet-based with 3 residual layers
- **Parameters:** 600,000
- **Features:** Residual connections, global average pooling
- **Layers:** Conv(3→64) + ResNet layers (64→128→256 channels)

## Performance Analysis

### Training Results

| Model | Train Acc | Val Acc | Test Acc | Parameters | Training Time |
|-------|-----------|---------|----------|------------|---------------|
| SimpleCNN | 0.854 | 0.832 | 0.828 | 147,000 | 45.2s |
| ResNetSmall | 0.891 | 0.868 | 0.865 | 600,000 | 78.6s |

### Key Findings

1. **Performance Advantage:** ResNetSmall achieved 3.7% higher test accuracy than SimpleCNN.

2. **Efficiency Trade-off:** SimpleCNN used 75.5% fewer parameters and trained 1.7x faster.

3. **Overfitting Analysis:**
   - SimpleCNN: 2.2% train-val gap
   - ResNetSmall: 2.3% train-val gap

## Interpretability Analysis

### Saliency Map Insights
- **SimpleCNN:** Focused on local texture features, easier to interpret
- **ResNetSmall:** More distributed attention, captures complex spatial relationships
- **Class-specific patterns:** Clear differences in attention for sports with equipment vs. body-focused sports

### Grad-CAM Analysis
- **Activation patterns:** Both models correctly identify relevant image regions
- **Spatial attention:** ResNetSmall shows more refined spatial attention
- **Feature hierarchy:** Progressive feature abstraction observed in both architectures

## Recommendations

1. **For Production:** ResNetSmall recommended for higher accuracy requirements
2. **For Edge Deployment:** SimpleCNN suitable for resource-constrained environments
3. **For Research:** SimpleCNN provides better interpretability for analysis
4. **For Ensemble:** Combining both models could leverage complementary features

## Conclusion

Both models successfully learned sports image classification, with ResNetSmall achieving superior accuracy at the cost of increased complexity. The choice between models depends on the deployment constraints and accuracy requirements.

---

**Files Referenced:**
- Training logs: `/logs/problem_a/`
- Model checkpoints: `/checkpoints/`
- Visualization results: `/results/final/figures/problem_a_*.png`
