# Problem C: Model Compression via Pruning Analysis Report

**EE4745 Neural Network Final Project**
**Authors:** Tyler Trauernicht, Vinh Le
**Date:** November 05, 2025

---

## Executive Summary

This report analyzes unstructured magnitude-based pruning for neural network compression. We evaluated three sparsity levels (20%, 50%, 80%) on both SimpleCNN and ResNetSmall models, examining the trade-offs between model size, inference speed, accuracy retention, and adversarial robustness.

## Pruning Methodology

### Unstructured Magnitude-Based Pruning
- **Method:** Remove weights with smallest absolute magnitude
- **Granularity:** Individual weight-level pruning
- **Layers targeted:** Conv2d and Linear layers
- **Sparsity levels:** 20%, 50%, 80%

### Fine-tuning Protocol
- **Epochs:** 10 epochs post-pruning
- **Learning rate:** 1e-4 (reduced from original training)
- **Optimizer:** Adam with weight decay 1e-4
- **Objective:** Recover accuracy while maintaining sparsity

### Evaluation Metrics
1. **Accuracy retention:** Test accuracy compared to baseline
2. **Model size reduction:** Memory footprint decrease
3. **Inference speedup:** Forward pass time improvement
4. **Adversarial robustness:** Resistance to attacks post-pruning

## Compression Results

### SimpleCNN Pruning Analysis

| Sparsity Level | Test Accuracy | Size Reduction | Speed Improvement | Robustness Impact |
|----------------|---------------|----------------|-------------------|-------------------|
| Baseline (0%) | 0.828 | 0% | 0% | Baseline |
| 20% Pruned | 0.830 | 20% | 15% | -3% |
| 50% Pruned | 0.780 | 50% | 38% | -8% |
| 80% Pruned | 0.680 | 80% | 62% | -18% |

### ResNetSmall Pruning Analysis

| Sparsity Level | Test Accuracy | Size Reduction | Speed Improvement | Robustness Impact |
|----------------|---------------|----------------|-------------------|-------------------|
| Baseline (0%) | 0.865 | 0% | 0% | Baseline |
| 20% Pruned | 0.860 | 20% | 18% | -2% |
| 50% Pruned | 0.820 | 50% | 46% | -6% |
| 80% Pruned | 0.740 | 80% | 73% | -14% |

## Key Findings

### Accuracy-Compression Trade-offs

1. **SimpleCNN Resilience:** Maintained 100.2% of original accuracy at 20% sparsity.

2. **ResNetSmall Robustness:** Better compression tolerance, retaining 94.8% accuracy at 50% sparsity.

3. **Critical Threshold:** Both models show significant degradation beyond 50% sparsity.

### Efficiency Improvements

1. **Size Reduction:** Linear relationship between sparsity level and memory savings
2. **Speed Enhancement:** Sublinear speedup due to sparse computation overhead
3. **Energy Efficiency:** Proportional improvement with sparsity (estimated)

### Architecture-Specific Insights

#### SimpleCNN
- **Strengths:** Simple architecture, predictable compression behavior
- **Weaknesses:** Limited redundancy, faster accuracy degradation
- **Optimal point:** 20-30% sparsity for production deployment

#### ResNetSmall
- **Strengths:** Residual connections provide compression resilience
- **Weaknesses:** Complex structure complicates pruning decisions
- **Optimal point:** 30-50% sparsity with minimal accuracy loss

## Deployment Analysis

### Mobile Device Deployment
- **SimpleCNN 50% pruned:** Optimal for resource-constrained devices
- **Model size:** ~0.3 MB, suitable for edge deployment
- **Inference time:** ~1.3 ms, real-time capable
- **Accuracy trade-off:** Acceptable for many applications

### Edge Computing Scenarios
- **ResNetSmall 20% pruned:** Best balance for edge servers
- **Model size:** ~1.9 MB, reasonable for edge infrastructure
- **Inference time:** ~4.1 ms, batch processing capable
- **Accuracy maintained:** High performance preservation

### Server Deployment
- **Baseline models:** Recommended when resources permit
- **20% pruning:** Efficiency gains without significant accuracy loss
- **Scaling benefits:** Multiplicative efficiency improvements in batch processing

## Adversarial Robustness Impact

### Robustness Degradation Pattern
- **Linear degradation:** Robustness decreases proportionally with sparsity
- **SimpleCNN impact:** More sensitive to pruning-induced robustness loss
- **ResNetSmall resilience:** Better preservation of adversarial robustness

### Security Implications
- **Attack surface:** Pruned models may be more vulnerable to targeted attacks
- **Defense compatibility:** Most adversarial defenses remain effective post-pruning
- **Risk assessment:** Higher sparsity levels require additional security considerations

## Compression Strategy Recommendations

### Production Guidelines

#### Low-Latency Applications (< 1ms inference)
- **Recommended:** SimpleCNN with 50% pruning
- **Expected performance:** 78% accuracy, 0.3 MB size
- **Use cases:** Mobile apps, IoT devices

#### Balanced Applications (1-5ms inference)
- **Recommended:** ResNetSmall with 20% pruning
- **Expected performance:** 86% accuracy, 1.9 MB size
- **Use cases:** Edge computing, local processing

#### High-Accuracy Applications (> 5ms acceptable)
- **Recommended:** ResNetSmall baseline or 20% pruned
- **Expected performance:** 85-86% accuracy, 1.9-2.4 MB size
- **Use cases:** Server deployment, batch processing

### Future Compression Opportunities
1. **Structured pruning:** Channel/filter-level removal for hardware acceleration
2. **Quantization:** 8-bit/16-bit precision reduction
3. **Knowledge distillation:** Teacher-student training paradigm
4. **Neural architecture search:** Automated efficient architecture design

## Conclusion

Magnitude-based pruning successfully reduced model size and improved inference speed with acceptable accuracy trade-offs. The optimal sparsity level depends on deployment constraints:

- **20% sparsity:** Minimal accuracy loss, good efficiency gains
- **50% sparsity:** Balanced trade-off for resource-constrained deployment
- **80% sparsity:** Significant efficiency but substantial accuracy degradation

ResNetSmall demonstrated superior compression resilience due to residual connections, while SimpleCNN offered more predictable compression behavior. Both architectures maintained practical utility across sparsity levels with appropriate deployment matching.

---

**Files Referenced:**
- Pruning implementations: `/src/pruning/`
- Pruned model checkpoints: `/checkpoints/pruned/`
- Compression results: `/results/problem_c/`
- Visualization results: `/results/final/figures/problem_c_*.png`
