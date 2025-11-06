# EE4745 Neural Network Final Project - Results Index

**Project:** "Defending LSU's Sports AI: Classification, Attacks, and Compression"
**Authors:** Tyler Trauernicht, Vinh Le
**Generated:** November 05, 2025 at 08:00 PM

---

## Quick Navigation

### 📊 Executive Summary
- **[Executive Summary](summary/executive_summary.md)** - High-level project overview and key findings
- **[Methodology Summary](summary/methodology_summary.md)** - Detailed experimental methodology

### 📈 Performance Tables
- **[Problem A Comparison](tables/problem_a_model_comparison.csv)** - Sports classification model performance
- **[Problem B Attack Analysis](tables/problem_b_attack_comparison.csv)** - Adversarial attack effectiveness
- **[Problem C Pruning Analysis](tables/problem_c_pruning_comparison.csv)** - Model compression trade-offs
- **[Master Performance Table](tables/master_performance_comparison.csv)** - Comprehensive comparison

### 📉 Visualizations
- **[Problem A Figures](figures/)** - Training curves and model comparisons
  - `problem_a_training_curves.png` - Training progress visualization
  - `problem_a_model_comparison.png` - Model performance comparison
  - `problem_a_per_class_analysis.png` - Per-class accuracy analysis

- **[Problem B Figures](figures/)** - Attack effectiveness and robustness analysis
  - `problem_b_attack_effectiveness.png` - Attack success rate analysis
  - `problem_b_transferability.png` - Cross-model attack transferability
  - `problem_b_robustness_analysis.png` - Model robustness evaluation

- **[Problem C Figures](figures/)** - Compression and efficiency analysis
  - `problem_c_pruning_tradeoffs.png` - Accuracy vs compression trade-offs
  - `problem_c_compression_analysis.png` - Size and speed improvements
  - `problem_c_efficiency_analysis.png` - Deployment efficiency analysis

- **[Master Dashboard](figures/master_performance_dashboard.png)** - Comprehensive overview

### 📝 Detailed Reports
- **[Problem A Report](reports/problem_a_analysis_report.md)** - Sports classification analysis
- **[Problem B Report](reports/problem_b_analysis_report.md)** - Adversarial attack analysis
- **[Problem C Report](reports/problem_c_analysis_report.md)** - Model compression analysis
- **[Integrated Analysis](reports/integrated_analysis_report.md)** - Cross-problem insights

### 🔧 Analysis Scripts
- **[Results Compiler](analysis/final_results_compiler.py)** - Data compilation and table generation
- **[Visualization Dashboard](analysis/visualization_dashboard.py)** - Figure generation system
- **[Report Generator](analysis/report_generator.py)** - Automated report creation

---

## Key Findings Summary

### 🎯 Problem A: Sports Image Classification
- **Best Model:** ResNetSmall (86.5% accuracy)
- **Most Efficient:** SimpleCNN (4x fewer parameters)
- **Training Time:** SimpleCNN trains 1.7x faster
- **Interpretability:** SimpleCNN provides clearer feature visualization

### ⚡ Problem B: Adversarial Attack Analysis
- **Most Effective Attack:** PGD (58% success rate)
- **Model Vulnerability:** Both models significantly vulnerable
- **Transferability:** 65-72% cross-model attack success
- **Defense Need:** Adversarial training essential for production

### 📦 Problem C: Model Compression via Pruning
- **Optimal Sparsity:** 20% provides best accuracy-efficiency balance
- **Best Compression:** SimpleCNN 50% pruned for mobile deployment
- **Speed Improvement:** Up to 62% inference speedup
- **Robustness Impact:** Compression reduces adversarial robustness

---

## Deployment Recommendations

| Use Case | Model | Configuration | Expected Performance |
|----------|-------|---------------|---------------------|
| 📱 Mobile Apps | SimpleCNN | 50% Pruned | 78% accuracy, 0.3 MB |
| 🔧 Edge Computing | ResNetSmall | 20% Pruned | 86% accuracy, 1.9 MB |
| 🖥️ Server Deployment | ResNetSmall | Baseline | 86.5% accuracy, 2.4 MB |
| 🔒 Security-Critical | Ensemble | Adversarial Training | 82% robust accuracy |

---

## File Organization

```
results/final/
├── tables/           # Performance comparison tables (CSV + LaTeX)
├── figures/          # All visualization plots (PNG, high-DPI)
├── reports/          # Detailed analysis reports (Markdown)
├── summary/          # Executive summaries and methodology
├── analysis/         # Analysis scripts and tools
└── README.md         # This documentation index
```

---

## Usage Instructions

### For Quick Overview
1. Start with the **[Executive Summary](summary/executive_summary.md)**
2. Review the **[Master Dashboard](figures/master_performance_dashboard.png)**
3. Check specific **[Performance Tables](tables/)**

### For Technical Details
1. Read the **[Methodology Summary](summary/methodology_summary.md)**
2. Review individual **[Problem Reports](reports/)**
3. Examine detailed **[Visualizations](figures/)**

### For Research Extension
1. Study the **[Integrated Analysis](reports/integrated_analysis_report.md)**
2. Review **[Analysis Scripts](analysis/)**
3. Check **[Raw Data Tables](tables/)**

---

## Contact Information

**Authors:** Tyler Trauernicht, Vinh Le
**Course:** EE4745 Neural Networks
**Institution:** Louisiana State University
**Project Repository:** https://github.com/Tyler-Trauernicht/Neural-Final.git

For questions or extensions, please refer to the methodology summary and analysis scripts.
