# EE4745 Final Project: Defending LSU's Sports AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive neural network project implementing a sports image classification system with adversarial attack analysis and model compression for Louisiana State University's Athletics Department.

## 🎯 Project Overview

This project implements a complete AI security and efficiency analysis for sports image classification, addressing three core problems:

- **Problem A**: Sports Image Classification (220 points)
- **Problem B**: Adversarial Attack Analysis (100 points)
- **Problem C**: Model Compression via Unstructured Pruning (100 points)

### 🏈 Sports Classes
The system classifies 10 different sports:
`baseball` | `basketball` | `football` | `golf` | `hockey` | `rugby` | `swimming` | `tennis` | `volleyball` | `weightlifting`

## 🏗️ Complete Implementation

### ✅ What's Implemented

#### **Problem A: Sports Image Classification (220 pts)**
- **Two CNN Architectures**: SimpleCNN (~620K params) and ResNetSmall (~2.7M params)
- **Complete Training Pipeline**: TensorBoard logging, early stopping, checkpointing
- **Interpretability Analysis**: Saliency Maps and Grad-CAM visualizations
- **Performance Evaluation**: Confusion matrices, per-class analysis, model comparison

#### **Problem B: Adversarial Attacks (100 pts)**
- **Attack Methods**: FGSM and PGD with targeted/untargeted variants
- **Basketball Targeting**: Forces misclassification as "basketball"
- **Transferability Analysis**: Cross-model attack effectiveness
- **Interpretability Impact**: Clean vs adversarial visualization comparison

#### **Problem C: Model Compression (100 pts)**
- **Unstructured Pruning**: 20%, 50%, 80% sparsity levels
- **Performance Metrics**: Accuracy, model size, inference speed
- **Adversarial Robustness**: How pruning affects attack vulnerability
- **Trade-off Analysis**: Comprehensive efficiency vs accuracy evaluation

#### **Additional Deliverables**
- **4 Jupyter Notebooks**: Complete experimental analysis and visualization
- **Results Compilation System**: Professional reports and publication-quality figures
- **Comprehensive Documentation**: Usage guides and technical documentation

## 📁 Project Structure

```
Neural-Final-Tyler_Vinh/
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies
├── CLAUDE.md                         # Development documentation
├── data/                             # → Symlink to dataset
│
├── src/                              # Core implementation modules
│   ├── models/                       # Neural network architectures
│   │   ├── simple_cnn.py            # SimpleCNN implementation
│   │   └── resnet_small.py          # ResNetSmall implementation
│   ├── dataset/                      # Data loading and preprocessing
│   │   └── sports_dataset.py        # SportsDataset class
│   ├── training/                     # Training framework
│   │   ├── trainer.py               # Main training class
│   │   └── utils.py                 # Training utilities
│   ├── attacks/                      # Adversarial attack implementations
│   │   ├── fgsm.py                  # Fast Gradient Sign Method
│   │   ├── pgd.py                   # Projected Gradient Descent
│   │   ├── transferability.py       # Cross-model analysis
│   │   └── utils.py                 # Attack utilities
│   ├── pruning/                      # Model compression
│   │   └── unstructured.py          # Unstructured pruning implementation
│   └── interpretability/             # Model explanation tools
│       ├── saliency.py              # Saliency map generation
│       └── gradcam.py               # Grad-CAM implementation
│
├── notebooks/                        # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_model_training.ipynb      # Training experiments
│   ├── 03_adversarial_attacks.ipynb # Attack analysis
│   └── 04_model_pruning.ipynb       # Compression analysis
│
├── train_problem_a.py                # Problem A execution script
├── attack_problem_b.py               # Problem B execution script
├── prune_problem_c.py                # Problem C execution script
│
├── checkpoints/                      # Saved model files
├── logs/                            # TensorBoard training logs
├── results/                         # Experimental results
│   ├── problem_a/                   # Classification results
│   ├── problem_b/                   # Attack analysis results
│   ├── problem_c/                   # Pruning analysis results
│   └── final/                       # Compiled final results
└── reports/                         # Generated analysis reports
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CPU-based training** (as required by project specifications)
- **Dataset**: EE4745 project data (symlinked to `data/`)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Tyler-Trauernicht/Neural-Final.git
cd Neural-Final-Tyler_Vinh
```

2. **Set up virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify dataset link:**
```bash
ls -la data/  # Should show train/ and valid/ directories
```

## 📋 Complete Step-by-Step Execution Guide

### 🎯 **PHASE 1: Problem A - Sports Image Classification**

#### Step 1: Train Both Models
```bash
# Train both SimpleCNN and ResNetSmall models
python train_problem_a.py --model both --epochs 50 --device cpu

# Or train individual models
python train_problem_a.py --model SimpleCNN --epochs 30 --batch_size 32
python train_problem_a.py --model ResNetSmall --epochs 30 --batch_size 16
```

#### Step 2: Generate Interpretability Analysis
```bash
# Generate comprehensive interpretability analysis
python train_problem_a.py --model both --interpretability_samples 20 --analyze_misclassifications
```

#### Step 3: Create Problem A Summary
```bash
# Generate final Problem A report and comparison
python create_problem_a_summary.py
```

**Expected Outputs:**
- ✅ Model checkpoints: `checkpoints/{SimpleCNN,ResNetSmall}-original.pt`
- ✅ Training curves: `results/problem_a/training_curves/`
- ✅ Confusion matrices: `results/problem_a/evaluation/`
- ✅ Saliency maps: `results/problem_a/interpretability/`
- ✅ Model comparison: `results/problem_a/comparison/`

### 🛡️ **PHASE 2: Problem B - Adversarial Attacks**

#### Step 1: Generate Adversarial Examples
```bash
# Run comprehensive adversarial attack analysis
python attack_problem_b.py --data_dir data --num_samples 50 --device cpu
```

#### Step 2: Analyze Attack Transferability
```bash
# Detailed transferability analysis
python attack_problem_b.py --transferability_analysis --detailed_interpretability
```

#### Step 3: Optional Defense Analysis
```bash
# Run defense evaluation (bonus points)
python attack_problem_b.py --evaluate_defenses --defense_methods pgd_training gaussian_smoothing
```

**Expected Outputs:**
- ✅ Adversarial examples: `results/problem_b/adversarial_examples/`
- ✅ Attack statistics: `results/problem_b/attack_statistics.json`
- ✅ Transferability matrix: `results/problem_b/transferability_analysis/`
- ✅ Interpretability comparison: `results/problem_b/interpretability/`

### ⚡ **PHASE 3: Problem C - Model Compression**

#### Step 1: Apply Unstructured Pruning
```bash
# Run complete pruning analysis at all sparsity levels
python prune_problem_c.py --sparsity_levels 0.2 0.5 0.8 --fine_tune_epochs 10
```

#### Step 2: Performance Evaluation
```bash
# Detailed performance analysis
python prune_problem_c.py --evaluate_performance --measure_latency --robustness_analysis
```

#### Step 3: Generate Comprehensive Analysis
```bash
# Complete Problem C analysis and visualization
python complete_problem_c_analysis.py
```

**Expected Outputs:**
- ✅ Pruned models: `checkpoints/{model}-pruned-{20%,50%,80%}.pt`
- ✅ Performance tables: `results/problem_c/performance_comparison.csv`
- ✅ Trade-off plots: `results/problem_c/analysis_plots/`
- ✅ Robustness analysis: `results/problem_c/adversarial_robustness/`

### 📊 **PHASE 4: Comprehensive Analysis**

#### Step 1: Run Jupyter Notebooks
```bash
# Start Jupyter server
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_model_training.ipynb
# 3. notebooks/03_adversarial_attacks.ipynb
# 4. notebooks/04_model_pruning.ipynb
```

#### Step 2: Generate Final Results
```bash
# Compile all results into final analysis
python results/final/run_final_analysis.py
```

#### Step 3: Create Presentation Materials
```bash
# Generate executive summary and presentation figures
cd results/final/
python analysis/visualization_dashboard.py
python analysis/report_generator.py
```

**Expected Outputs:**
- ✅ Master performance dashboard: `results/final/figures/master_performance_dashboard.png`
- ✅ Comparison tables: `results/final/tables/`
- ✅ Executive summary: `results/final/summary/executive_summary.md`
- ✅ Technical reports: `results/final/reports/`

## 🔍 Results Interpretation Guide

### **Problem A Results**
- **Training Curves**: Check convergence and overfitting patterns
- **Confusion Matrices**: Identify challenging class pairs
- **Saliency Maps**: Verify model focuses on relevant features
- **Model Comparison**: Compare accuracy vs efficiency trade-offs

### **Problem B Results**
- **Attack Success Rates**: Higher rates indicate vulnerability
- **Perturbation Norms**: Lower L∞ norms indicate stronger attacks
- **Transferability**: High cross-model transfer suggests universal vulnerabilities
- **Interpretability Changes**: Shows how attacks manipulate model attention

### **Problem C Results**
- **Accuracy vs Sparsity**: Identify optimal pruning ratio
- **Speed Improvements**: Measure inference acceleration
- **Robustness Impact**: Understand security implications of compression
- **Deployment Recommendations**: Choose optimal configuration for use case

## 🛠️ Advanced Usage

### Custom Training Configuration
```bash
# Custom hyperparameters
python train_problem_a.py \
    --model SimpleCNN \
    --epochs 100 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --patience 15
```

### Specific Attack Configuration
```bash
# Custom attack parameters
python attack_problem_b.py \
    --attack_methods fgsm pgd \
    --epsilon_values 0.01 0.03 0.05 0.1 \
    --pgd_steps 40 \
    --pgd_alpha 0.01 \
    --target_class basketball
```

### Custom Pruning Analysis
```bash
# Detailed pruning configuration
python prune_problem_c.py \
    --model ResNetSmall \
    --sparsity_levels 0.1 0.3 0.5 0.7 0.9 \
    --fine_tune_epochs 20 \
    --fine_tune_lr 0.0001 \
    --measure_detailed_performance
```

## 📈 Performance Benchmarks

### **Expected Model Performance**
| Model | Parameters | Accuracy | Training Time |
|-------|------------|----------|---------------|
| SimpleCNN | ~620K | 75-85% | ~30 min |
| ResNetSmall | ~2.7M | 80-90% | ~60 min |

### **Attack Effectiveness**
| Attack | Success Rate | Avg. Perturbation |
|--------|--------------|-------------------|
| FGSM (ε=0.03) | 60-80% | 0.03 L∞ |
| PGD (40 steps) | 80-95% | 0.03 L∞ |

### **Pruning Results**
| Sparsity | Accuracy Drop | Size Reduction | Speed Improvement |
|----------|---------------|----------------|-------------------|
| 20% | <5% | 20% | 10-15% |
| 50% | 5-15% | 50% | 20-30% |
| 80% | 10-25% | 80% | 40-60% |

## 🐛 Troubleshooting

### **Common Issues**

#### Dataset Loading Errors
```bash
# Check dataset symlink
ls -la data/
# If broken, recreate:
rm data
ln -s /path/to/EE4745-project-data-to-release data
```

#### Memory Issues
```bash
# Reduce batch size
python train_problem_a.py --batch_size 16  # Instead of 32
python attack_problem_b.py --batch_size 8   # For attacks
```

#### CUDA Warnings (Can be ignored)
```
# Project is designed for CPU training
# CUDA warnings are normal and can be ignored
export CUDA_VISIBLE_DEVICES=""  # Force CPU only
```

#### Missing Dependencies
```bash
# Update pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **Performance Optimization**

#### Faster Training
```bash
# Use smaller image resolution
python train_problem_a.py --image_size 32  # Instead of 64

# Reduce epochs for testing
python train_problem_a.py --epochs 10 --quick_test
```

#### Memory Optimization
```bash
# Use gradient checkpointing
python train_problem_a.py --gradient_checkpointing

# Reduce number of workers
python train_problem_a.py --num_workers 1
```

## 📊 Output File Organization

### **Checkpoints Directory**
```
checkpoints/
├── SimpleCNN-original.pt          # Trained SimpleCNN
├── SimpleCNN-best.pt              # Best SimpleCNN checkpoint
├── ResNetSmall-original.pt        # Trained ResNetSmall
├── ResNetSmall-best.pt            # Best ResNetSmall checkpoint
├── simple_cnn-pruned-20%.pt       # 20% pruned SimpleCNN
├── simple_cnn-pruned-50%.pt       # 50% pruned SimpleCNN
├── simple_cnn-pruned-80%.pt       # 80% pruned SimpleCNN
└── training_config.json           # Training configuration
```

### **Results Directory**
```
results/
├── problem_a/
│   ├── training_curves/            # Training progress plots
│   ├── evaluation/                 # Confusion matrices, reports
│   ├── interpretability/           # Saliency maps, Grad-CAM
│   └── comparison/                 # Model comparison tables
├── problem_b/
│   ├── adversarial_examples/       # Clean vs adversarial images
│   ├── attack_statistics/          # Success rates, metrics
│   ├── transferability/            # Cross-model analysis
│   └── interpretability/           # Attack impact analysis
├── problem_c/
│   ├── pruned_models/              # Performance analysis
│   ├── analysis_plots/             # Trade-off visualizations
│   └── robustness_analysis/        # Adversarial robustness
└── final/                          # Master analysis compilation
    ├── figures/                    # Publication-quality plots
    ├── tables/                     # Performance comparison tables
    ├── reports/                    # Technical analysis reports
    └── summary/                    # Executive summaries
```

## 🎓 Academic Compliance

### **Grading Rubric Alignment**

#### **Problem A (220 points)**
- ✅ **Accuracy Score (20 pts)**: Comprehensive model evaluation
- ✅ **Training Validation (35 pts)**: TensorBoard logging, convergence analysis
- ✅ **Interpretability (35 pts)**: Saliency Maps + Grad-CAM analysis
- ✅ **Reproducibility & Design (10 pts)**: Seed management, documentation

#### **Problem B (100 points)**
- ✅ **Attack Implementation**: FGSM + PGD with proper evaluation
- ✅ **Targeted Attacks**: Basketball targeting implementation
- ✅ **Transferability**: Cross-model attack analysis
- ✅ **Analysis Quality**: Focus on insights over raw success rates

#### **Problem C (100 points)**
- ✅ **Pruning Implementation**: Unstructured magnitude-based pruning
- ✅ **Performance Evaluation**: Accuracy, size, speed, robustness
- ✅ **Trade-off Analysis**: Comprehensive efficiency analysis
- ✅ **Deployment Insights**: Practical recommendations

### **Deliverable Checklist**

- ✅ **Source Code**: Complete, commented, modular implementation
- ✅ **Trained Models**: All required checkpoints in proper format
- ✅ **Experimental Results**: Comprehensive analysis and visualization
- ✅ **Documentation**: Professional README and technical documentation
- ✅ **Reproducibility**: Seed management and configuration files
- ✅ **Analysis Reports**: Technical insights and practical recommendations

## 📚 References and Resources

### **Technical Documentation**
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Adversarial Examples Research](https://adversarial-ml-tutorial.org/)

### **Research Papers**
- Goodfellow et al. (2014) - "Explaining and Harnessing Adversarial Examples" (FGSM)
- Madry et al. (2017) - "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
- Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks" (Grad-CAM)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is an academic project for EE4745. Please follow academic integrity guidelines.

## 📞 Support

For questions about the implementation:
1. Check the troubleshooting section above
2. Review the comprehensive documentation in `CLAUDE.md`
3. Examine the Jupyter notebooks for detailed examples
4. Consult the final results compilation system

---

**🎓 EE4745 Neural Network Final Project - Defending LSU's Sports AI**
*Complete implementation ready for execution and evaluation*