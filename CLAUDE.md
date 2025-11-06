# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EE4745 Final Project: "Defending LSU's Sports AI"**

This is a comprehensive neural network project implementing a sports image classification system with adversarial attack analysis and model compression. The project is connected to: https://github.com/Tyler-Trauernicht/Neural-Final.git

### Project Components
- **Problem A**: Sports Image Classification (10 classes: baseball, basketball, football, golf, hockey, rugby, swimming, tennis, volleyball, weightlifting)
- **Problem B**: Adversarial Attack Analysis (targeted/untargeted attacks, transferability)
- **Problem C**: Model Compression (unstructured pruning at 20%, 50%, 80% sparsity)

## Development Setup

### Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset
- Data is symlinked from `/Users/ty/Downloads/EE4745-project-data-to-release/`
- Training: ~1,593 images (131-191 per class)
- Validation: 50 images (5 per class)
- Images should be resized to 32x32 or 64x64 pixels

## Project Structure

```
src/
├── models/          # Model architectures (SimpleCNN, ResNetSmall)
├── dataset/         # Data loading and preprocessing
├── training/        # Training pipeline and utilities
├── attacks/         # Adversarial attack implementations
├── pruning/         # Model compression utilities
└── interpretability/ # Saliency maps and Grad-CAM

notebooks/           # Jupyter notebooks for experiments
checkpoints/         # Saved model checkpoints
logs/               # TensorBoard logs
results/            # Output figures and tables
```

## Common Commands

### Training Models
```bash
# Train SimpleCNN
python -m src.models.simple_cnn

# Train ResNet-Small
python -m src.models.resnet_small

# Test dataset loading
python -m src.dataset.sports_dataset
```

### Development Workflow
```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Start TensorBoard
tensorboard --logdir=logs

# Jupyter notebooks
jupyter notebook notebooks/
```

### Model Management
```bash
# Save checkpoint
torch.save(model.state_dict(), 'checkpoints/model-name-original.pt')

# Load checkpoint
model.load_state_dict(torch.load('checkpoints/model-name-original.pt'))
```

### Git Commands
- `git add .` - Stage all changes
- `git commit -m "message"` - Commit changes
- `git push origin main` - Push to GitHub repository
- `git pull origin main` - Pull latest changes from GitHub

## Architecture Notes

### Model Architectures

**SimpleCNN**
- 3 Conv blocks (32→64→128 channels)
- BatchNorm + ReLU + MaxPool/AdaptiveAvgPool
- 2-layer MLP classifier with dropout
- ~147K parameters (32x32 input)

**ResNetSmall**
- Initial Conv(3→64) + 3 ResNet layers
- Layer1: 64 channels, Layer2: 128 channels, Layer3: 256 channels
- Global average pooling + linear classifier
- ~600K parameters (32x32 input)

### Training Configuration
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Early stopping with patience=10

### Interpretability
- **Saliency Maps**: Pixel-level gradient visualization
- **Grad-CAM**: Feature-level attention heatmaps
- Target layers: SimpleCNN (features[-2]), ResNet (layer3[-1])

## Attack Methods

### FGSM (Fast Gradient Sign Method)
- Single-step attack using sign of gradients
- Epsilon values: [0.01, 0.03, 0.05, 0.1]

### PGD (Projected Gradient Descent)
- Iterative attack with projection
- Parameters: alpha=0.01, steps=40, epsilon=0.03

### Targeting
- **Untargeted**: Cause any misclassification
- **Targeted**: Force classification as "basketball"

## Model Compression

### Unstructured Pruning
- Magnitude-based pruning of Conv2d and Linear layers
- Sparsity levels: 20%, 50%, 80%
- Fine-tuning: 10 epochs with lr=1e-4
- Evaluation: accuracy, model size, inference speed, adversarial robustness

## File Naming Conventions

### Model Checkpoints
- Original models: `{model-name}-original.pt`
- Best models: `{model-name}-best.pt`
- Pruned models: `{model-name}-pruned-{ratio}.pt`

### Results
- Saliency maps: `saliency_sample_{idx:03d}.png`
- Grad-CAM: `gradcam_sample_{idx:03d}.png`
- Misclassifications: `misclassified_{true}_as_{pred}_{idx:03d}.png`

## Important Notes

- All models must be CPU-trainable for this project
- Use fixed seeds for reproducibility (seed=42)
- Save experimental settings and hyperparameters
- Generate visualizations for at least 3 classes (correct + misclassified examples)
- Document all experimental results with timestamps
- Focus on analysis quality over raw performance metrics