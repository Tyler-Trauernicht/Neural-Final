#!/usr/bin/env python3
"""
Training script for Problem A: Sports Image Classification
EE4745 Neural Network Final Project

This script trains SimpleCNN and ResNetSmall models on the sports dataset
and performs comprehensive evaluation and interpretability analysis.
"""

import os
import argparse
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Import project modules
from src.dataset.sports_dataset import get_dataloaders, SportsDataset
from src.models.simple_cnn import create_simple_cnn
from src.models.resnet_small import create_resnet_small
from src.training.trainer import Trainer
from src.training.utils import set_seed, save_checkpoint, count_parameters
from src.interpretability.saliency import SaliencyMap
from src.interpretability.gradcam import GradCAM, get_target_layer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train sports classification models for Problem A')

    # Model selection
    parser.add_argument('--model', type=str, default='both',
                       choices=['SimpleCNN', 'ResNetSmall', 'both'],
                       help='Model to train (default: both)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory (default: data)')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Input image size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers (default: 2)')

    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results/problem_a',
                       help='Results directory (default: results/problem_a)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='TensorBoard log directory (default: logs)')

    # Analysis parameters
    parser.add_argument('--interpretability_samples', type=int, default=20,
                       help='Number of samples for interpretability analysis (default: 20)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: cpu)')

    # Control flags
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only run analysis')
    parser.add_argument('--skip_interpretability', action='store_true',
                       help='Skip interpretability analysis')

    return parser.parse_args()


def create_training_config(args, model_name):
    """Create training configuration for a specific model"""
    return {
        'model_name': model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'experiment_name': f'{model_name}_problem_a',
        'use_tensorboard': True,
        'patience': 15,
        'min_delta': 0.001
    }


def plot_training_curves(history, model_name, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Training and validation loss
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training and validation accuracy
    axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, data_loader, device, class_names):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets)) * 100

    # Classification report
    report = classification_report(all_targets, all_preds,
                                 target_names=class_names,
                                 output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(report, model_name, save_dir):
    """Save classification report to text file"""
    report_path = os.path.join(save_dir, f'{model_name}_classification_report.txt')

    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")

        # Per-class metrics
        f.write("Per-class Performance:\n")
        f.write("-" * 30 + "\n")
        for class_name in SportsDataset.CLASSES:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                f.write(f"{class_name:12}: P={precision:.3f}, R={recall:.3f}, "
                       f"F1={f1:.3f}, N={support}\n")

        # Overall metrics
        f.write(f"\nOverall Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n")


def run_interpretability_analysis(model, model_name, data_loader, device,
                                class_names, save_dir, num_samples=20):
    """Run comprehensive interpretability analysis"""
    print(f"\nRunning interpretability analysis for {model_name}...")

    # Create subdirectories for interpretability results
    saliency_dir = os.path.join(save_dir, f'{model_name}_saliency')
    gradcam_dir = os.path.join(save_dir, f'{model_name}_gradcam')
    os.makedirs(saliency_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    # Initialize interpretability tools
    saliency = SaliencyMap(model, device=device)

    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer, device=device)

    # Generate saliency maps for random samples
    print("Generating saliency maps...")
    saliency_results = saliency.generate_batch(
        data_loader,
        num_samples=num_samples,
        class_names=class_names,
        save_dir=saliency_dir
    )

    # Generate Grad-CAM heatmaps
    print("Generating Grad-CAM heatmaps...")
    gradcam_results = gradcam.generate_batch(
        data_loader,
        num_samples=num_samples,
        class_names=class_names,
        save_dir=gradcam_dir
    )

    # Analyze misclassifications
    print("Analyzing misclassified samples...")
    misclassified_saliency = saliency.analyze_misclassifications(
        data_loader,
        class_names=class_names,
        save_dir=os.path.join(saliency_dir, 'misclassified')
    )

    # Generate class comparison for Grad-CAM
    print("Generating class comparison visualizations...")
    # Get a sample from each class for comparison
    class_samples = {}
    for images, labels in data_loader:
        for i, label in enumerate(labels):
            class_idx = label.item()
            class_name = class_names[class_idx]
            if class_name not in class_samples:
                class_samples[class_name] = images[i:i+1]
                if len(class_samples) >= 3:  # Limit to first 3 classes for comparison
                    break
        if len(class_samples) >= 3:
            break

    # Create comparison visualizations
    for i, (class_name, sample) in enumerate(list(class_samples.items())[:3]):
        # Compare with 3 different target classes
        target_classes = [0, 1, 2] if len(class_names) >= 3 else list(range(len(class_names)))
        comparison_path = os.path.join(gradcam_dir, f'class_comparison_{class_name}.png')
        fig = gradcam.compare_classes(sample, target_classes, class_names, comparison_path)
        plt.close(fig)

    return {
        'saliency_results': saliency_results,
        'gradcam_results': gradcam_results,
        'misclassified_count': len(misclassified_saliency)
    }


def train_model(model_name, args):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Create model
    if model_name == 'SimpleCNN':
        model = create_simple_cnn(num_classes=10, input_size=args.image_size)
    elif model_name == 'ResNetSmall':
        model = create_resnet_small(num_classes=10, input_size=args.image_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Get data loaders
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # Create training configuration
    config = create_training_config(args, model_name)

    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config)

    if not args.skip_training:
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save training curves
        results_dir = os.path.join(args.results_dir, 'training_curves')
        os.makedirs(results_dir, exist_ok=True)
        plot_training_curves(history, model_name, results_dir)

        # Save the trained model as {model-name}-original.pt
        model_checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}-original.pt')
        save_checkpoint(model, trainer.optimizer, trainer.scheduler,
                       config['epochs'], 0, 0, model_checkpoint_path)

    else:
        # Load existing model if skipping training
        model_checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}-original.pt')
        if os.path.exists(model_checkpoint_path):
            checkpoint = torch.load(model_checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded existing model from {model_checkpoint_path}")
        else:
            print(f"Warning: No existing model found at {model_checkpoint_path}")
            return None

        training_time = 0  # Unknown if model was loaded
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Evaluate model
    print(f"\nEvaluating {model_name}...")
    model.to(args.device)
    evaluation = evaluate_model(model, val_loader, args.device, SportsDataset.CLASSES)

    # Save evaluation results
    eval_dir = os.path.join(args.results_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    plot_confusion_matrix(evaluation['confusion_matrix'], SportsDataset.CLASSES,
                         model_name, eval_dir)
    save_classification_report(evaluation['classification_report'], model_name, eval_dir)

    # Run interpretability analysis
    interpretability_results = None
    if not args.skip_interpretability:
        interp_dir = os.path.join(args.results_dir, 'interpretability')
        os.makedirs(interp_dir, exist_ok=True)
        interpretability_results = run_interpretability_analysis(
            model, model_name, val_loader, args.device,
            SportsDataset.CLASSES, interp_dir, args.interpretability_samples
        )

    return {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time': training_time,
        'accuracy': evaluation['accuracy'],
        'classification_report': evaluation['classification_report'],
        'confusion_matrix': evaluation['confusion_matrix'],
        'interpretability_results': interpretability_results,
        'history': history
    }


def create_model_comparison(results_list, save_dir):
    """Create model comparison table and visualizations"""
    print("\nCreating model comparison...")

    # Create comparison dataframe
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            'Model': result['model_name'],
            'Total Parameters': result['total_params'],
            'Trainable Parameters': result['trainable_params'],
            'Training Time (s)': result['training_time'],
            'Validation Accuracy (%)': result['accuracy'],
            'Macro F1': result['classification_report']['macro avg']['f1-score'],
            'Weighted F1': result['classification_report']['weighted avg']['f1-score']
        })

    df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_path = os.path.join(save_dir, 'model_comparison.csv')
    df.to_csv(comparison_path, index=False)

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy comparison
    axes[0, 0].bar(df['Model'], df['Validation Accuracy (%)'])
    axes[0, 0].set_title('Validation Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Parameters comparison
    axes[0, 1].bar(df['Model'], df['Total Parameters'])
    axes[0, 1].set_title('Model Size Comparison')
    axes[0, 1].set_ylabel('Total Parameters')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Training time comparison (if available)
    if df['Training Time (s)'].sum() > 0:
        axes[1, 0].bar(df['Model'], df['Training Time (s)'])
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'Training time not available\n(models were loaded)',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Time Comparison')

    # F1 scores comparison
    x = np.arange(len(df))
    width = 0.35
    axes[1, 1].bar(x - width/2, df['Macro F1'], width, label='Macro F1')
    axes[1, 1].bar(x + width/2, df['Weighted F1'], width, label='Weighted F1')
    axes[1, 1].set_title('F1 Scores Comparison')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df['Model'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_plots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed comparison text
    comparison_text_path = os.path.join(save_dir, 'model_comparison.txt')
    with open(comparison_text_path, 'w') as f:
        f.write("Model Comparison Report\n")
        f.write("=" * 50 + "\n\n")

        for result in results_list:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  Total Parameters: {result['total_params']:,}\n")
            f.write(f"  Trainable Parameters: {result['trainable_params']:,}\n")
            f.write(f"  Training Time: {result['training_time']:.2f} seconds\n")
            f.write(f"  Validation Accuracy: {result['accuracy']:.2f}%\n")
            f.write(f"  Macro F1: {result['classification_report']['macro avg']['f1-score']:.4f}\n")
            f.write(f"  Weighted F1: {result['classification_report']['weighted avg']['f1-score']:.4f}\n")
            if result['interpretability_results']:
                f.write(f"  Misclassified samples analyzed: {result['interpretability_results']['misclassified_count']}\n")
            f.write("\n")

    print(f"Model comparison saved to {save_dir}")
    print(df.to_string(index=False))


def main():
    """Main training and evaluation pipeline"""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("EE4745 Neural Network Final Project - Problem A")
    print("Sports Image Classification")
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")

    # Determine which models to train
    if args.model == 'both':
        models_to_train = ['SimpleCNN', 'ResNetSmall']
    else:
        models_to_train = [args.model]

    # Train models
    results = []
    for model_name in models_to_train:
        result = train_model(model_name, args)
        if result is not None:
            results.append(result)

    # Create model comparison if multiple models were trained
    if len(results) > 1:
        comparison_dir = os.path.join(args.results_dir, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        create_model_comparison(results, comparison_dir)

    # Save experiment configuration
    config_path = os.path.join(args.results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print("Problem A Implementation Completed!")
    print(f"{'='*60}")
    print(f"Results saved in: {args.results_dir}")
    print(f"Model checkpoints saved in: {args.checkpoint_dir}")
    print("\nSummary:")
    for result in results:
        print(f"  {result['model_name']}: {result['accuracy']:.2f}% accuracy "
              f"({result['total_params']:,} parameters)")


if __name__ == "__main__":
    main()