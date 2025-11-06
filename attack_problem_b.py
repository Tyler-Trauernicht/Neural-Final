#!/usr/bin/env python3
"""
Problem B: Adversarial Attacks Implementation

This script implements comprehensive adversarial attacks evaluation including:
1. FGSM and PGD attacks (targeted and untargeted)
2. Transferability analysis across models
3. Interpretability analysis of adversarial examples
4. Generation of 40+ adversarial examples with detailed analysis

Author: Claude Code
Course: EE4745 Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime

# Project imports
from src.dataset.sports_dataset import SportsDataset, get_dataloaders
from src.models.simple_cnn import create_simple_cnn
from src.models.resnet_small import create_resnet_small
from src.attacks.fgsm import FGSM
from src.attacks.pgd import PGD
from src.attacks.transferability import TransferabilityAnalyzer
from src.attacks.utils import (
    denormalize_image, calculate_perturbation_metrics,
    evaluate_attack_success, visualize_adversarial_examples,
    save_adversarial_examples, load_model_checkpoint
)
from src.interpretability.saliency import SaliencyMap
from src.interpretability.gradcam import GradCAM, get_target_layer


def load_trained_models(device='cpu'):
    """
    Load trained models from Problem A checkpoints.

    Args:
        device: Device to load models on

    Returns:
        Dictionary of loaded models
    """
    models = {}

    # Model configurations
    model_configs = {
        'SimpleCNN': {
            'constructor': create_simple_cnn,
            'checkpoint': 'checkpoints/simple_cnn-original.pt',
            'params': {'num_classes': 10, 'input_size': 32}
        },
        'ResNetSmall': {
            'constructor': create_resnet_small,
            'checkpoint': 'checkpoints/resnet_small-original.pt',
            'params': {'num_classes': 10, 'input_size': 32}
        }
    }

    for model_name, config in model_configs.items():
        checkpoint_path = config['checkpoint']

        if os.path.exists(checkpoint_path):
            print(f"Loading {model_name} from {checkpoint_path}")
            try:
                model = load_model_checkpoint(
                    config['constructor'],
                    checkpoint_path,
                    device
                )
                models[model_name] = model
                print(f"  Successfully loaded {model_name}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                # Create untrained model as fallback
                print(f"  Creating untrained {model_name} as fallback")
                model = config['constructor'](**config['params'])
                model.to(device)
                model.eval()
                models[model_name] = model
        else:
            print(f"Checkpoint not found for {model_name}: {checkpoint_path}")
            print(f"Creating untrained {model_name}")
            model = config['constructor'](**config['params'])
            model.to(device)
            model.eval()
            models[model_name] = model

    return models


def get_test_data(data_dir='data', num_samples=50, batch_size=10):
    """
    Get test dataset for adversarial attacks.

    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to use for testing
        batch_size: Batch size for data loader

    Returns:
        DataLoader for test data
    """
    # Create test dataset
    test_dataset = SportsDataset(
        root_dir=data_dir,
        split='valid',  # Use validation set as test set
        image_size=32,
        augment=False
    )

    print(f"Total test samples available: {len(test_dataset)}")

    # Select subset if needed
    if num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_dataset = Subset(test_dataset, indices)

    print(f"Using {len(test_dataset)} test samples")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return test_loader


def generate_adversarial_examples(models, test_loader, results_dir, device='cpu'):
    """
    Generate adversarial examples using FGSM and PGD attacks.

    Args:
        models: Dictionary of trained models
        test_loader: Test data loader
        results_dir: Directory to save results
        device: Device to run on

    Returns:
        Dictionary containing attack results
    """
    # Basketball class index (our target class)
    BASKETBALL_CLASS = SportsDataset.CLASSES.index('basketball')
    print(f"Basketball class index: {BASKETBALL_CLASS}")

    # Attack parameters
    fgsm_epsilons = [0.01, 0.03, 0.05, 0.1]
    pgd_params = {'epsilon': 0.03, 'alpha': 0.01, 'steps': 40}

    all_results = {}

    # Get a batch of test data
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    print(f"Test batch shape: {test_inputs.shape}")
    print(f"Test labels: {test_labels.cpu().numpy()}")

    # For each model
    for model_name, model in models.items():
        print(f"\nGenerating attacks for {model_name}")

        model_results = {}
        model_dir = os.path.join(results_dir, 'adversarial_examples', model_name)
        os.makedirs(model_dir, exist_ok=True)

        # FGSM attacks
        fgsm_attacker = FGSM(model, device)

        # FGSM Untargeted
        print("  Running FGSM untargeted attacks...")
        for epsilon in fgsm_epsilons:
            adv_inputs, attack_info = fgsm_attacker.attack(
                test_inputs, test_labels, epsilon=epsilon, targeted=False
            )

            attack_name = f'fgsm_untargeted_eps_{epsilon}'
            model_results[attack_name] = attack_info

            # Save examples
            example_dir = os.path.join(model_dir, attack_name)
            os.makedirs(example_dir, exist_ok=True)

            save_adversarial_examples(
                test_inputs, adv_inputs, attack_info,
                example_dir, f'fgsm_untargeted_eps_{epsilon}'
            )

            print(f"    Epsilon {epsilon}: Success rate {attack_info['success_rate']:.3f}")

        # FGSM Targeted (basketball)
        print("  Running FGSM targeted attacks...")
        # Filter samples not already basketball
        non_basketball_mask = test_labels != BASKETBALL_CLASS
        if non_basketball_mask.sum() > 0:
            filtered_inputs = test_inputs[non_basketball_mask]
            filtered_labels = test_labels[non_basketball_mask]
            target_labels = torch.full_like(filtered_labels, BASKETBALL_CLASS)

            # Take up to 10 examples for targeted attack
            n_targeted = min(10, len(filtered_inputs))
            targeted_inputs = filtered_inputs[:n_targeted]
            targeted_true_labels = filtered_labels[:n_targeted]
            targeted_target_labels = target_labels[:n_targeted]

            for epsilon in fgsm_epsilons:
                adv_inputs, attack_info = fgsm_attacker.attack(
                    targeted_inputs, targeted_true_labels,
                    epsilon=epsilon, targeted=True, target_labels=targeted_target_labels
                )

                attack_name = f'fgsm_targeted_eps_{epsilon}'
                model_results[attack_name] = attack_info

                # Save examples
                example_dir = os.path.join(model_dir, attack_name)
                os.makedirs(example_dir, exist_ok=True)

                save_adversarial_examples(
                    targeted_inputs, adv_inputs, attack_info,
                    example_dir, f'fgsm_targeted_eps_{epsilon}'
                )

                print(f"    Epsilon {epsilon}: Success rate {attack_info['success_rate']:.3f}")

        # PGD attacks
        pgd_attacker = PGD(model, device)

        # PGD Untargeted
        print("  Running PGD untargeted attack...")
        adv_inputs, attack_info = pgd_attacker.attack(
            test_inputs, test_labels,
            epsilon=pgd_params['epsilon'],
            alpha=pgd_params['alpha'],
            steps=pgd_params['steps'],
            targeted=False
        )

        attack_name = 'pgd_untargeted'
        model_results[attack_name] = attack_info

        # Save examples
        example_dir = os.path.join(model_dir, attack_name)
        os.makedirs(example_dir, exist_ok=True)

        save_adversarial_examples(
            test_inputs, adv_inputs, attack_info,
            example_dir, 'pgd_untargeted'
        )

        print(f"    Success rate: {attack_info['success_rate']:.3f}")

        # PGD Targeted (basketball)
        print("  Running PGD targeted attack...")
        if non_basketball_mask.sum() > 0:
            adv_inputs, attack_info = pgd_attacker.attack(
                targeted_inputs, targeted_true_labels,
                epsilon=pgd_params['epsilon'],
                alpha=pgd_params['alpha'],
                steps=pgd_params['steps'],
                targeted=True,
                target_labels=targeted_target_labels
            )

            attack_name = 'pgd_targeted'
            model_results[attack_name] = attack_info

            # Save examples
            example_dir = os.path.join(model_dir, attack_name)
            os.makedirs(example_dir, exist_ok=True)

            save_adversarial_examples(
                targeted_inputs, adv_inputs, attack_info,
                example_dir, 'pgd_targeted'
            )

            print(f"    Success rate: {attack_info['success_rate']:.3f}")

        all_results[model_name] = model_results

    return all_results


def analyze_transferability(models, test_loader, results_dir, device='cpu'):
    """
    Analyze adversarial transferability between models.

    Args:
        models: Dictionary of trained models
        test_loader: Test data loader
        results_dir: Directory to save results
        device: Device to run on

    Returns:
        Transferability analysis results
    """
    print("\nAnalyzing adversarial transferability...")

    # Basketball class index
    BASKETBALL_CLASS = SportsDataset.CLASSES.index('basketball')

    # Get test data
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    # Attack parameters for transferability analysis
    attack_params = {
        'fgsm': {'epsilon': 0.03},
        'pgd': {'epsilon': 0.03, 'alpha': 0.01, 'steps': 40}
    }

    # Create transferability analyzer
    analyzer = TransferabilityAnalyzer(models, device)

    # Run transferability analysis
    transferability_results = analyzer.analyze_transferability(
        test_inputs, test_labels, attack_params, target_class=BASKETBALL_CLASS
    )

    # Save transferability results
    transfer_dir = os.path.join(results_dir, 'transferability')
    analyzer.save_results(transferability_results, transfer_dir)

    # Calculate transferability metrics
    transfer_metrics = analyzer.calculate_transferability_metrics(transferability_results)

    print("Transferability Analysis Results:")
    print(f"  Overall mean success rate: {transfer_metrics.get('overall_mean_success_rate', 0):.3f}")
    print(f"  Cross-model mean success rate: {transfer_metrics.get('cross_model_mean_success_rate', 0):.3f}")
    print(f"  Same-model mean success rate: {transfer_metrics.get('same_model_mean_success_rate', 0):.3f}")
    print(f"  Transferability ratio: {transfer_metrics.get('transferability_ratio', 0):.3f}")

    return transferability_results, transfer_metrics


def analyze_interpretability(models, test_loader, results_dir, device='cpu'):
    """
    Analyze interpretability of adversarial examples using saliency maps and Grad-CAM.

    Args:
        models: Dictionary of trained models
        test_loader: Test data loader
        results_dir: Directory to save results
        device: Device to run on
    """
    print("\nAnalyzing interpretability of adversarial examples...")

    # Get test data
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)

    # Take first few samples for detailed analysis
    num_samples = min(5, len(test_inputs))
    sample_inputs = test_inputs[:num_samples]
    sample_labels = test_labels[:num_samples]

    class_names = SportsDataset.CLASSES

    for model_name, model in models.items():
        print(f"  Analyzing interpretability for {model_name}")

        model_interp_dir = os.path.join(results_dir, 'interpretability', model_name)
        os.makedirs(model_interp_dir, exist_ok=True)

        # Generate adversarial examples for interpretation
        fgsm_attacker = FGSM(model, device)
        adv_inputs, _ = fgsm_attacker.attack(
            sample_inputs, sample_labels, epsilon=0.03, targeted=False
        )

        # Saliency Maps
        print(f"    Generating saliency maps...")
        saliency_analyzer = SaliencyMap(model, device)

        for i in range(num_samples):
            # Original image saliency
            orig_sample = sample_inputs[i:i+1]
            orig_label = sample_labels[i].item()

            saliency_dir = os.path.join(model_interp_dir, 'saliency')
            os.makedirs(saliency_dir, exist_ok=True)

            fig_orig = saliency_analyzer.visualize(
                orig_sample,
                target_class=None,
                class_names=class_names,
                save_path=os.path.join(saliency_dir, f'original_sample_{i}.png')
            )
            plt.close(fig_orig)

            # Adversarial image saliency
            adv_sample = adv_inputs[i:i+1]
            fig_adv = saliency_analyzer.visualize(
                adv_sample,
                target_class=None,
                class_names=class_names,
                save_path=os.path.join(saliency_dir, f'adversarial_sample_{i}.png')
            )
            plt.close(fig_adv)

        # Grad-CAM analysis
        print(f"    Generating Grad-CAM visualizations...")
        try:
            target_layer = get_target_layer(model, model_name)
            gradcam_analyzer = GradCAM(model, target_layer, device)

            for i in range(num_samples):
                # Original image Grad-CAM
                orig_sample = sample_inputs[i:i+1]

                gradcam_dir = os.path.join(model_interp_dir, 'gradcam')
                os.makedirs(gradcam_dir, exist_ok=True)

                fig_orig = gradcam_analyzer.visualize(
                    orig_sample,
                    target_class=None,
                    class_names=class_names,
                    save_path=os.path.join(gradcam_dir, f'original_sample_{i}.png')
                )
                plt.close(fig_orig)

                # Adversarial image Grad-CAM
                adv_sample = adv_inputs[i:i+1]
                fig_adv = gradcam_analyzer.visualize(
                    adv_sample,
                    target_class=None,
                    class_names=class_names,
                    save_path=os.path.join(gradcam_dir, f'adversarial_sample_{i}.png')
                )
                plt.close(fig_adv)

        except Exception as e:
            print(f"    Grad-CAM analysis failed for {model_name}: {e}")


def create_summary_report(attack_results, transfer_results, transfer_metrics, results_dir):
    """
    Create a comprehensive summary report of all experiments.

    Args:
        attack_results: Results from adversarial attacks
        transfer_results: Results from transferability analysis
        transfer_metrics: Transferability metrics
        results_dir: Directory to save report
    """
    print("\nCreating summary report...")

    report = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "problem": "Problem B: Adversarial Attacks",
            "models_tested": list(attack_results.keys()),
            "total_attack_types": sum(len(model_results) for model_results in attack_results.values())
        },
        "attack_summary": {},
        "transferability_summary": transfer_metrics,
        "key_findings": []
    }

    # Summarize attack results
    for model_name, model_results in attack_results.items():
        model_summary = {
            "total_attacks": len(model_results),
            "attack_details": {}
        }

        for attack_name, attack_info in model_results.items():
            attack_summary = {
                "success_rate": attack_info.get('success_rate', 0),
                "mean_l2_norm": attack_info.get('mean_l2_norm', 0),
                "mean_linf_norm": attack_info.get('mean_linf_norm', 0),
                "successful_samples": attack_info.get('successful_samples', 0),
                "total_samples": attack_info.get('total_samples', 0)
            }
            model_summary["attack_details"][attack_name] = attack_summary

        report["attack_summary"][model_name] = model_summary

    # Add key findings
    findings = []

    # Find most successful attacks
    best_attacks = []
    for model_name, model_results in attack_results.items():
        for attack_name, attack_info in model_results.items():
            success_rate = attack_info.get('success_rate', 0)
            if success_rate > 0.7:  # High success rate threshold
                best_attacks.append((model_name, attack_name, success_rate))

    if best_attacks:
        best_attacks.sort(key=lambda x: x[2], reverse=True)
        findings.append(f"Most successful attack: {best_attacks[0][1]} on {best_attacks[0][0]} "
                       f"with {best_attacks[0][2]:.1%} success rate")

    # Transferability findings
    transfer_ratio = transfer_metrics.get('transferability_ratio', 0)
    if transfer_ratio > 0.5:
        findings.append(f"High transferability observed with ratio {transfer_ratio:.3f}")
    elif transfer_ratio < 0.3:
        findings.append(f"Low transferability observed with ratio {transfer_ratio:.3f}")

    report["key_findings"] = findings

    # Save report
    report_path = os.path.join(results_dir, 'experiment_summary.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Create human-readable summary
    summary_path = os.path.join(results_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EE4745 Problem B: Adversarial Attacks - Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment conducted on: {report['experiment_info']['timestamp']}\n")
        f.write(f"Models tested: {', '.join(report['experiment_info']['models_tested'])}\n")
        f.write(f"Total attack configurations: {report['experiment_info']['total_attack_types']}\n\n")

        f.write("ATTACK RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for model_name, model_summary in report["attack_summary"].items():
            f.write(f"\n{model_name}:\n")
            for attack_name, attack_details in model_summary["attack_details"].items():
                success_rate = attack_details["success_rate"]
                l2_norm = attack_details["mean_l2_norm"]
                f.write(f"  {attack_name}: {success_rate:.1%} success, L2 norm: {l2_norm:.4f}\n")

        f.write(f"\nTRANSFERABILITY ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall transferability ratio: {transfer_metrics.get('transferability_ratio', 0):.3f}\n")
        f.write(f"Cross-model success rate: {transfer_metrics.get('cross_model_mean_success_rate', 0):.3f}\n")
        f.write(f"Same-model success rate: {transfer_metrics.get('same_model_mean_success_rate', 0):.3f}\n")

        if findings:
            f.write(f"\nKEY FINDINGS:\n")
            f.write("-" * 30 + "\n")
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")

    print(f"Summary report saved to {results_dir}")


def main():
    """Main execution function for Problem B."""
    parser = argparse.ArgumentParser(description='EE4745 Problem B: Adversarial Attacks')
    parser.add_argument('--data-dir', default='data', help='Path to dataset directory')
    parser.add_argument('--results-dir', default='results/problem_b', help='Directory to save results')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')

    args = parser.parse_args()

    # Set up device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    print("Starting Problem B: Adversarial Attacks Analysis")
    print("=" * 60)

    # Load models
    print("Loading trained models...")
    models = load_trained_models(device)
    if not models:
        print("ERROR: No models could be loaded!")
        return

    print(f"Loaded models: {list(models.keys())}")

    # Get test data
    print("\nPreparing test data...")
    test_loader = get_test_data(args.data_dir, args.num_samples, args.batch_size)

    # Generate adversarial examples
    print("\nStep 1: Generating adversarial examples...")
    attack_results = generate_adversarial_examples(models, test_loader, args.results_dir, device)

    # Analyze transferability
    print("\nStep 2: Analyzing transferability...")
    transfer_results, transfer_metrics = analyze_transferability(
        models, test_loader, args.results_dir, device
    )

    # Analyze interpretability
    print("\nStep 3: Analyzing interpretability...")
    analyze_interpretability(models, test_loader, args.results_dir, device)

    # Create summary report
    print("\nStep 4: Creating summary report...")
    create_summary_report(attack_results, transfer_results, transfer_metrics, args.results_dir)

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print("\nGenerated outputs:")
    print(f"  - Adversarial examples: {args.results_dir}/adversarial_examples/")
    print(f"  - Transferability analysis: {args.results_dir}/transferability/")
    print(f"  - Interpretability analysis: {args.results_dir}/interpretability/")
    print(f"  - Summary reports: {args.results_dir}/experiment_summary.*")


if __name__ == "__main__":
    main()