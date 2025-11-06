#!/usr/bin/env python3
"""
EE4745 Neural Network Final Project - Final Results Compiler
===========================================================

Comprehensive results compilation, analysis, and reporting system for the
three-part neural network project:
- Problem A: Sports Image Classification
- Problem B: Adversarial Attack Analysis
- Problem C: Model Compression via Pruning

This script consolidates all experimental results, generates comparison tables,
creates comprehensive visualizations, and produces analysis reports.

Authors: Tyler Trauernicht, Vinh Le
Date: November 2025
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinalResultsCompiler:
    """
    Main class for compiling and analyzing final project results.
    """

    def __init__(self, project_root: str):
        """Initialize the results compiler."""
        self.project_root = Path(project_root)
        self.results_root = self.project_root / "results"
        self.final_root = self.results_root / "final"

        # Create subdirectories
        self.tables_dir = self.final_root / "tables"
        self.figures_dir = self.final_root / "figures"
        self.reports_dir = self.final_root / "reports"
        self.summary_dir = self.final_root / "summary"
        self.analysis_dir = self.final_root / "analysis"

        # Problem-specific results directories
        self.problem_a_dir = self.results_root / "problem_a"
        self.problem_b_dir = self.results_root / "problem_b"
        self.problem_c_dir = self.results_root / "problem_c"

        # Initialize storage for results
        self.problem_a_results = {}
        self.problem_b_results = {}
        self.problem_c_results = {}

        print(f"Initialized Final Results Compiler")
        print(f"Project root: {self.project_root}")
        print(f"Results will be saved to: {self.final_root}")

    def load_all_results(self):
        """Load all available results from the three problems."""
        print("\n" + "="*60)
        print("LOADING ALL EXPERIMENTAL RESULTS")
        print("="*60)

        self.load_problem_a_results()
        self.load_problem_b_results()
        self.load_problem_c_results()

        print(f"\nResults loading completed!")

    def load_problem_a_results(self):
        """Load Problem A (Classification) results."""
        print("\nLoading Problem A (Sports Image Classification) results...")

        # Template structure for Problem A results
        self.problem_a_results = {
            'models': {
                'SimpleCNN': {
                    'train_accuracy': 0.0,
                    'val_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'parameters': 0,
                    'training_time': 0.0,
                    'train_losses': [],
                    'val_losses': [],
                    'train_accuracies': [],
                    'val_accuracies': [],
                    'confusion_matrix': None,
                    'per_class_accuracy': {},
                    'interpretability_scores': {}
                },
                'ResNetSmall': {
                    'train_accuracy': 0.0,
                    'val_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'parameters': 0,
                    'training_time': 0.0,
                    'train_losses': [],
                    'val_losses': [],
                    'train_accuracies': [],
                    'val_accuracies': [],
                    'confusion_matrix': None,
                    'per_class_accuracy': {},
                    'interpretability_scores': {}
                }
            },
            'dataset_info': {
                'classes': ['baseball', 'basketball', 'football', 'golf', 'hockey',
                           'rugby', 'swimming', 'tennis', 'volleyball', 'weightlifting'],
                'train_samples': 1593,
                'val_samples': 50,
                'test_samples': 50,
                'image_size': (32, 32)
            }
        }

        # Try to load actual results if they exist
        try:
            problem_a_file = self.problem_a_dir / "classification_results.json"
            if problem_a_file.exists():
                with open(problem_a_file, 'r') as f:
                    loaded_results = json.load(f)
                    self.problem_a_results.update(loaded_results)
                print(f"✓ Loaded Problem A results from {problem_a_file}")
            else:
                print(f"⚠ No existing Problem A results found, using template structure")
        except Exception as e:
            print(f"⚠ Error loading Problem A results: {e}")

    def load_problem_b_results(self):
        """Load Problem B (Adversarial Attacks) results."""
        print("\nLoading Problem B (Adversarial Attack Analysis) results...")

        # Template structure for Problem B results
        self.problem_b_results = {
            'attacks': {
                'FGSM': {
                    'untargeted': {
                        'success_rates': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'perturbation_norms': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'model_robustness': {'SimpleCNN': 0.0, 'ResNetSmall': 0.0}
                    },
                    'targeted': {
                        'success_rates': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'perturbation_norms': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'target_class': 'basketball'
                    }
                },
                'PGD': {
                    'untargeted': {
                        'success_rates': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'perturbation_norms': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'model_robustness': {'SimpleCNN': 0.0, 'ResNetSmall': 0.0}
                    },
                    'targeted': {
                        'success_rates': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'perturbation_norms': {'eps_0.01': 0.0, 'eps_0.03': 0.0, 'eps_0.05': 0.0, 'eps_0.1': 0.0},
                        'target_class': 'basketball'
                    }
                }
            },
            'transferability': {
                'SimpleCNN_to_ResNetSmall': 0.0,
                'ResNetSmall_to_SimpleCNN': 0.0
            },
            'defense_analysis': {
                'baseline_accuracy': 0.0,
                'adversarial_accuracy': 0.0,
                'robust_accuracy': 0.0
            }
        }

        # Try to load actual results if they exist
        try:
            problem_b_file = self.problem_b_dir / "adversarial_results.json"
            if problem_b_file.exists():
                with open(problem_b_file, 'r') as f:
                    loaded_results = json.load(f)
                    self.problem_b_results.update(loaded_results)
                print(f"✓ Loaded Problem B results from {problem_b_file}")
            else:
                print(f"⚠ No existing Problem B results found, using template structure")
        except Exception as e:
            print(f"⚠ Error loading Problem B results: {e}")

    def load_problem_c_results(self):
        """Load Problem C (Model Compression) results."""
        print("\nLoading Problem C (Model Compression via Pruning) results...")

        # Template structure for Problem C results
        self.problem_c_results = {
            'pruning_levels': {
                '20%': {
                    'SimpleCNN': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.8
                    },
                    'ResNetSmall': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.8
                    }
                },
                '50%': {
                    'SimpleCNN': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.5
                    },
                    'ResNetSmall': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.5
                    }
                },
                '80%': {
                    'SimpleCNN': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.2
                    },
                    'ResNetSmall': {
                        'accuracy': 0.0,
                        'model_size_mb': 0.0,
                        'inference_time_ms': 0.0,
                        'adversarial_robustness': 0.0,
                        'compression_ratio': 0.2
                    }
                }
            },
            'baseline_performance': {
                'SimpleCNN': {
                    'accuracy': 0.0,
                    'model_size_mb': 0.0,
                    'inference_time_ms': 0.0
                },
                'ResNetSmall': {
                    'accuracy': 0.0,
                    'model_size_mb': 0.0,
                    'inference_time_ms': 0.0
                }
            }
        }

        # Try to load actual results if they exist
        try:
            problem_c_file = self.problem_c_dir / "pruning_results.json"
            if problem_c_file.exists():
                with open(problem_c_file, 'r') as f:
                    loaded_results = json.load(f)
                    self.problem_c_results.update(loaded_results)
                print(f"✓ Loaded Problem C results from {problem_c_file}")
            else:
                print(f"⚠ No existing Problem C results found, using template structure")
        except Exception as e:
            print(f"⚠ Error loading Problem C results: {e}")

    def generate_all_tables(self):
        """Generate all comparison tables."""
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE COMPARISON TABLES")
        print("="*60)

        self.generate_problem_a_table()
        self.generate_problem_b_table()
        self.generate_problem_c_table()
        self.generate_master_comparison_table()

        print(f"\nAll comparison tables generated!")

    def generate_problem_a_table(self):
        """Generate Problem A model comparison table."""
        print("\nGenerating Problem A (Model Comparison) table...")

        # Create DataFrame for model comparison
        models = ['SimpleCNN', 'ResNetSmall']
        data = []

        for model in models:
            model_data = self.problem_a_results['models'][model]
            data.append([
                model,
                f"{model_data['train_accuracy']:.3f}",
                f"{model_data['val_accuracy']:.3f}",
                f"{model_data['test_accuracy']:.3f}",
                f"{model_data['parameters']:,}",
                f"{model_data['training_time']:.1f}s",
                "High" if model == 'SimpleCNN' else "Medium"  # Interpretability
            ])

        df = pd.DataFrame(data, columns=[
            'Model', 'Train Acc', 'Val Acc', 'Test Acc',
            'Parameters', 'Training Time', 'Interpretability'
        ])

        # Save as CSV
        csv_file = self.tables_dir / "problem_a_model_comparison.csv"
        df.to_csv(csv_file, index=False)

        # Save as LaTeX
        latex_file = self.tables_dir / "problem_a_model_comparison.tex"
        latex_table = df.to_latex(index=False, caption="Problem A: Model Performance Comparison",
                                  label="tab:problem_a_comparison")

        with open(latex_file, 'w') as f:
            f.write(latex_table)

        print(f"✓ Problem A table saved to {csv_file} and {latex_file}")

    def generate_problem_b_table(self):
        """Generate Problem B attack effectiveness table."""
        print("\nGenerating Problem B (Attack Effectiveness) table...")

        # Create DataFrame for attack comparison
        data = []

        for attack in ['FGSM', 'PGD']:
            for attack_type in ['untargeted', 'targeted']:
                attack_data = self.problem_b_results['attacks'][attack][attack_type]
                for eps in ['eps_0.01', 'eps_0.03', 'eps_0.05', 'eps_0.1']:
                    data.append([
                        attack,
                        attack_type.title(),
                        eps.replace('eps_', 'ε='),
                        f"{attack_data['success_rates'][eps]:.3f}",
                        f"{attack_data['perturbation_norms'][eps]:.4f}"
                    ])

        df = pd.DataFrame(data, columns=[
            'Attack', 'Type', 'Epsilon', 'Success Rate', 'Avg Perturbation'
        ])

        # Save as CSV
        csv_file = self.tables_dir / "problem_b_attack_comparison.csv"
        df.to_csv(csv_file, index=False)

        # Save as LaTeX
        latex_file = self.tables_dir / "problem_b_attack_comparison.tex"
        latex_table = df.to_latex(index=False, caption="Problem B: Attack Effectiveness Comparison",
                                  label="tab:problem_b_comparison")

        with open(latex_file, 'w') as f:
            f.write(latex_table)

        print(f"✓ Problem B table saved to {csv_file} and {latex_file}")

    def generate_problem_c_table(self):
        """Generate Problem C pruning trade-off table."""
        print("\nGenerating Problem C (Pruning Trade-offs) table...")

        # Create DataFrame for pruning comparison
        data = []

        for sparsity in ['20%', '50%', '80%']:
            for model in ['SimpleCNN', 'ResNetSmall']:
                model_data = self.problem_c_results['pruning_levels'][sparsity][model]
                data.append([
                    model,
                    sparsity,
                    f"{model_data['accuracy']:.3f}",
                    f"{model_data['model_size_mb']:.2f}",
                    f"{model_data['inference_time_ms']:.1f}",
                    f"{model_data['adversarial_robustness']:.3f}",
                    f"{model_data['compression_ratio']:.1f}x"
                ])

        df = pd.DataFrame(data, columns=[
            'Model', 'Sparsity', 'Accuracy', 'Size (MB)',
            'Inference (ms)', 'Robustness', 'Compression'
        ])

        # Save as CSV
        csv_file = self.tables_dir / "problem_c_pruning_comparison.csv"
        df.to_csv(csv_file, index=False)

        # Save as LaTeX
        latex_file = self.tables_dir / "problem_c_pruning_comparison.tex"
        latex_table = df.to_latex(index=False, caption="Problem C: Pruning Trade-off Analysis",
                                  label="tab:problem_c_comparison")

        with open(latex_file, 'w') as f:
            f.write(latex_table)

        print(f"✓ Problem C table saved to {csv_file} and {latex_file}")

    def generate_master_comparison_table(self):
        """Generate master performance comparison table."""
        print("\nGenerating Master Performance Comparison table...")

        # Create comprehensive comparison
        data = []

        # Problem A results
        for model in ['SimpleCNN', 'ResNetSmall']:
            model_data = self.problem_a_results['models'][model]
            data.append([
                'A: Classification',
                model,
                'Original',
                f"{model_data['test_accuracy']:.3f}",
                f"{model_data['parameters']:,}",
                'N/A',
                'High' if model == 'SimpleCNN' else 'Medium'
            ])

        # Problem C results (showing trade-offs)
        for model in ['SimpleCNN', 'ResNetSmall']:
            for sparsity in ['20%', '50%', '80%']:
                model_data = self.problem_c_results['pruning_levels'][sparsity][model]
                data.append([
                    'C: Compression',
                    model,
                    f'{sparsity} Pruned',
                    f"{model_data['accuracy']:.3f}",
                    f"{int(model_data['compression_ratio'] * self.problem_a_results['models'][model]['parameters']):,}",
                    f"{model_data['adversarial_robustness']:.3f}",
                    'Reduced'
                ])

        df = pd.DataFrame(data, columns=[
            'Problem', 'Model', 'Variant', 'Accuracy',
            'Parameters', 'Robustness', 'Interpretability'
        ])

        # Save as CSV
        csv_file = self.tables_dir / "master_performance_comparison.csv"
        df.to_csv(csv_file, index=False)

        # Save as LaTeX
        latex_file = self.tables_dir / "master_performance_comparison.tex"
        latex_table = df.to_latex(index=False, caption="Master Performance Comparison Across All Problems",
                                  label="tab:master_comparison")

        with open(latex_file, 'w') as f:
            f.write(latex_table)

        print(f"✓ Master comparison table saved to {csv_file} and {latex_file}")

if __name__ == "__main__":
    # Initialize the compiler
    project_root = "/Users/ty/Neural-Final-Tyler_Vinh"
    compiler = FinalResultsCompiler(project_root)

    # Load all results
    compiler.load_all_results()

    # Generate all tables
    compiler.generate_all_tables()

    print("\n" + "="*60)
    print("FINAL RESULTS COMPILATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {compiler.final_root}")