import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .utils import (
    AverageMeter, EarlyStopping, accuracy, save_checkpoint,
    format_time, count_parameters, save_training_config
)

class Trainer:
    """Training class for sports image classification"""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        self.device = torch.device(config.get('device', 'cpu'))
        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )

        # TensorBoard
        self.writer = None
        if config.get('use_tensorboard', True):
            log_dir = config.get('log_dir', 'logs')
            experiment_name = config.get('experiment_name', 'default')
            self.writer = SummaryWriter(os.path.join(log_dir, experiment_name))

        # Checkpoints
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        print(f"Training on device: {self.device}")
        total_params, trainable_params = count_parameters(self.model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr,
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            T_max = self.config.get('epochs', 100)
            eta_min = self.config.get('eta_min', 1e-6)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()

        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # TensorBoard logging
            if self.writer:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', acc1.item(), global_step)

        epoch_time = time.time() - start_time

        # Log epoch metrics
        if self.writer:
            self.writer.add_scalar('Train/EpochLoss', losses.avg, epoch)
            self.writer.add_scalar('Train/EpochAcc', top1.avg, epoch)
            self.writer.add_scalar('Train/EpochTime', epoch_time, epoch)
            self.writer.add_scalar('Train/LearningRate',
                                 self.optimizer.param_groups[0]['lr'], epoch)

        return losses.avg, top1.avg, epoch_time

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()

        all_preds = []
        all_targets = []

        start_time = time.time()

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Metrics
                acc1 = accuracy(output, target, topk=(1,))[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))

                # Store predictions for detailed analysis
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_time = time.time() - start_time

        # Log validation metrics
        if self.writer:
            self.writer.add_scalar('Val/EpochLoss', losses.avg, epoch)
            self.writer.add_scalar('Val/EpochAcc', top1.avg, epoch)
            self.writer.add_scalar('Val/EpochTime', val_time, epoch)

        return losses.avg, top1.avg, val_time, all_preds, all_targets

    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('epochs', 100)

        # Save training configuration
        config_path = os.path.join(self.checkpoint_dir, 'training_config.json')
        save_training_config(self.config, config_path)

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        total_start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            # Training
            train_loss, train_acc, train_time = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, val_time, val_preds, val_targets = self.validate(epoch)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% ({format_time(train_time)})')
            print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}% ({format_time(val_time)})')

            # Save best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                best_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f'{self.config["model_name"]}-best.pt'
                )
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_loss, val_acc, best_checkpoint_path
                )

                # Save detailed validation report for best model
                self._save_validation_report(val_targets, val_preds, epoch)

            # Save last checkpoint
            last_checkpoint_path = os.path.join(
                self.checkpoint_dir, f'{self.config["model_name"]}-last.pt'
            )
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss, val_acc, last_checkpoint_path
            )

            # Early stopping
            if self.early_stopping(val_acc, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break

            print()

        total_time = time.time() - total_start_time
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        return self.training_history

    def _save_validation_report(self, targets, preds, epoch):
        """Save detailed validation report"""
        from ..dataset.sports_dataset import SportsDataset

        class_names = SportsDataset.CLASSES

        # Classification report
        report = classification_report(
            targets, preds, target_names=class_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(targets, preds)

        report_path = os.path.join(self.checkpoint_dir, f'validation_report_epoch_{epoch+1}.txt')

        with open(report_path, 'w') as f:
            f.write(f"Validation Report - Epoch {epoch+1}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Per-class Performance:\n")
            f.write("-" * 30 + "\n")
            for class_name in class_names:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    support = report[class_name]['support']
                    f.write(f"{class_name:12}: P={precision:.3f}, R={recall:.3f}, "
                           f"F1={f1:.3f}, N={support}\n")

            f.write(f"\nOverall Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n")

            f.write("\nConfusion Matrix:\n")
            f.write("-" * 20 + "\n")
            f.write("    " + " ".join(f"{cls[:3]:>3}" for cls in class_names) + "\n")
            for i, row in enumerate(cm):
                f.write(f"{class_names[i][:3]:>3} " + " ".join(f"{val:3d}" for val in row) + "\n")

    def get_model(self):
        """Get the trained model"""
        return self.model

    def get_training_history(self):
        """Get training history"""
        return self.training_history