import torch
import torch.nn as nn
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from datetime import datetime

class ModelAnalyzer:
    """Model analysis and statistics utilities"""
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def count_parameters(self):
        """Count total, trainable, and non-trainable parameters"""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        param_details = {}
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
            
            param_details[name] = {
                'shape': list(param.shape),
                'numel': param_count,
                'requires_grad': param.requires_grad,
                'dtype': str(param.dtype)
            }
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'details': param_details
        }
    
    def get_model_size_mb(self):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def analyze_layer_parameters(self):
        """Analyze parameters by layer type"""
        layer_stats = defaultdict(lambda: {'count': 0, 'params': 0})
        
        for name, param in self.model.named_parameters():
            layer_type = name.split('.')[0]  # Get layer type from name
            layer_stats[layer_type]['count'] += 1
            layer_stats[layer_type]['params'] += param.numel()
        
        return dict(layer_stats)
    
    def get_parameter_distribution(self):
        """Get parameter value distribution statistics"""
        param_stats = {}
        
        for name, param in self.model.named_parameters():
            param_data = param.data.cpu().numpy().flatten()
            param_stats[name] = {
                'mean': np.mean(param_data),
                'std': np.std(param_data),
                'min': np.min(param_data),
                'max': np.max(param_data),
                'median': np.median(param_data)
            }
        
        return param_stats
    
    def plot_parameter_distribution(self, save_path='parameter_distribution.png'):
        """Plot parameter distribution"""
        param_stats = self.get_parameter_distribution()
        
        # Select a few key parameters to plot
        key_params = ['src_embedding.weight', 'tgt_embedding.weight', 'output_projection.weight']
        available_params = [p for p in key_params if p in param_stats]
        
        if not available_params:
            # If key params not available, plot first few
            available_params = list(param_stats.keys())[:3]
        
        fig, axes = plt.subplots(1, len(available_params), figsize=(15, 5))
        if len(available_params) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(available_params):
            param = self.model.get_parameter(param_name)
            if param is not None:
                param_data = param.data.cpu().numpy().flatten()
                axes[i].hist(param_data, bins=50, alpha=0.7)
                axes[i].set_title(f'{param_name}\nMean: {np.mean(param_data):.4f}, Std: {np.std(param_data):.4f}')
                axes[i].set_xlabel('Parameter Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to avoid blocking
    
    def get_gradient_norms(self):
        """Get gradient norms for each parameter"""
        grad_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms[name] = grad_norm
        
        return grad_norms
    
    def print_model_summary(self):
        """Print comprehensive model summary"""
        param_info = self.count_parameters()
        layer_stats = self.analyze_layer_parameters()
        
        print("=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        
        print(f"Total Parameters: {param_info['total']:,}")
        print(f"Trainable Parameters: {param_info['trainable']:,}")
        print(f"Non-trainable Parameters: {param_info['non_trainable']:,}")
        print(f"Model Size: {self.get_model_size_mb():.2f} MB")
        print()
        
        print("Layer-wise Parameter Distribution:")
        print("-" * 50)
        for layer_type, stats in layer_stats.items():
            print(f"{layer_type:20s}: {stats['params']:>10,} params ({stats['count']} tensors)")
        print()
        
        print("Detailed Parameter Information:")
        print("-" * 50)
        for name, info in param_info['details'].items():
            shape_str = str(info['shape'])[:20]  # Convert list to string and truncate
            print(f"{name:40s}: {shape_str:20s} | {info['numel']:>8,} | {info['dtype']:>10s} | {'✓' if info['requires_grad'] else '✗'}")
        print("=" * 80)

class ModelCheckpoint:
    """Model checkpointing utilities"""
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                       config, additional_info=None, filename=None):
        """Save model checkpoint"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.step_count if hasattr(scheduler, 'step_count') else 0,
            'loss': loss,
            'config': config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def load_checkpoint(self, model, optimizer, scheduler, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if hasattr(scheduler, 'step_count'):
            scheduler.step_count = checkpoint['scheduler_state_dict']
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
        
        return epoch, loss, checkpoint.get('additional_info', {})
    
    def save_best_model(self, model, config, metrics, filename='best_model.pt'):
        """Save best model based on metrics"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Best model saved: {filepath}")
        
        return filepath
    
    def load_best_model(self, model, filepath='best_model.pt'):
        """Load best model"""
        full_path = os.path.join(self.save_dir, filepath)
        checkpoint = torch.load(full_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = checkpoint.get('metrics', {})
        
        print(f"Best model loaded: {full_path}")
        print(f"Metrics: {metrics}")
        
        return metrics

class ExperimentTracker:
    """Track and log experiments"""
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, config, experiment_name=None):
        """Start a new experiment"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.current_experiment = {
            'name': experiment_name,
            'config': config.__dict__,
            'start_time': datetime.now().isoformat(),
            'metrics': [],
            'checkpoints': [],
            'status': 'running'
        }
        
        print(f"Started experiment: {experiment_name}")
    
    def log_metrics(self, epoch, train_loss, valid_loss, additional_metrics=None):
        """Log metrics for current epoch"""
        if self.current_experiment is None:
            print("Warning: No active experiment. Call start_experiment() first.")
            return
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.current_experiment['metrics'].append(metrics)
    
    def log_checkpoint(self, checkpoint_path, epoch, loss):
        """Log checkpoint information"""
        if self.current_experiment is None:
            return
        
        checkpoint_info = {
            'path': checkpoint_path,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_experiment['checkpoints'].append(checkpoint_info)
    
    def end_experiment(self, final_metrics=None):
        """End current experiment"""
        if self.current_experiment is None:
            return
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['status'] = 'completed'
        
        if final_metrics:
            self.current_experiment['final_metrics'] = final_metrics
        
        # Save experiment log
        log_path = os.path.join(self.log_dir, f"{self.current_experiment['name']}.json")
        with open(log_path, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        self.experiments.append(self.current_experiment)
        print(f"Experiment completed: {self.current_experiment['name']}")
        
        self.current_experiment = None
    
    def get_experiment_summary(self):
        """Get summary of all experiments"""
        summary = []
        
        for exp in self.experiments:
            if exp['metrics']:
                best_valid_loss = min(m['valid_loss'] for m in exp['metrics'])
                final_train_loss = exp['metrics'][-1]['train_loss']
                final_valid_loss = exp['metrics'][-1]['valid_loss']
            else:
                best_valid_loss = None
                final_train_loss = None
                final_valid_loss = None
            
            summary.append({
                'name': exp['name'],
                'status': exp['status'],
                'start_time': exp['start_time'],
                'end_time': exp.get('end_time', 'N/A'),
                'best_valid_loss': best_valid_loss,
                'final_train_loss': final_train_loss,
                'final_valid_loss': final_valid_loss,
                'num_epochs': len(exp['metrics']),
                'num_checkpoints': len(exp['checkpoints'])
            })
        
        return summary
    
    def plot_experiment_comparison(self, save_path='experiment_comparison.png'):
        """Plot comparison of different experiments"""
        if len(self.experiments) < 2:
            print("Need at least 2 experiments to compare")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training losses
        for exp in self.experiments:
            if exp['metrics']:
                epochs = [m['epoch'] for m in exp['metrics']]
                train_losses = [m['train_loss'] for m in exp['metrics']]
                valid_losses = [m['valid_loss'] for m in exp['metrics']]
                
                axes[0].plot(epochs, train_losses, label=f"{exp['name']} (train)", linestyle='-')
                axes[0].plot(epochs, valid_losses, label=f"{exp['name']} (valid)", linestyle='--')
        
        axes[0].set_title('Training and Validation Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot final metrics comparison
        exp_names = [exp['name'] for exp in self.experiments if exp['metrics']]
        final_train_losses = [exp['metrics'][-1]['train_loss'] for exp in self.experiments if exp['metrics']]
        final_valid_losses = [exp['metrics'][-1]['valid_loss'] for exp in self.experiments if exp['metrics']]
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        axes[1].bar(x - width/2, final_train_losses, width, label='Final Train Loss', alpha=0.8)
        axes[1].bar(x + width/2, final_valid_losses, width, label='Final Valid Loss', alpha=0.8)
        
        axes[1].set_title('Final Loss Comparison')
        axes[1].set_xlabel('Experiment')
        axes[1].set_ylabel('Loss')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(exp_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to avoid blocking

if __name__ == "__main__":
    # Test model analysis utilities
    from config import Config
    from transformer import Transformer
    
    config = Config()
    model = Transformer(
        src_vocab_size=config.vocab_size,
        tgt_vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.dim_feedforward,
        dropout=config.dropout
    )
    
    analyzer = ModelAnalyzer(model)
    analyzer.print_model_summary()
    
    checkpoint_manager = ModelCheckpoint()
    experiment_tracker = ExperimentTracker()
    
    print(f"Model size: {analyzer.get_model_size_mb():.2f} MB")
