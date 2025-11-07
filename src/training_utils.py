import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, base_lr=None):
        self.optimizer = optimizer
        self.d_model = d_model  # Kept for compatibility, not used in calculation
        self.warmup_steps = warmup_steps
        self.step_count = 0
        # Use provided base_lr or get from optimizer's initial lr
        if base_lr is not None:
            self.base_lr = base_lr
        else:
            # Get the initial learning rate from optimizer
            self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self):
        """Calculate learning rate with warmup (ResNet-style)
        
        Simple warmup schedule:
        - During warmup: lr linearly increases from 0 to base_lr
        - After warmup: lr stays constant at base_lr
        
        This ensures the learning rate stabilizes at base_lr after warmup,
        which is more intuitive and easier to tune.
        """
        if self.step_count <= self.warmup_steps:
            # Warmup phase: linear increase from 0 to base_lr
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # After warmup: stay constant at base_lr
            lr = self.base_lr
        
        return lr

class LabelSmoothing(nn.Module):
    """Label smoothing for better training stability"""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = pred.size()
        
        # Calculate true distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))  # Exclude padding and target token
        true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
        true_dist[:, :, self.padding_idx] = 0
        
        # Mask padding tokens
        mask = (target != self.padding_idx).unsqueeze(2)
        true_dist = true_dist * mask.float()
        
        # Calculate loss
        log_pred = torch.log_softmax(pred, dim=2)
        loss = -torch.sum(true_dist * log_pred, dim=2)
        
        # Average over non-padding tokens
        mask = (target != self.padding_idx).float()
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        
        return loss

class GradientClipper:
    """Gradient clipping utility"""
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
        
    def clip_gradients(self, model):
        """Clip gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class TrainingStability:
    """Training stability utilities"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize optimizer with AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        # Pass base_lr explicitly so sensitivity analysis can test different learning rates
        self.scheduler = WarmupScheduler(
            self.optimizer, 
            config.d_model, 
            config.warmup_steps,
            base_lr=config.learning_rate  # Use config learning rate as base
        )
        
        # Initialize gradient clipper
        self.gradient_clipper = GradientClipper(config.max_grad_norm)
        
        # Initialize label smoothing
        self.label_smoothing = LabelSmoothing(
            config.vocab_size,
            padding_idx=0,
            smoothing=0.1
        )
        
        # Training statistics
        self.training_stats = {
            'train_losses': [],
            'valid_losses': [],
            'learning_rates': [],
            'gradient_norms': []
        }
        
    def get_optimizer(self):
        """Get optimizer"""
        return self.optimizer
    
    def get_scheduler(self):
        """Get scheduler"""
        return self.scheduler
    
    def step(self, loss):
        """Perform optimization step with stability measures"""
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm before clipping
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        self.gradient_clipper.clip_gradients(self.model)
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduling
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return total_norm
    
    def calculate_loss(self, predictions, targets, use_label_smoothing=True):
        """Calculate loss with optional label smoothing"""
        if use_label_smoothing:
            return self.label_smoothing(predictions, targets)
        else:
            # Standard cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            return loss_fn(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    
    def update_stats(self, train_loss, valid_loss, lr, grad_norm):
        """Update training statistics"""
        self.training_stats['train_losses'].append(train_loss)
        self.training_stats['valid_losses'].append(valid_loss)
        self.training_stats['learning_rates'].append(lr)
        self.training_stats['gradient_norms'].append(grad_norm)
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_stats['train_losses'], label='Train Loss')
        axes[0, 0].plot(self.training_stats['valid_losses'], label='Valid Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate curve
        axes[0, 1].plot(self.training_stats['learning_rates'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Gradient norm curve
        axes[1, 0].plot(self.training_stats['gradient_norms'])
        axes[1, 0].set_title('Gradient Norms')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].grid(True)
        
        # Loss comparison (log scale)
        axes[1, 1].semilogy(self.training_stats['train_losses'], label='Train Loss')
        axes[1, 1].semilogy(self.training_stats['valid_losses'], label='Valid Loss')
        axes[1, 1].set_title('Loss Curves (Log Scale)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to avoid blocking
    
    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_training_stats(self, path):
        """Save training statistics"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def load_training_stats(self, path):
        """Load training statistics"""
        import pickle
        with open(path, 'rb') as f:
            self.training_stats = pickle.load(f)

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def create_optimizer_and_scheduler(model, config):
    """Create optimizer and scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    scheduler = WarmupScheduler(
        optimizer,
        config.d_model,
        config.warmup_steps
    )
    
    return optimizer, scheduler

if __name__ == "__main__":
    # Test training stability utilities
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
    
    stability = TrainingStability(model, config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Initial learning rate: {stability.get_current_lr():.6f}")
    
    # Test scheduler
    for i in range(10):
        stability.scheduler.step()
        print(f"Step {i+1}: LR = {stability.get_current_lr():.6f}")
