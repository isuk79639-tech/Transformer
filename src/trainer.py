import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import logging
from components import create_masks
from training_utils import TrainingStability, EarlyStopping
from model_utils import ModelAnalyzer, ModelCheckpoint, ExperimentTracker

class Trainer:
    """Main training class for Transformer model"""
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize training utilities
        self.training_stability = TrainingStability(model, config)
        self.early_stopping = EarlyStopping(patience=getattr(config, 'patience', 7))
        
        # Initialize analysis utilities
        self.model_analyzer = ModelAnalyzer(model)
        self.checkpoint_manager = ModelCheckpoint(config.save_dir)
        self.experiment_tracker = ExperimentTracker(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_valid_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_token=0)
            
            # Forward pass
            predictions = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = self.training_stability.calculate_loss(predictions, tgt_output)
            
            # Backward pass and optimization step
            grad_norm = self.training_stability.step(loss)
            
            # Update statistics
            total_loss += loss.item()
            total_grad_norm += grad_norm
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'GradNorm': f'{grad_norm:.4f}',
                'LR': f'{self.training_stability.get_current_lr():.6f}'
            })
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {epoch+1}, Batch {batch_idx}, '
                    f'Loss: {loss.item():.4f}, '
                    f'GradNorm: {grad_norm:.4f}, '
                    f'LR: {self.training_stability.get_current_lr():.6f}'
                )
        
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
        return avg_loss, avg_grad_norm
    
    def validate_epoch(self, valid_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc=f'Epoch {epoch+1} [Valid]')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Create masks
                src_mask, tgt_mask = create_masks(src, tgt_input, pad_token=0)
                
                # Forward pass
                predictions = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Calculate loss
                loss = self.training_stability.calculate_loss(predictions, tgt_output)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, valid_loader, num_epochs=None, experiment_name=None):
        """Main training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        
        # Start experiment tracking
        self.experiment_tracker.start_experiment(self.config, experiment_name=experiment_name)
        
        # Print model summary
        self.model_analyzer.print_model_summary()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model_analyzer.count_parameters()['total']:,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, avg_grad_norm = self.train_epoch(train_loader, epoch)
            
            # Validation
            valid_loss = self.validate_epoch(valid_loader, epoch)
            
            # Update training statistics
            current_lr = self.training_stability.get_current_lr()
            self.training_stability.update_stats(train_loss, valid_loss, current_lr, avg_grad_norm)
            
            # Log metrics
            self.experiment_tracker.log_metrics(epoch, train_loss, valid_loss, {
                'learning_rate': current_lr
            })
            
            # Save checkpoint if best model
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                checkpoint_path = self.checkpoint_manager.save_best_model(
                    self.model, self.config, {'valid_loss': valid_loss, 'epoch': epoch}
                )
                self.experiment_tracker.log_checkpoint(checkpoint_path, epoch, valid_loss)
            
            # Regular checkpoint saving
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, 
                    self.training_stability.optimizer,
                    self.training_stability.scheduler,
                    epoch, valid_loss, self.config
                )
                self.experiment_tracker.log_checkpoint(checkpoint_path, epoch, valid_loss)
            
            # Log epoch results
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Valid Loss: {valid_loss:.4f}, '
                f'Best Valid Loss: {self.best_valid_loss:.4f}, '
                f'LR: {current_lr:.6f}'
            )
            
            # Early stopping check
            if self.early_stopping(valid_loss):
                self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # End experiment tracking
        training_time = time.time() - start_time
        self.experiment_tracker.end_experiment({
            'training_time': training_time,
            'best_valid_loss': self.best_valid_loss,
            'final_epoch': epoch
        })
        
        self.logger.info(f'Training completed in {training_time:.2f} seconds')
        self.logger.info(f'Best validation loss: {self.best_valid_loss:.4f}')
        
        # Plot training curves with unique filename
        # Use experiment name if available, otherwise use timestamp
        if self.experiment_tracker.current_experiment:
            exp_name = self.experiment_tracker.current_experiment['name']
            curve_path = f'results/training_curves_{exp_name}.png'
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            curve_path = f'results/training_curves_{timestamp}.png'
        
        self.training_stability.plot_training_curves(curve_path)
        self.logger.info(f'Training curves saved to {curve_path}')
        
        return self.best_valid_loss
    
    def evaluate(self, test_loader, src_vocab, tgt_vocab):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # For translation evaluation, we'll also compute BLEU score
        predictions = []
        targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Evaluating')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Create masks
                src_mask, tgt_mask = create_masks(src, tgt_input, pad_token=0)
                
                # Forward pass
                pred = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Calculate loss
                loss = self.training_stability.calculate_loss(pred, tgt_output)
                total_loss += loss.item()
                num_batches += 1
                
                # Generate translations for BLEU evaluation
                # Collect up to 100 samples for BLEU calculation
                # Take first 10 samples from each batch until we have 100 samples
                if len(predictions) < 100:
                    batch_size = src.size(0)
                    remaining_needed = 100 - len(predictions)
                    samples_per_batch = 10  # Take first 10 samples from each batch
                    samples_to_take = min(batch_size, samples_per_batch, remaining_needed)
                    
                    for i in range(samples_to_take):
                        # Generate translation
                        src_single = src[i:i+1]
                        src_mask_single = src_mask[i:i+1]
                        
                        generated = self.model.generate(
                            src_single, src_mask_single, 
                            max_length=self.config.max_length,
                            start_token=2, end_token=3
                        )
                        
                        # Convert to text (generated is now 1D tensor)
                        src_text = src_vocab.indices_to_sentence(src[i].cpu().numpy())
                        tgt_text = tgt_vocab.indices_to_sentence(tgt_output[i].cpu().numpy())
                        pred_text = tgt_vocab.indices_to_sentence(generated.cpu().numpy())
                        
                        predictions.append(pred_text)
                        targets.append(tgt_text)
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        # Calculate BLEU score (simplified version)
        num_sequences = len(predictions)
        bleu_score = self.calculate_bleu_score(predictions, targets)
        
        self.logger.info(f'Test Loss: {avg_loss:.4f}')
        self.logger.info(f'BLEU Score calculated on {num_sequences} sequences: {bleu_score:.4f}')
        
        return avg_loss, bleu_score
    
    def calculate_bleu_score(self, predictions, targets):
        """Calculate simplified BLEU score"""
        if not predictions or not targets:
            return 0.0
        
        total_score = 0
        for pred, target in zip(predictions, targets):
            pred_words = pred.split()
            target_words = target.split()
            
            # Simple 1-gram precision
            if not pred_words:
                continue
            
            matches = sum(1 for word in pred_words if word in target_words)
            precision = matches / len(pred_words)
            total_score += precision
        
        return total_score / len(predictions) if predictions else 0.0
    
    def generate_translation(self, src_text, src_vocab, tgt_vocab, max_length=100):
        self.model.eval()
        
        # Convert source text to indices
        src_indices = src_vocab.sentence_to_indices(src_text, max_length)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Create source mask
        src_mask = create_masks(src_tensor, src_tensor, pad_token=0)[0]
        
        # Generate translation
        with torch.no_grad():
            generated = self.model.generate(
                src_tensor, src_mask,
                max_length=max_length,
                start_token=2, end_token=3
            )
        
        # Convert to text (generated is now 1D tensor)
        translation = tgt_vocab.indices_to_sentence(generated.cpu().numpy())
        
        return translation
    
    def save_model(self, path):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'best_valid_loss': self.best_valid_loss,
            'current_epoch': self.current_epoch
        }, path)
        self.logger.info(f'Model saved to {path}')
    
    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.logger.info(f'Model loaded from {path}')

def create_trainer(model, config, device=None):
    """Create trainer instance"""
    return Trainer(model, config, device)

if __name__ == "__main__":
    # Test trainer
    from config import Config
    from transformer import Transformer
    from data_utils import DataProcessor
    
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
    
    trainer = Trainer(model, config)
    print("Trainer created successfully")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {trainer.model_analyzer.count_parameters()['total']:,}")
