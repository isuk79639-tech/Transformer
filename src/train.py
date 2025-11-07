#!/usr/bin/env python3
"""
Main training script for Transformer model
"""

import argparse
import torch
import logging
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, set_seed, setup_logging, create_directories
from transformer import Transformer
from trainer import Trainer
from data_utils import DataProcessor
from experiments import ExperimentRunner

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Transformer model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: use config value)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=30000, help='Vocabulary size')
    
    # Training stability arguments
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps for learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    
    # Experiment arguments
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--run_sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--ablation_epochs', type=int, default=10, help='Epochs for ablation study')
    parser.add_argument('--sensitivity_epochs', type=int, default=5, help='Epochs for sensitivity analysis')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create config from command line arguments"""
    config = Config()
    
    # Update config with command line arguments
    config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.d_model = args.d_model
    config.nhead = args.nhead
    config.num_encoder_layers = args.num_layers
    config.num_decoder_layers = args.num_layers
    config.dropout = args.dropout
    config.max_length = args.max_length
    config.vocab_size = args.vocab_size
    config.warmup_steps = args.warmup_steps
    config.max_grad_norm = args.max_grad_norm
    config.seed = args.seed
    config.save_dir = args.save_dir
    config.log_dir = args.log_dir
    
    return config

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create config
    config = create_config_from_args(args)
    
    # Create directories
    create_directories(config)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f'training_{timestamp}.log')
    logger = setup_logging(log_file)
    
    logger.info("Starting Transformer training")
    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config.__dict__}")
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        logger.info("Loading data...")
        logger.info("Using real IWSLT2017 dataset")
        data_processor = DataProcessor(
            data_dir='data',
            max_length=config.max_length,
            min_freq=2,
            use_real_data=True  # Use real IWSLT2017 dataset
        )
        
        train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = data_processor.create_data_loaders(
            batch_size=config.batch_size
        )
        
        logger.info(f"Data loaded successfully")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Valid batches: {len(valid_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        logger.info(f"Source vocab size: {len(src_vocab)}")
        logger.info(f"Target vocab size: {len(tgt_vocab)}")
        
        # Auto-adjust warmup_steps based on dataset size
        # Strategy: warmup should cover ~5-10% of total training steps or ~1-2 epochs
        # This adapts automatically to dataset size
        steps_per_epoch = len(train_loader)
        total_training_steps = steps_per_epoch * config.num_epochs
        
        # Calculate recommended warmup_steps using multiple heuristics
        # 1. 5-10% of total training steps (good for large datasets)
        warmup_by_ratio = int(total_training_steps * 0.08)  # 8% of total steps
        
        # 2. 1-2 epochs worth of steps (good for any dataset size)
        warmup_by_epochs = min(steps_per_epoch * 2, steps_per_epoch + 500)
        
        # 3. Use the smaller of the two, but with reasonable bounds
        recommended_warmup = min(warmup_by_ratio, warmup_by_epochs)
        recommended_warmup = max(recommended_warmup, 100)  # At least 100 steps
        recommended_warmup = min(recommended_warmup, 10000)  # At most 10000 steps (for very large datasets)
        
        # Auto-adjust if user-specified value seems inappropriate
        # If user value is more than 3x recommended, use recommended instead
        if config.warmup_steps > recommended_warmup * 3:
            logger.info(f"Auto-adjusting warmup_steps based on dataset size:")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_training_steps}")
            logger.info(f"  User specified: {config.warmup_steps}")
            logger.info(f"  Recommended: {recommended_warmup} (covers ~{recommended_warmup/steps_per_epoch:.1f} epochs)")
            config.warmup_steps = recommended_warmup
            logger.info(f"  → Using warmup_steps: {config.warmup_steps}")
        elif config.warmup_steps == 100:  # Default value, auto-use recommended
            logger.info(f"Auto-setting warmup_steps based on dataset size:")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_training_steps}")
            config.warmup_steps = recommended_warmup
            logger.info(f"  → Set warmup_steps to: {config.warmup_steps} (covers ~{config.warmup_steps/steps_per_epoch:.1f} epochs)")
        else:
            logger.info(f"Using user-specified warmup_steps: {config.warmup_steps}")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Covers ~{config.warmup_steps/steps_per_epoch:.1f} epochs")
            logger.info(f"  (Recommended: {recommended_warmup} for this dataset size)")
        
        # Update config with actual vocab sizes
        config.src_vocab_size = len(src_vocab)
        config.tgt_vocab_size = len(tgt_vocab)
        
        # Create model
        logger.info("Creating model...")
        model = Transformer(
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_ff=config.dim_feedforward,
            dropout=config.dropout
        )
        
        logger.info(f"Model created with {model.count_parameters():,} parameters")
        logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")
        
        # Create trainer
        trainer = Trainer(model, config, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_model(args.resume)
        
        # Run experiments if requested (before normal training)
        if args.run_ablation or args.run_sensitivity:
            logger.info("Running experiments...")
            experiment_runner = ExperimentRunner()
            experiment_runner.config = config
            experiment_runner.data_processor = data_processor
            
            ablation_results = None
            sensitivity_results = None
            
            # Run ablation study if requested
            if args.run_ablation:
                logger.info("Running ablation study...")
                ablation_study = experiment_runner._create_ablation_study()
                ablation_results = ablation_study.run_ablation_experiments(
                    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, args.ablation_epochs
                )
                logger.info("Ablation study completed")
            
            # Run sensitivity analysis if requested
            if args.run_sensitivity:
                logger.info("Running sensitivity analysis...")
                sensitivity_analysis = experiment_runner._create_sensitivity_analysis()
                sensitivity_results = sensitivity_analysis.run_sensitivity_analysis(
                    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, args.sensitivity_epochs
                )
                logger.info("Sensitivity analysis completed")
            
            # Generate summary report if both experiments were run
            if ablation_results is not None and sensitivity_results is not None:
                experiment_runner.generate_summary_report(ablation_results, sensitivity_results)
            
            logger.info("Experiments completed")
            # Exit after experiments, don't run normal training
            return
        
        # Train model (only if not running experiments)
        logger.info("Starting training...")
        best_valid_loss = trainer.train(train_loader, valid_loader, config.num_epochs)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, bleu_score = trainer.evaluate(test_loader, src_vocab, tgt_vocab)
        
        logger.info(f"Training completed!")
        logger.info(f"Best validation loss: {best_valid_loss:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"BLEU score: {bleu_score:.4f}")
        
        # Save final model
        final_model_path = os.path.join(config.save_dir, f'final_model_{timestamp}.pt')
        trainer.save_model(final_model_path)
        
        # Generate some example translations
        logger.info("Generating example translations...")
        # Examples suitable for real IWSLT2017 dataset (TED talks style)
        # Using sentences that better reflect the complexity of the training data
        examples = [
            "hello world",  # Keep one simple example for comparison
            "how are you",  # Keep one simple example
            "thank you so much",  # Common in TED talks
            "i want to talk about one of the biggest problems",  # More complex, TED-style
            "do you know how many choices you make in a typical day",  # Realistic TED sentence
            "we need to think about the future",  # Medium complexity
            "this is a very important topic"  # Medium complexity
        ]
        
        for example in examples:
            try:
                translation = trainer.generate_translation(example, src_vocab, tgt_vocab)
                # Direct logging without ASCII filtering to preserve real dataset characters
                logger.info(f"'{example}' -> '{translation}'")
            except Exception as e:
                logger.warning(f"Failed to translate '{example}': {str(e)}")
        
        logger.info("Training script completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
