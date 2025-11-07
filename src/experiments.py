import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
import json
import os
from copy import deepcopy
import time
from tqdm import tqdm

from config import Config
from transformer import Transformer
from trainer import Trainer
from data_utils import DataProcessor
from model_utils import ExperimentTracker

class AblationStudy:
    """Ablation study for Transformer components"""
    def __init__(self, base_config, data_processor):
        self.base_config = base_config
        self.data_processor = data_processor
        self.results = {}
        
    def run_ablation_experiments(self, train_loader, valid_loader, test_loader, 
                                src_vocab, tgt_vocab, num_epochs=10):
        """Run ablation experiments"""
        print("Starting Ablation Study...")
        
        # Define ablation configurations
        ablation_configs = {
            'baseline': {
                'name': 'Baseline',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            # Dropout variations
            'no_dropout': {
                'name': 'No Dropout',
                'dropout': 0.0,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'high_dropout': {
                'name': 'High Dropout',
                'dropout': 0.3,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            # Layer depth variations (both encoder and decoder)
            'fewer_layers': {
                'name': 'Fewer Layers (3+3)',
                'dropout': 0.1,
                'num_encoder_layers': 3,
                'num_decoder_layers': 3,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'more_layers': {
                'name': 'More Layers (9+9)',
                'dropout': 0.1,
                'num_encoder_layers': 9,
                'num_decoder_layers': 9,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            # Separate encoder/decoder layer variations
            'fewer_encoder_layers': {
                'name': 'Fewer Encoder Layers (3+6)',
                'dropout': 0.1,
                'num_encoder_layers': 3,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'fewer_decoder_layers': {
                'name': 'Fewer Decoder Layers (6+3)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 3,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'more_encoder_layers': {
                'name': 'More Encoder Layers (9+6)',
                'dropout': 0.1,
                'num_encoder_layers': 9,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'more_decoder_layers': {
                'name': 'More Decoder Layers (6+9)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 9,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            # Attention head variations
            'fewer_heads': {
                'name': 'Fewer Heads (4)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 4,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            'more_heads': {
                'name': 'More Heads (16)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 16,
                'd_model': 512,
                'dim_feedforward': 2048
            },
            # Model dimension variations
            'smaller_model': {
                'name': 'Smaller Model (d_model=256)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 256,
                'dim_feedforward': 1024  # Keep d_ff = 4 * d_model
            },
            'larger_model': {
                'name': 'Larger Model (d_model=1024)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 1024,
                'dim_feedforward': 4096  # Keep d_ff = 4 * d_model
            },
            # Feed-forward network dimension variations
            'smaller_ff': {
                'name': 'Smaller FF Network (d_ff=1024)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 1024
            },
            'larger_ff': {
                'name': 'Larger FF Network (d_ff=4096)',
                'dropout': 0.1,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'nhead': 8,
                'd_model': 512,
                'dim_feedforward': 4096
            }
        }
        
        # Run experiments
        for config_name, config_params in ablation_configs.items():
            print(f"\nRunning experiment: {config_params['name']}")
            
            # Create config
            config = deepcopy(self.base_config)
            for key, value in config_params.items():
                if key != 'name':
                    setattr(config, key, value)
            
            # Create model
            model = Transformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                num_decoder_layers=config.num_decoder_layers,
                d_ff=config.dim_feedforward,
                dropout=config.dropout
            )
            
            # Create trainer
            trainer = Trainer(model, config)
            
            # Create unique experiment name for this ablation config
            experiment_name = f"ablation_{config_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            # Train model
            start_time = time.time()
            best_valid_loss = trainer.train(train_loader, valid_loader, num_epochs, experiment_name=experiment_name)
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_loss, bleu_score = trainer.evaluate(test_loader, src_vocab, tgt_vocab)
            
            # Store results
            self.results[config_name] = {
                'name': config_params['name'],
                'config': config_params,
                'best_valid_loss': best_valid_loss,
                'test_loss': test_loss,
                'bleu_score': bleu_score,
                'training_time': training_time,
                'num_parameters': trainer.model_analyzer.count_parameters()['total'],
                'model_size_mb': trainer.model_analyzer.get_model_size_mb()
            }
            
            print(f"Completed: {config_params['name']}")
            print(f"  Best Valid Loss: {best_valid_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  BLEU Score: {bleu_score:.4f}")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Parameters: {self.results[config_name]['num_parameters']:,}")
        
        # Save results
        self.save_results()
        
        # Plot results
        self.plot_ablation_results()
        
        return self.results
    
    def save_results(self, path='results/ablation_results.json'):
        """Save ablation results"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Ablation results saved to {path}")
    
    def plot_ablation_results(self):
        """Plot ablation study results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Prepare data for plotting
        names = []
        valid_losses = []
        test_losses = []
        bleu_scores = []
        training_times = []
        num_parameters = []
        
        for config_name, result in self.results.items():
            names.append(result['name'])
            valid_losses.append(result['best_valid_loss'])
            test_losses.append(result['test_loss'])
            bleu_scores.append(result['bleu_score'])
            training_times.append(result['training_time'])
            num_parameters.append(result['num_parameters'])
        
        # Create plots with larger figure size to accommodate rotated labels
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Set smaller font size for all text elements
        plt.rcParams.update({'font.size': 8})
        
        # Valid Loss
        axes[0, 0].bar(names, valid_losses, alpha=0.7)
        axes[0, 0].set_title('Best Validation Loss', fontsize=10)
        axes[0, 0].set_ylabel('Loss', fontsize=9)
        axes[0, 0].tick_params(axis='x', rotation=90, labelsize=7)
        axes[0, 0].tick_params(axis='y', labelsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test Loss
        axes[0, 1].bar(names, test_losses, alpha=0.7, color='orange')
        axes[0, 1].set_title('Test Loss', fontsize=10)
        axes[0, 1].set_ylabel('Loss', fontsize=9)
        axes[0, 1].tick_params(axis='x', rotation=90, labelsize=7)
        axes[0, 1].tick_params(axis='y', labelsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # BLEU Score
        axes[0, 2].bar(names, bleu_scores, alpha=0.7, color='green')
        axes[0, 2].set_title('BLEU Score', fontsize=10)
        axes[0, 2].set_ylabel('BLEU', fontsize=9)
        axes[0, 2].tick_params(axis='x', rotation=90, labelsize=7)
        axes[0, 2].tick_params(axis='y', labelsize=8)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training Time
        axes[1, 0].bar(names, training_times, alpha=0.7, color='red')
        axes[1, 0].set_title('Training Time', fontsize=10)
        axes[1, 0].set_ylabel('Time (seconds)', fontsize=9)
        axes[1, 0].tick_params(axis='x', rotation=90, labelsize=7)
        axes[1, 0].tick_params(axis='y', labelsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of Parameters
        axes[1, 1].bar(names, num_parameters, alpha=0.7, color='purple')
        axes[1, 1].set_title('Number of Parameters', fontsize=10)
        axes[1, 1].set_ylabel('Parameters', fontsize=9)
        axes[1, 1].tick_params(axis='x', rotation=90, labelsize=7)
        axes[1, 1].tick_params(axis='y', labelsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Loss vs Parameters scatter plot
        axes[1, 2].scatter(num_parameters, valid_losses, alpha=0.7, s=100)
        axes[1, 2].set_title('Valid Loss vs Parameters', fontsize=10)
        axes[1, 2].set_xlabel('Number of Parameters', fontsize=9)
        axes[1, 2].set_ylabel('Valid Loss', fontsize=9)
        axes[1, 2].tick_params(axis='both', labelsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add labels to scatter plot with smaller font
        for i, name in enumerate(names):
            axes[1, 2].annotate(name, (num_parameters[i], valid_losses[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=6)
        
        plt.tight_layout()
        plt.savefig('results/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()  # Use close() instead of show() to avoid blocking

class HyperparameterSensitivity:
    """Hyperparameter sensitivity analysis"""
    def __init__(self, base_config, data_processor):
        self.base_config = base_config
        self.data_processor = data_processor
        self.results = {}
        
    def run_sensitivity_analysis(self, train_loader, valid_loader, test_loader,
                                src_vocab, tgt_vocab, num_epochs=5):
        """Run hyperparameter sensitivity analysis"""
        print("Starting Hyperparameter Sensitivity Analysis...")
        
        # Define parameter ranges
        # Note: Only learning_rate is tested here, as other parameters (dropout, d_model, nhead, num_layers)
        # are already covered in ablation experiments. batch_size is fixed at 128.
        param_ranges = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        }
        
        # Run experiments for learning_rate only
        # Other parameters are tested in ablation experiments
        for param_name, param_values in param_ranges.items():
            print(f"\nAnalyzing sensitivity to {param_name}...")
            
            param_results = []
            
            for param_value in tqdm(param_values, desc=f"{param_name}"):
                # Create config
                config = deepcopy(self.base_config)
                setattr(config, param_name, param_value)
                
                # Create model
                model = Transformer(
                    src_vocab_size=len(src_vocab),
                    tgt_vocab_size=len(tgt_vocab),
                    d_model=config.d_model,
                    nhead=config.nhead,
                    num_encoder_layers=config.num_encoder_layers,
                    num_decoder_layers=config.num_decoder_layers,
                    d_ff=config.dim_feedforward,
                    dropout=config.dropout
                )
                
                # Create trainer
                trainer = Trainer(model, config)
                
                # Create unique experiment name for this sensitivity test
                # Replace special characters in param_value for filename safety
                safe_value = str(param_value).replace('.', '_').replace('-', '_')
                experiment_name = f"sensitivity_{param_name}_{safe_value}_{time.strftime('%Y%m%d_%H%M%S')}"
                
                # Train model
                start_time = time.time()
                best_valid_loss = trainer.train(train_loader, valid_loader, num_epochs, experiment_name=experiment_name)
                training_time = time.time() - start_time
                
                # Evaluate on test set
                test_loss, bleu_score = trainer.evaluate(test_loader, src_vocab, tgt_vocab)
                
                param_results.append({
                    'value': param_value,
                    'best_valid_loss': best_valid_loss,
                    'test_loss': test_loss,
                    'bleu_score': bleu_score,
                    'training_time': training_time
                })
            
            self.results[param_name] = param_results
        
        # Save results
        self.save_results()
        
        # Plot results
        self.plot_sensitivity_results()
        
        return self.results
    
    def save_results(self, path='results/sensitivity_results.json'):
        """Save sensitivity results"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Sensitivity results saved to {path}")
    
    def plot_sensitivity_results(self):
        """Plot sensitivity analysis results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create subplots
        num_params = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (param_name, param_results) in enumerate(self.results.items()):
            if i >= 6:  # Only plot first 6 parameters
                break
                
            values = [r['value'] for r in param_results]
            valid_losses = [r['best_valid_loss'] for r in param_results]
            test_losses = [r['test_loss'] for r in param_results]
            bleu_scores = [r['bleu_score'] for r in param_results]
            
            # Plot validation loss
            axes[i].plot(values, valid_losses, 'o-', label='Valid Loss', linewidth=2, markersize=6)
            axes[i].plot(values, test_losses, 's-', label='Test Loss', linewidth=2, markersize=6)
            
            # Create secondary y-axis for BLEU score
            ax2 = axes[i].twinx()
            ax2.plot(values, bleu_scores, '^-', color='green', label='BLEU Score', linewidth=2, markersize=6)
            
            axes[i].set_title(f'Sensitivity to {param_name}')
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Loss', color='blue')
            ax2.set_ylabel('BLEU Score', color='green')
            
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Hide unused subplots
        for i in range(len(self.results), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to avoid blocking

class ExperimentRunner:
    """Main experiment runner"""
    def __init__(self, config_path=None):
        self.config = Config()
        if config_path:
            self.config.load(config_path)
        
        self.data_processor = DataProcessor(use_real_data=True)  # Use real IWSLT2017 dataset
    
    def _create_ablation_study(self):
        """Create ablation study instance"""
        return AblationStudy(self.config, self.data_processor)
    
    def _create_sensitivity_analysis(self):
        """Create sensitivity analysis instance"""
        return HyperparameterSensitivity(self.config, self.data_processor)
    
    def run_all_experiments(self, num_epochs_ablation=10, num_epochs_sensitivity=5):
        """Run all experiments (both ablation and sensitivity)"""
        print("Setting up experiments...")
        
        # Load data
        train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = self.data_processor.create_data_loaders()
        
        print(f"Data loaded: {len(train_loader)} train batches, {len(valid_loader)} valid batches, {len(test_loader)} test batches")
        
        # Run ablation study
        print("\n" + "="*50)
        print("RUNNING ABLATION STUDY")
        print("="*50)
        
        ablation_study = self._create_ablation_study()
        ablation_results = ablation_study.run_ablation_experiments(
            train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, num_epochs_ablation
        )
        
        # Run sensitivity analysis
        print("\n" + "="*50)
        print("RUNNING SENSITIVITY ANALYSIS")
        print("="*50)
        
        sensitivity_analysis = self._create_sensitivity_analysis()
        sensitivity_results = sensitivity_analysis.run_sensitivity_analysis(
            train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, num_epochs_sensitivity
        )
        
        # Generate summary report
        self.generate_summary_report(ablation_results, sensitivity_results)
        
        return ablation_results, sensitivity_results
    
    def generate_summary_report(self, ablation_results, sensitivity_results):
        """Generate summary report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ablation_study': {
                'summary': self._summarize_ablation_results(ablation_results),
                'best_config': self._find_best_ablation_config(ablation_results)
            },
            'sensitivity_analysis': {
                'summary': self._summarize_sensitivity_results(sensitivity_results),
                'recommendations': self._generate_recommendations(sensitivity_results)
            }
        }
        
        # Save report
        with open('results/experiment_summary.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Best ablation config: {report['ablation_study']['best_config']}")
        print(f"Recommendations: {report['sensitivity_analysis']['recommendations']}")
    
    def _summarize_ablation_results(self, results):
        """Summarize ablation results"""
        best_loss = min(r['best_valid_loss'] for r in results.values())
        best_config = min(results.values(), key=lambda x: x['best_valid_loss'])
        
        return {
            'best_valid_loss': best_loss,
            'best_config_name': best_config['name'],
            'total_experiments': len(results)
        }
    
    def _find_best_ablation_config(self, results):
        """Find best ablation configuration"""
        return min(results.values(), key=lambda x: x['best_valid_loss'])['name']
    
    def _summarize_sensitivity_results(self, results):
        """Summarize sensitivity results"""
        summary = {}
        for param_name, param_results in results.items():
            best_result = min(param_results, key=lambda x: x['best_valid_loss'])
            summary[param_name] = {
                'best_value': best_result['value'],
                'best_loss': best_result['best_valid_loss']
            }
        return summary
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on sensitivity analysis"""
        recommendations = []
        
        for param_name, param_results in results.items():
            if len(param_results) < 2:
                continue
                
            losses = [r['best_valid_loss'] for r in param_results]
            values = [r['value'] for r in param_results]
            
            # Find optimal value
            best_idx = np.argmin(losses)
            best_value = values[best_idx]
            
            recommendations.append(f"Optimal {param_name}: {best_value}")
        
        return recommendations

if __name__ == "__main__":
    # Run experiments
    runner = ExperimentRunner()
    ablation_results, sensitivity_results = runner.run_all_experiments(
        num_epochs_ablation=5,  # Reduced for testing
        num_epochs_sensitivity=3  # Reduced for testing
    )
    
    print("All experiments completed!")
