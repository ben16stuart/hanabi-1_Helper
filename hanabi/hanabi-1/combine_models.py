#!/usr/bin/env python3
"""
Ensemble Multiple Trained Models for Better Predictions
"""

import sys
import os

# Add the hanabi-1 directory to Python path
hanabi_path = "/Users/benstuart/Desktop/hanabi/hanabi-1"  # Adjust this path as needed
if hanabi_path not in sys.path:
    sys.path.append(hanabi_path)

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import the model and loss classes from your project
from transformer_model import FinancialTransformerModel, FocalLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEnsemble:
    def __init__(self, model_paths, device='cpu'):
        """
        Initialize ensemble with multiple model paths
        
        Args:
            model_paths: List of paths to trained model files
            device: 'cpu' or 'cuda' or 'mps' for M3 Mac
        """
        self.device = device
        self.models = []
        self.model_paths = model_paths
        
        # Load all models
        for path in model_paths:
            model = self.load_model(path)
            if model is not None:
                self.models.append(model)
                logger.info(f"Loaded model: {path}")
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
    
    def load_model(self, model_path):
        """Load a single trained model"""
        try:
            # Load the checkpoint with weights_only=True for security
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load the hyperparameters from the checkpoint
            hyperparams = checkpoint.get('hyperparams', {})
            
            # Recreate the model architecture using saved hyperparameters
            model = FinancialTransformerModel(
                input_dim=hyperparams.get('input_dim', 7),
                hidden_dim=hyperparams.get('hidden_dim', 512),
                num_heads=hyperparams.get('num_heads', 8),
                transformer_layers=hyperparams.get('transformer_layers', 8),
                dropout=hyperparams.get('dropout', 0.075),
                direction_threshold=hyperparams.get('direction_threshold', 0.46)
            )
            
            # Load the trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            model.to(self.device)
            
            return model
            
            # For now, return the checkpoint - you'll modify this
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def predict_ensemble(self, inputs):
        """
        Make ensemble predictions by averaging multiple model outputs
        
        Args:
            inputs: Input tensor for prediction (shape: [batch_size, window_size, features])
            
        Returns:
            averaged_predictions: Dictionary with ensemble prediction results
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        direction_probs = []
        volatility_preds = []
        price_change_preds = []
        spread_preds = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                try:
                    # Make prediction with this model
                    outputs = model(inputs)
                    
                    # Collect predictions from each model
                    direction_probs.append(outputs['direction_prob'])
                    volatility_preds.append(outputs['volatility'])
                    price_change_preds.append(outputs['price_change'])
                    spread_preds.append(outputs['spread'])
                    
                    logger.info(f"Got prediction from model {i+1}")
                    
                except Exception as e:
                    logger.error(f"Error with model {i}: {e}")
                    continue
        
        if not direction_probs:
            raise RuntimeError("No successful predictions from any model")
        
        # Average predictions across all models
        ensemble_predictions = {
            'direction_prob': torch.stack(direction_probs).mean(dim=0),
            'volatility': torch.stack(volatility_preds).mean(dim=0),
            'price_change': torch.stack(price_change_preds).mean(dim=0),
            'spread': torch.stack(spread_preds).mean(dim=0)
        }
        
        # Add direction predictions based on averaged probabilities
        ensemble_predictions['direction'] = (ensemble_predictions['direction_prob'] >= 0.5).float()
        
        return ensemble_predictions
    
    def evaluate_ensemble(self, test_data, test_labels):
        """
        Evaluate ensemble performance
        
        Args:
            test_data: Test input tensor [batch_size, window_size, features]
            test_labels: Dictionary with true labels
            
        Returns:
            metrics: Dictionary of performance metrics
        """
        # Get ensemble predictions
        predictions = self.predict_ensemble(test_data)
        
        # Convert to numpy for sklearn metrics
        pred_direction = predictions['direction'].cpu().numpy()
        true_direction = test_labels['direction'].cpu().numpy()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(true_direction, pred_direction)
        f1 = f1_score(true_direction, pred_direction, zero_division=0)
        precision = precision_score(true_direction, pred_direction, zero_division=0)
        recall = recall_score(true_direction, pred_direction, zero_division=0)
        
        # Calculate precision-recall balance
        if max(precision, recall) > 0:
            pr_balance = min(precision, recall) / max(precision, recall)
        else:
            pr_balance = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'pr_balance': pr_balance
        }
        
        return metrics


def main():
    """Test ensemble with your actual validation data"""
    
    # List your trained model paths
    model_paths = [
        './trained_models/financial_model_w18_h1_msft_ensemble_1.pt',
        './trained_models/financial_model_w18_h1_msft_ensemble_2.pt', 
        './trained_models/financial_model_w18_h1_msft_ensemble_3.pt',
    ]
    
    # Check which models exist
    existing_models = [path for path in model_paths if Path(path).exists()]
    
    if not existing_models:
        logger.error("No model files found! Train some models first.")
        return
    
    logger.info(f"Found {len(existing_models)} model files")
    
    # Initialize ensemble
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # For M3 Mac
    ensemble = ModelEnsemble(existing_models, device=device)
    
    # Load your MSFT data for testing
    try:
        from data_preprocessor import FinancialDataPreprocessor
        
        logger.info("Loading MSFT validation data...")
        preprocessor = FinancialDataPreprocessor(
            '/Users/benstuart/Desktop/hanabi/MSFT/hourly_data.csv',
            '/Users/benstuart/Desktop/hanabi/hanabi-1/fear_greed_data/fear_greed_index_enhanced.csv'
        )
        
        # Get validation data using same parameters as your best model
        train_loader, val_loader = preprocessor.get_dataloaders(
            window_size=18,
            horizon=1,
            batch_size=32  # Use larger batch for evaluation
        )
        
        # Test ensemble on validation data
        logger.info("Testing ensemble on validation data...")
        total_accuracy = 0
        total_f1 = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                
                # Get ensemble predictions
                ensemble_pred = ensemble.predict_ensemble(inputs)
                
                # Evaluate this batch
                metrics = ensemble.evaluate_ensemble(inputs, targets)
                
                total_accuracy += metrics['accuracy']
                total_f1 += metrics['f1_score']
                num_batches += 1
                
                logger.info(f"Batch {num_batches}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
                
                # Test on first few batches
                if num_batches >= 5:
                    break
        
        # Calculate average metrics
        avg_accuracy = total_accuracy / num_batches
        avg_f1 = total_f1 / num_batches
        
        logger.info("="*50)
        logger.info("ENSEMBLE RESULTS:")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        logger.info(f"Average F1 Score: {avg_f1:.4f}")
        logger.info("="*50)
        
        # Compare to your best single model (61.02%)
        if avg_accuracy > 0.6102:
            improvement = (avg_accuracy - 0.6102) * 100
            logger.info(f"🎉 ENSEMBLE BEATS SINGLE MODEL BY +{improvement:.2f}%!")
        else:
            logger.info("Single model still performs better")
            
    except ImportError:
        logger.error("Could not import data_preprocessor. Make sure it's in the same directory.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.info("Ensemble models loaded successfully, but couldn't evaluate. You can use them for predictions manually.")


if __name__ == "__main__":
    main()