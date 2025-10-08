# continuous_optimization.py - Improved version with local LLM
# Continuous model training optimization using LLM feedback

import os
import json
import subprocess
import sys
import time
from datetime import datetime
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import re
from openai import OpenAI
import argparse
from pathlib import Path

# Get the base directory - defaults to ~/Desktop/hanabi or can be set via environment variable
DEFAULT_BASE = str(Path.home() / "Desktop" / "hanabi")
BASE_DIR = Path(os.getenv('HANABI_BASE_DIR', DEFAULT_BASE))

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# === CONFIGURATION ===
@dataclass
class ModelParams:
    """Model hyperparameters with validation"""
    WINDOW_SIZE: int = 20
    HORIZON: int = 1
    BATCH_SIZE: int = 32
    HIDDEN_DIM: int = 256
    TRANSFORMER_LAYERS: int = 4
    NUM_HEADS: int = 4
    DROPOUT: float = 0.15
    LEARNING_RATE: float = 0.0005
    WEIGHT_DECAY: float = 0.001
    DIRECTION_WEIGHT: float = 0.7
    FOCAL_GAMMA: float = 0.6
    EPOCHS: int = 150
    PATIENCE: int = 15
    MIN_PRICE_CHANGE: float = 0.001
    DIRECTION_THRESHOLD: float = 0.55
    SEED: int = 7890

    def validate(self) -> bool:
        """Validate parameter combinations to prevent training errors"""
        issues = []
        
        # Check if NUM_HEADS divides HIDDEN_DIM evenly
        if self.HIDDEN_DIM % self.NUM_HEADS != 0:
            issues.append(f"HIDDEN_DIM ({self.HIDDEN_DIM}) must be divisible by NUM_HEADS ({self.NUM_HEADS})")
        
        # Check reasonable ranges
        if self.BATCH_SIZE < 8:
            issues.append(f"BATCH_SIZE ({self.BATCH_SIZE}) too small, minimum 8")
        
        if self.WINDOW_SIZE > 50:
            issues.append(f"WINDOW_SIZE ({self.WINDOW_SIZE}) too large, maximum 50")
            
        # Check memory-intensive combinations
        model_complexity = self.HIDDEN_DIM * self.TRANSFORMER_LAYERS * self.BATCH_SIZE
        if model_complexity > 500000:
            issues.append(f"Model too complex (complexity: {model_complexity}), reduce HIDDEN_DIM, LAYERS, or BATCH_SIZE")
        
        if issues:
            print(f"❌ Parameter validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True

    def fix_validation_issues(self) -> 'ModelParams':
        """Auto-fix common validation issues"""
        fixed = ModelParams(**asdict(self))
        
        # Fix NUM_HEADS divisibility
        if fixed.HIDDEN_DIM % fixed.NUM_HEADS != 0:
            for heads in [8, 4, 2, 1]:
                if fixed.HIDDEN_DIM % heads == 0:
                    fixed.NUM_HEADS = heads
                    break
        
        # Ensure minimum batch size
        if fixed.BATCH_SIZE < 8:
            fixed.BATCH_SIZE = 8
        
        # Cap window size
        if fixed.WINDOW_SIZE > 50:
            fixed.WINDOW_SIZE = 50
            
        # Reduce complexity if too high
        model_complexity = fixed.HIDDEN_DIM * fixed.TRANSFORMER_LAYERS * fixed.BATCH_SIZE
        if model_complexity > 500000:
            scale_factor = (500000 / model_complexity) ** 0.5
            fixed.HIDDEN_DIM = int(fixed.HIDDEN_DIM * scale_factor)
            fixed.BATCH_SIZE = max(8, int(fixed.BATCH_SIZE * scale_factor))
            
            for heads in [8, 4, 2, 1]:
                if fixed.HIDDEN_DIM % heads == 0:
                    fixed.NUM_HEADS = heads
                    break
        
        print(f"🔧 Auto-fixed parameters:")
        if fixed.NUM_HEADS != self.NUM_HEADS:
            print(f"   - NUM_HEADS: {self.NUM_HEADS} → {fixed.NUM_HEADS}")
        if fixed.BATCH_SIZE != self.BATCH_SIZE:
            print(f"   - BATCH_SIZE: {self.BATCH_SIZE} → {fixed.BATCH_SIZE}")
        if fixed.HIDDEN_DIM != self.HIDDEN_DIM:
            print(f"   - HIDDEN_DIM: {self.HIDDEN_DIM} → {fixed.HIDDEN_DIM}")
            
        return fixed

@dataclass
class TrainingResult:
    """Results from a training run"""
    iteration: int
    timestamp: str
    params: ModelParams
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    training_time: Optional[float] = None
    error: Optional[str] = None
    success: bool = False

class ContinuousOptimizer:
    def __init__(self,
                 rolling_retrain_script: str = None,
                 max_iterations: int = 50,
                 log_file: str = None,
                 results_file: str = None,
                 local_model: str = "qwen2.5-coder:14b",
                 TICKER: str = None,
                 resume: bool = True,
                 base_dir: Path = None):

        # Store TICKER first
        if TICKER is None:
            raise ValueError("TICKER must be provided")
        self.TICKER = TICKER
        
        # Set base directory (can be overridden, otherwise uses BASE_DIR constant)
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        
        # Create ticker-specific directory
        self.ticker_dir = self.base_dir / self.TICKER / "trained_models"
        
        # Now construct the paths using TICKER
        if log_file is None:
            self.log_file = str(self.ticker_dir / "optimization_log.csv")
        else:
            self.log_file = log_file
            
        if results_file is None:
            self.results_file = str(self.ticker_dir / "optimization_results.json")
        else:
            self.results_file = results_file

        # ✅ Resolve the absolute path to rolling_retrain.py
        if rolling_retrain_script is None:
            # Look for rolling_retrain.py in the same directory as this script
            script_dir = Path(__file__).parent
            self.rolling_retrain_script = str(script_dir / "rolling_retrain.py")
        else:
            self.rolling_retrain_script = str(Path(rolling_retrain_script).resolve())
        
        self.max_iterations = max_iterations
        self.local_model = local_model
        self.resume = resume
        
        self.iteration = 0
        self.results_history: List[TrainingResult] = []
        self.best_result: Optional[TrainingResult] = None
        self.failed_params: List[ModelParams] = []
        
        # Ensure the directory exists
        self.ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Load previous results if resuming
        if self.resume:
            self.load_previous_results()
        
    def setup_logging(self):
        """Setup CSV logging for results"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'timestamp', 'accuracy', 'f1_score', 'precision', 'recall',
                    'training_time', 'success', 'error',
                    'window_size', 'horizon', 'batch_size', 'hidden_dim', 'transformer_layers',
                    'num_heads', 'dropout', 'learning_rate', 'weight_decay', 'direction_weight',
                    'focal_gamma', 'epochs', 'patience', 'min_price_change', 'direction_threshold', 'seed'
                ])
    
    def load_previous_results(self):
        """Load previous optimization results to resume from best configuration"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    previous_results = json.load(f)
                
                if previous_results:
                    print(f"\n📂 Loading previous results from {self.results_file}")
                    
                    for result_dict in previous_results:
                        params_dict = result_dict['params']
                        params = ModelParams(**params_dict)
                        
                        result = TrainingResult(
                            iteration=result_dict['iteration'],
                            timestamp=result_dict['timestamp'],
                            params=params,
                            accuracy=result_dict.get('accuracy'),
                            f1_score=result_dict.get('f1_score'),
                            precision=result_dict.get('precision'),
                            recall=result_dict.get('recall'),
                            training_time=result_dict.get('training_time'),
                            error=result_dict.get('error'),
                            success=result_dict.get('success', False)
                        )
                        
                        self.results_history.append(result)
                        
                        if result.success:
                            self.update_best_result(result)
                    
                    if self.results_history:
                        self.iteration = max(r.iteration for r in self.results_history)
                    
                    print(f"✅ Loaded {len(self.results_history)} previous results")
                    print(f"📊 Starting from iteration {self.iteration + 1}")
                    
                    if self.best_result:
                        print(f"🏆 Best previous result:")
                        print(f"   Iteration: {self.best_result.iteration}")
                        print(f"   Accuracy: {self.best_result.accuracy:.4f}")
                        print(f"   F1 Score: {self.best_result.f1_score:.4f}")
                    
            except Exception as e:
                print(f"⚠️ Could not load previous results: {e}")
                print("   Starting fresh...")
        else:
            print(f"🔍 No previous results found at {self.results_file}")
            print("   Starting fresh optimization...")
    
    def get_initial_params(self) -> ModelParams:
        """Get initial parameters - either from best result or defaults"""
        if self.best_result is not None:
            print(f"\n🎯 Starting with best previous configuration")
            print(f"   (Iteration {self.best_result.iteration}: {self.best_result.accuracy:.4f} accuracy)")
            return ModelParams(**asdict(self.best_result.params))
        else:
            print(f"\n🎯 Starting with default configuration")
            return ModelParams()
    
    def log_result(self, result: TrainingResult):
        """Log result to CSV file"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.iteration, result.timestamp, result.accuracy, result.f1_score,
                result.precision, result.recall, result.training_time, result.success, result.error,
                result.params.WINDOW_SIZE, result.params.HORIZON, result.params.BATCH_SIZE,
                result.params.HIDDEN_DIM, result.params.TRANSFORMER_LAYERS, result.params.NUM_HEADS,
                result.params.DROPOUT, result.params.LEARNING_RATE, result.params.WEIGHT_DECAY,
                result.params.DIRECTION_WEIGHT, result.params.FOCAL_GAMMA, result.params.EPOCHS,
                result.params.PATIENCE, result.params.MIN_PRICE_CHANGE, result.params.DIRECTION_THRESHOLD,
                result.params.SEED
            ])
    
    def save_results(self):
        """Save all results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump([asdict(result) for result in self.results_history], f, indent=2)
    
    def modify_rolling_retrain_params(self, params: ModelParams) -> str:
        """Create a modified version of rolling_retrain.py with new parameters"""
        with open(self.rolling_retrain_script, 'r') as f:
            script_content = f.read()
        
        param_updates = {
            'WINDOW_SIZE': params.WINDOW_SIZE,
            'HORIZON': params.HORIZON,
            'BATCH_SIZE': params.BATCH_SIZE,
            'HIDDEN_DIM': params.HIDDEN_DIM,
            'TRANSFORMER_LAYERS': params.TRANSFORMER_LAYERS,
            'NUM_HEADS': params.NUM_HEADS,
            'DROPOUT': params.DROPOUT,
            'LEARNING_RATE': params.LEARNING_RATE,
            'WEIGHT_DECAY': params.WEIGHT_DECAY,
            'DIRECTION_WEIGHT': params.DIRECTION_WEIGHT,
            'FOCAL_GAMMA': params.FOCAL_GAMMA,
            'EPOCHS': params.EPOCHS,
            'PATIENCE': params.PATIENCE,
            'MIN_PRICE_CHANGE': params.MIN_PRICE_CHANGE,
            'DIRECTION_THRESHOLD': params.DIRECTION_THRESHOLD,
            'SEED': params.SEED
        }
        
        for param_name, param_value in param_updates.items():
            pattern = f'^{param_name} = .*$'
            replacement = f'{param_name} = {param_value}'
            script_content = re.sub(pattern, replacement, script_content, flags=re.MULTILINE)
        
        # Write modified script to ticker directory
        modified_script = str(self.ticker_dir / f"rolling_retrain_modified_{self.iteration}.py")
        with open(modified_script, 'w') as f:
            f.write(script_content)
        
        return modified_script
    
    def run_training(self, params: ModelParams) -> TrainingResult:
        """Run training with given parameters and capture results"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Check current best model timestamp BEFORE training
        model_path = self.ticker_dir / "best_financial_model.pt"
        old_mtime = None
        if model_path.exists():
            old_mtime = model_path.stat().st_mtime
            print(f"📦 Current model timestamp: {datetime.fromtimestamp(old_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        result = TrainingResult(
            iteration=self.iteration,
            timestamp=timestamp,
            params=params
        )
        
        # Validate and fix parameters
        if not params.validate():
            print("🔧 Attempting to auto-fix parameter issues...")
            params = params.fix_validation_issues()
            if not params.validate():
                result.error = "Parameter validation failed even after auto-fix"
                result.success = False
                return result
            result.params = params
        
        try:
            modified_script = self.modify_rolling_retrain_params(params)
            
            print(f"\n{'='*60}")
            print(f"🚀 ITERATION {self.iteration}: Training with new parameters")
            print(f"📅 {timestamp}")
            print(f"{'='*60}")
            print("Parameters:")
            for key, value in asdict(params).items():
                print(f"  {key}: {value}")
            print(f"{'='*60}")
            
            cmd = ["python", modified_script, "--TICKER", self.TICKER]
            process_result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Clean up modified script
            if os.path.exists(modified_script):
                os.remove(modified_script)
            
            training_time = time.time() - start_time
            result.training_time = training_time
            
            full_output = process_result.stdout + "\n" + process_result.stderr
            
            # Check if model file was updated
            if model_path.exists():
                new_mtime = model_path.stat().st_mtime
                if old_mtime and new_mtime > old_mtime:
                    print(f"✅ Model file was updated!")
                    print(f"   New timestamp: {datetime.fromtimestamp(new_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                elif old_mtime:
                    print(f"⚠️  Model file was NOT updated (same timestamp)")
            
            if process_result.returncode == 0:
                metrics = self.parse_training_output(full_output)
                
                if metrics:
                    result.accuracy = metrics.get('accuracy')
                    result.f1_score = metrics.get('f1')
                    result.precision = metrics.get('precision')
                    result.recall = metrics.get('recall')
                    result.success = True
                    
                    print(f"✅ Training completed successfully!")
                    print(f"📊 Results: Accuracy={result.accuracy:.4f}, F1={result.f1_score:.4f}")
                    print(f"⏱️ Training time: {training_time:.1f} seconds")
                else:
                    debug_file = self.ticker_dir / f"debug_output_iter_{self.iteration}.txt"
                    with open(debug_file, 'w') as f:
                        f.write(full_output)
                    
                    result.error = f"Could not parse training metrics. Debug saved to {debug_file}"
                    result.success = False
            else:
                self.failed_params.append(params)
                error_summary = self.extract_error_summary(full_output)
                result.error = f"Training failed: {error_summary}"
                result.success = False
                
                error_file = self.ticker_dir / f"error_output_iter_{self.iteration}.txt"
                with open(error_file, 'w') as f:
                    f.write(full_output)
                
                print(f"❌ Training failed: {error_summary}")
                print(f"   Full error saved to: {error_file}")
                
        except subprocess.TimeoutExpired:
            result.error = "Training timed out after 1 hour"
            result.success = False
        except Exception as e:
            result.error = f"Unexpected error: {str(e)}"
            result.success = False
        
        return result
    
    def extract_error_summary(self, output: str) -> str:
        """Extract a concise error summary from training output"""
        lines = output.split('\n')
        
        error_patterns = [
            r'ValueError: (.+)',
            r'RuntimeError: (.+)',
            r'CUDA out of memory',
            r'Expected more than 1 value per channel',
            r'CalledProcessError: (.+)',
        ]
        
        for line in lines:
            for pattern in error_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1) if match.groups() else match.group(0)
        
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:200]
        
        return "Unknown error"
    
    def parse_training_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse training metrics from script output"""
        metrics = {}
        
        patterns = {
            'accuracy': r'Active model performance:\s*([0-9]*\.?[0-9]+)\s*accuracy',
            'f1': r'accuracy,\s*([0-9]*\.?[0-9]+)\s*F1',
            'accuracy_alt': r'-\s*Accuracy:\s*([0-9]*\.?[0-9]+)',
            'f1_alt': r'-\s*F1 Score:\s*([0-9]*\.?[0-9]+)',
            'precision': r'Direction Precision:\s*([0-9]*\.?[0-9]+)',
            'recall': r'Direction Recall:\s*([0-9]*\.?[0-9]+)',
        }
        
        for line in output.split('\n'):
            if 'accuracy' not in metrics:
                for pattern_name in ['accuracy', 'accuracy_alt']:
                    match = re.search(patterns[pattern_name], line, re.IGNORECASE)
                    if match:
                        metrics['accuracy'] = float(match.group(1))
                        break
            
            if 'f1' not in metrics:
                for pattern_name in ['f1', 'f1_alt']:
                    match = re.search(patterns[pattern_name], line, re.IGNORECASE)
                    if match:
                        metrics['f1'] = float(match.group(1))
                        break
            
            if 'precision' not in metrics:
                match = re.search(patterns['precision'], line, re.IGNORECASE)
                if match:
                    metrics['precision'] = float(match.group(1))
            
            if 'recall' not in metrics:
                match = re.search(patterns['recall'], line, re.IGNORECASE)
                if match:
                    metrics['recall'] = float(match.group(1))
        
        return metrics if len(metrics) >= 2 else None
    
    def create_llm_prompt(self, recent_results: List[TrainingResult]) -> str:
        """Create prompt for LLM analysis with failure information"""
        prompt = """You are an expert in deep learning and financial time series forecasting. 
        I am training transformer-based models for stock price direction prediction. 
        You will analyze recent training outcomes and recommend the next parameter set to test. 
        Base your reasoning on *exploration vs exploitation balance*, validation consistency, and potential overfitting or underfitting patterns.

        RECENT RESULTS:
        """
        for result in recent_results[-5:]:
            if result.success:
                prompt += f"""
        Iteration {result.iteration}: ✅ SUCCESS
        - Accuracy: {result.accuracy:.4f}
        - F1 Score: {result.f1_score:.4f}  
        - Time: {result.training_time:.1f}s
        - Parameters: Window={result.params.WINDOW_SIZE}, Batch={result.params.BATCH_SIZE}, Hidden={result.params.HIDDEN_DIM}, Layers={result.params.TRANSFORMER_LAYERS}, Heads={result.params.NUM_HEADS}, LR={result.params.LEARNING_RATE}, Dropout={result.params.DROPOUT}
        """
            else:
                prompt += f"""
        Iteration {result.iteration}: ❌ FAILED - {result.error[:100]}
        - Parameters: Window={result.params.WINDOW_SIZE}, Batch={result.params.BATCH_SIZE}, Hidden={result.params.HIDDEN_DIM}, Layers={result.params.TRANSFORMER_LAYERS}, Heads={result.params.NUM_HEADS}
        """

        if self.best_result:
            prompt += f"""
        BEST RESULT SO FAR:
        - Accuracy: {self.best_result.accuracy:.4f}, F1: {self.best_result.f1_score:.4f}
        - Best Parameters: {asdict(self.best_result.params)}
        """

        if self.failed_params:
            prompt += """
        FAILED PARAMETER COMBINATIONS TO AVOID:
        """
            for params in self.failed_params[-3:]:
                prompt += f"- Hidden={params.HIDDEN_DIM}, Layers={params.TRANSFORMER_LAYERS}, Batch={params.BATCH_SIZE}, Window={params.WINDOW_SIZE}\n"

        prompt += """
        PARAMETER CONSTRAINTS (IMPORTANT):
        - HIDDEN_DIM must be divisible by NUM_HEADS evenly.
        - Avoid very large models (Hidden*Layers*Batch < 500,000 complexity).
        - BATCH_SIZE should be >= 8 to avoid normalization issues.
        - WINDOW_SIZE should be <= 50 for memory efficiency.

        SAFE PARAMETER RANGES:
        - WINDOW_SIZE: 15-40 (sequence length)
        - HORIZON: 1 (keep prediction horizon at 1)
        - BATCH_SIZE: 16-64
        - HIDDEN_DIM: 128-512 (must divide evenly by NUM_HEADS)
        - TRANSFORMER_LAYERS: 2-6
        - NUM_HEADS: 2, 4, 8, 16
        - DROPOUT: 0.1-0.25
        - LEARNING_RATE: 0.0001-0.002
        - WEIGHT_DECAY: 0.0001-0.01
        - FOCAL_GAMMA: 0.5-2.0
        - DIRECTION_WEIGHT: 0.5-1.0
        - EPOCHS: 100-250

        STRATEGIC ADJUSTMENT RULES:
        - If the last 5 successful runs show **less than 0.2% improvement**, increase search diversity by changing 2-3 parameters more significantly (20-50% from last best).
        - If accuracy is volatile or unstable, prioritize **stabilization**: smaller learning rate changes (x0.5-x1.2), higher dropout, or slightly larger batch.
        - If all recent models underfit, increase model capacity (Hidden_DIM or Layers).
        - If all overfit, lower model capacity or raise dropout.
        - If plateaued for >50 iterations, propose **a broader search range** (e.g., test new Window sizes or alternate Head counts).
        - Always maintain NUM_HEADS dividing HIDDEN_DIM evenly.

        REPORTING REQUIREMENTS:
        1. Briefly classify the recent performance trend as one of: ["improving", "plateaued", "overfitting", "unstable"].
        2. State whether to make **small incremental changes** or **larger exploratory jumps** next.
        3. Suggest exactly ONE new parameter configuration as JSON.
        4. Include small justification (<40 words) in a "reason" field explaining your rationale.

        Respond ONLY with a JSON object in this format:
        {
            "trend": "plateaued",
            "strategy": "explore_larger_changes",
            "reason": "Recent runs show diminishing improvement; increasing window and hidden dim may capture longer context.",
            "WINDOW_SIZE": 30,
            "HORIZON": 1,
            "BATCH_SIZE": 32,
            "HIDDEN_DIM": 384,
            "TRANSFORMER_LAYERS": 4,
            "NUM_HEADS": 8,
            "DROPOUT": 0.18,
            "LEARNING_RATE": 0.0006,
            "WEIGHT_DECAY": 0.001,
            "DIRECTION_WEIGHT": 0.7,
            "FOCAL_GAMMA": 0.8,
            "EPOCHS": 150,
            "PATIENCE": 15,
            "MIN_PRICE_CHANGE": 0.001,
            "DIRECTION_THRESHOLD": 0.55,
            "SEED": 7890
        }
        """
        return prompt
    
    def query_ollama(self, prompt: str, model: str = "qwen2.5-coder:32b") -> Optional[str]:
        """Query local Ollama instance"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama API error: {e}")
            return None
    
    def get_llm_suggestions(self, recent_results: List[TrainingResult]) -> Optional[ModelParams]:
        """Get parameter suggestions from LLM"""
        prompt = self.create_llm_prompt(recent_results)
        
        print(f"\n🤖 Querying local LLM ({self.local_model}) for parameter suggestions...")
        
        response = self.query_ollama(prompt, self.local_model)
        
        if not response:
            print(f"❌ No LLM response available")
            return self.get_fallback_params()
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                full_response = json.loads(json_str)
                
                # ✅ Extract and display analysis fields (trend, strategy, reason)
                trend = full_response.get('trend', 'unknown')
                strategy = full_response.get('strategy', 'unknown')
                reason = full_response.get('reason', 'No reason provided')
                
                print(f"\n📊 LLM Analysis:")
                print(f"   📈 Trend: {trend}")
                print(f"   🎯 Strategy: {strategy}")
                print(f"   💡 Reason: {reason}")
                
                # ✅ Filter to only ModelParams fields
                valid_param_keys = set(ModelParams.__dataclass_fields__.keys())
                params_dict = {k: v for k, v in full_response.items() if k in valid_param_keys}
                
                new_params = ModelParams(**params_dict)
                
                if not new_params.validate():
                    print(f"⚠️ LLM suggested invalid parameters, auto-fixing...")
                    new_params = new_params.fix_validation_issues()
                
                print(f"✅ LLM suggested new parameters")
                return new_params
            else:
                print(f"❌ Could not extract JSON from LLM response")
                return self.get_fallback_params()
                
        except Exception as e:
            print(f"❌ Error parsing LLM response: {e}")
            return self.get_fallback_params()
    
    def get_fallback_params(self) -> ModelParams:
        """Get conservative fallback parameters"""
        if self.best_result:
            fallback = ModelParams(**asdict(self.best_result.params))
            fallback.LEARNING_RATE *= 0.9
            fallback.DROPOUT = min(0.3, fallback.DROPOUT + 0.02)
            return fallback
        return ModelParams()
    
    def update_best_result(self, result: TrainingResult):
        """Update best result if current is better"""
        if not result.success:
            return
        
        if (self.best_result is None or 
            result.accuracy > self.best_result.accuracy or 
            (result.accuracy == self.best_result.accuracy and result.f1_score > self.best_result.f1_score)):
            self.best_result = result
            print(f"🏆 NEW BEST RESULT! Accuracy: {result.accuracy:.4f}, F1: {result.f1_score:.4f}")
    
    def run_optimization_loop(self):
        """Main optimization loop"""
        print(f"""
🎯 CONTINUOUS MODEL OPTIMIZATION {'(RESUMING)' if self.resume and self.results_history else '(STARTING FRESH)'}
📊 Max iterations: {self.max_iterations}
📁 Base directory: {self.base_dir}
📝 Logging to: {self.log_file}
💾 Results saved to: {self.results_file}
🤖 LLM: {self.local_model}
📊 Ticker: {self.TICKER}
""")
        
        current_params = self.get_initial_params()
        consecutive_failures = 0
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            result = self.run_training(current_params)
            self.results_history.append(result)
            self.log_result(result)
            self.save_results()
            self.update_best_result(result)
            
            if result.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            
            if consecutive_failures >= 3:
                print("⚠️ Too many failures. Reverting to best parameters...")
                current_params = self.get_fallback_params()
                consecutive_failures = 0
            
            if self.iteration < self.max_iterations:
                new_params = self.get_llm_suggestions(self.results_history)
                if new_params:
                    current_params = new_params
                time.sleep(3)
        
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        if self.best_result:
            best_params_file = self.ticker_dir / f"best_params_{self.TICKER}.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    'iteration': self.best_result.iteration,
                    'accuracy': self.best_result.accuracy,
                    'f1_score': self.best_result.f1_score,
                    'parameters': asdict(self.best_result.params)
                }, f, indent=2)
            print(f"💾 Best parameters saved to: {best_params_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous Optimization for Stock Predictions")
    parser.add_argument("--TICKER", type=str, required=True, help="Stock symbol")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--local-model", type=str, default="qwen2.5-coder:14b")
    parser.add_argument("--base-dir", type=str, help="Base directory (default: current directory or HANABI_BASE_DIR env var)")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else None
    
    optimizer = ContinuousOptimizer(
        max_iterations=args.max_iterations,
        local_model=args.local_model,
        TICKER=args.TICKER,
        resume=not args.no_resume,
        base_dir=base_dir
    )
    
    try:
        optimizer.run_optimization_loop()
    except KeyboardInterrupt:
        print(f"\n🛑 Optimization stopped by user")

if __name__ == "__main__":
    main()
