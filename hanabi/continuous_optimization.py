# continuous_optimization.py - Improved version with better error handling
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

# === CONFIGURATION ===
@dataclass
class ModelParams:
    """Model hyperparameters with validation"""
    WINDOW_SIZE: int = 20
    HORIZON: int = 2
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
        if model_complexity > 500000:  # Arbitrary threshold
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
            # Find the largest valid NUM_HEADS
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
            # Scale down proportionally
            scale_factor = (500000 / model_complexity) ** 0.5
            fixed.HIDDEN_DIM = int(fixed.HIDDEN_DIM * scale_factor)
            fixed.BATCH_SIZE = max(8, int(fixed.BATCH_SIZE * scale_factor))
            
            # Ensure NUM_HEADS still divides HIDDEN_DIM
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
                 rolling_retrain_script: str = "rolling_retrain.py",
                 max_iterations: int = 50,
                 log_file: str = "optimization_log.csv",
                 results_file: str = "optimization_results.json",
                 openai_client: OpenAI = None,
                 claude_api_key: str = None):
        
        self.rolling_retrain_script = rolling_retrain_script
        self.max_iterations = max_iterations
        self.log_file = log_file
        self.results_file = results_file
        self.openai_client = openai_client
        self.claude_api_key = claude_api_key
        
        self.iteration = 0
        self.results_history: List[TrainingResult] = []
        self.best_result: Optional[TrainingResult] = None
        self.failed_params: List[ModelParams] = []  # Track failed parameter sets
        
        # Initialize logging
        self.setup_logging()
        
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
        # Read the original script
        with open(self.rolling_retrain_script, 'r') as f:
            script_content = f.read()
        
        # Update parameters in the script content
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
        
        # Replace parameter values in the script
        for param_name, param_value in param_updates.items():
            pattern = f'^{param_name} = .*$'
            replacement = f'{param_name} = {param_value}'
            script_content = re.sub(pattern, replacement, script_content, flags=re.MULTILINE)
        
        # Write modified script
        modified_script = f"rolling_retrain_modified_{self.iteration}.py"
        with open(modified_script, 'w') as f:
            f.write(script_content)
        
        return modified_script
    
    def run_training(self, params: ModelParams) -> TrainingResult:
        """Run training with given parameters and capture results"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
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
            result.params = params  # Update with fixed params
        
        try:
            # Create modified script with new parameters
            modified_script = self.modify_rolling_retrain_params(params)
            
            print(f"\n{'='*60}")
            print(f"🚀 ITERATION {self.iteration}: Training with new parameters")
            print(f"📅 {timestamp}")
            print(f"{'='*60}")
            print("Parameters:")
            for key, value in asdict(params).items():
                print(f"  {key}: {value}")
            print(f"{'='*60}")
            
            # Run the modified script
            cmd = ["python", modified_script]
            process_result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Clean up modified script
            if os.path.exists(modified_script):
                os.remove(modified_script)
            
            training_time = time.time() - start_time
            result.training_time = training_time
            
            # Save full output for debugging
            full_output = process_result.stdout + "\n" + process_result.stderr
            
            if process_result.returncode == 0:
                # Parse results from output
                metrics = self.parse_training_output(full_output)
                
                if metrics:
                    result.accuracy = metrics.get('accuracy')
                    result.f1_score = metrics.get('f1')
                    result.precision = metrics.get('precision')
                    result.recall = metrics.get('recall')
                    result.success = True
                    
                    print(f"✅ Training completed successfully!")
                    print(f"📊 Results: Accuracy={result.accuracy:.4f}, F1={result.f1_score:.4f}")
                    print(f"⏱️  Training time: {training_time:.1f} seconds")
                else:
                    # Save output to file for debugging
                    debug_file = f"debug_output_iter_{self.iteration}.txt"
                    with open(debug_file, 'w') as f:
                        f.write(full_output)
                    
                    result.error = f"Could not parse training metrics from output. Debug saved to {debug_file}"
                    result.success = False
                    print(f"❌ Could not parse training results - check {debug_file}")
            else:
                # Training script failed
                self.failed_params.append(params)
                
                # Extract key error information
                error_summary = self.extract_error_summary(full_output)
                result.error = f"Training failed: {error_summary}"
                result.success = False
                
                # Save detailed error output
                error_file = f"error_output_iter_{self.iteration}.txt"
                with open(error_file, 'w') as f:
                    f.write(full_output)
                
                print(f"❌ Training failed: {error_summary}")
                print(f"   Full error saved to: {error_file}")
                
        except subprocess.TimeoutExpired:
            result.error = "Training timed out after 1 hour"
            result.success = False
            print(f"❌ Training timed out")
        except Exception as e:
            result.error = f"Unexpected error: {str(e)}"
            result.success = False
            print(f"❌ Unexpected error: {e}")
        
        return result
    
    def extract_error_summary(self, output: str) -> str:
        """Extract a concise error summary from training output"""
        lines = output.split('\n')
        
        # Look for common error patterns
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
        
        # Fallback: return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()[:200]  # Limit length
        
        return "Unknown error"
    
    def parse_training_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse training metrics from script output - improved version"""
        metrics = {}
        
        # Look for metrics in the output - Updated patterns
        patterns = {
            # Match: "🎯 Active model performance: 0.5668 accuracy, 0.4521 F1"
            'accuracy': r'Active model performance:\s*([0-9]*\.?[0-9]+)\s*accuracy',
            'f1': r'accuracy,\s*([0-9]*\.?[0-9]+)\s*F1',
            
            # Alternative patterns for individual metric lines:
            'accuracy_alt': r'-\s*Accuracy:\s*([0-9]*\.?[0-9]+)',
            'f1_alt': r'-\s*F1 Score:\s*([0-9]*\.?[0-9]+)',
            'precision': r'Direction Precision:\s*([0-9]*\.?[0-9]+)',
            'recall': r'Direction Recall:\s*([0-9]*\.?[0-9]+)',
            
            # Additional patterns for evaluation output:
            'accuracy_eval': r'Direction Accuracy:\s*([0-9]*\.?[0-9]+)',
            'f1_eval': r'Direction F1 Score:\s*([0-9]*\.?[0-9]+)',
        }
        
        # Try to extract metrics using all patterns
        for line in output.split('\n'):
            # Try accuracy patterns
            if 'accuracy' not in metrics:
                for pattern_name in ['accuracy', 'accuracy_alt', 'accuracy_eval']:
                    match = re.search(patterns[pattern_name], line, re.IGNORECASE)
                    if match:
                        metrics['accuracy'] = float(match.group(1))
                        break
            
            # Try F1 patterns
            if 'f1' not in metrics:
                for pattern_name in ['f1', 'f1_alt', 'f1_eval']:
                    match = re.search(patterns[pattern_name], line, re.IGNORECASE)
                    if match:
                        metrics['f1'] = float(match.group(1))
                        break
            
            # Try precision pattern
            if 'precision' not in metrics:
                match = re.search(patterns['precision'], line, re.IGNORECASE)
                if match:
                    metrics['precision'] = float(match.group(1))
            
            # Try recall pattern
            if 'recall' not in metrics:
                match = re.search(patterns['recall'], line, re.IGNORECASE)
                if match:
                    metrics['recall'] = float(match.group(1))
        
        # Debug output to help troubleshoot
        if len(metrics) < 2:
            print(f"🔍 DEBUG: Only found {len(metrics)} metrics: {metrics}")
            print("🔍 DEBUG: Sample output lines:")
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['accuracy', 'f1', 'precision', 'recall', 'performance']):
                    print(f"   Line {i}: {line}")
        
        return metrics if len(metrics) >= 2 else None
    
    def create_llm_prompt(self, recent_results: List[TrainingResult]) -> str:
        """Create prompt for LLM analysis with failure information"""
        prompt = """You are an expert in deep learning and financial time series forecasting. I'm training transformer-based models for stock price direction prediction and need your help optimizing hyperparameters.

RECENT RESULTS:
"""
        
        for result in recent_results[-5:]:  # Last 5 results
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
        
        # Add information about failed parameter combinations
        if self.failed_params:
            prompt += f"""
FAILED PARAMETER COMBINATIONS TO AVOID:
"""
            for params in self.failed_params[-3:]:  # Last 3 failed attempts
                prompt += f"- Hidden={params.HIDDEN_DIM}, Layers={params.TRANSFORMER_LAYERS}, Batch={params.BATCH_SIZE}, Window={params.WINDOW_SIZE}\n"
        
        prompt += """
PARAMETER CONSTRAINTS (IMPORTANT):
- HIDDEN_DIM must be divisible by NUM_HEADS evenly
- Avoid very large models (Hidden*Layers*Batch < 500,000 complexity)
- BATCH_SIZE should be >= 8 to avoid batch normalization issues
- WINDOW_SIZE should be <= 50 for memory efficiency
- Conservative changes work better than dramatic ones

SAFE PARAMETER RANGES:
- WINDOW_SIZE: 15-40 (sequence length)
- HORIZON: 1-3 (prediction horizon)  
- BATCH_SIZE: 16-64 (avoid too large)
- HIDDEN_DIM: 128-512 (must divide evenly by NUM_HEADS)
- TRANSFORMER_LAYERS: 2-6 (avoid too deep)
- NUM_HEADS: 2, 4, 8, 16 (must divide HIDDEN_DIM)
- DROPOUT: 0.1-0.25
- LEARNING_RATE: 0.0001-0.002
- EPOCHS: 100-250

Based on the results, suggest parameters that:
1. Avoid the failed combinations above
2. Make incremental improvements from successful runs
3. Stay within safe complexity limits
4. Ensure NUM_HEADS divides HIDDEN_DIM evenly

Respond with ONLY a JSON object:
{
    "WINDOW_SIZE": 25,
    "HORIZON": 2,
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 256,
    "TRANSFORMER_LAYERS": 4,
    "NUM_HEADS": 8,
    "DROPOUT": 0.15,
    "LEARNING_RATE": 0.0008,
    "WEIGHT_DECAY": 0.001,
    "DIRECTION_WEIGHT": 0.7,
    "FOCAL_GAMMA": 0.6,
    "EPOCHS": 150,
    "PATIENCE": 15,
    "MIN_PRICE_CHANGE": 0.001,
    "DIRECTION_THRESHOLD": 0.55,
    "SEED": 7890
}
"""
        return prompt
    
    def query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI GPT for parameter suggestions"""
        if not self.openai_client:
            return None
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return None
    
    def query_claude(self, prompt: str) -> Optional[str]:
        """Query Claude for parameter suggestions"""
        if not self.claude_api_key:
            return None
        
        import requests
            
        headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", 
                                   headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()
        except Exception as e:
            print(f"❌ Claude API error: {e}")
            return None
    
    def get_llm_suggestions(self, recent_results: List[TrainingResult]) -> Optional[ModelParams]:
        """Get parameter suggestions from LLM"""
        prompt = self.create_llm_prompt(recent_results)
        
        print(f"\n🤖 Querying LLM for parameter suggestions...")
        
        # Try OpenAI first, then Claude
        response = None
        if self.openai_client:
            response = self.query_openai(prompt)
        
        if not response and self.claude_api_key:
            response = self.query_claude(prompt)
        
        if not response:
            print(f"❌ No LLM response available")
            return self.get_fallback_params()
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                params_dict = json.loads(json_str)
                
                new_params = ModelParams(**params_dict)
                
                # Validate the suggested parameters
                if not new_params.validate():
                    print(f"⚠️  LLM suggested invalid parameters, auto-fixing...")
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
        """Get conservative fallback parameters when LLM fails"""
        if self.best_result:
            # Make small modifications to best known parameters
            best_params = self.best_result.params
            fallback = ModelParams(**asdict(best_params))
            
            # Make small conservative changes
            fallback.LEARNING_RATE *= 0.9  # Slightly lower learning rate
            fallback.DROPOUT = min(0.3, fallback.DROPOUT + 0.02)  # Slightly more dropout
            
            print("🔄 Using conservative fallback based on best result")
            return fallback
        else:
            # Use default conservative parameters
            print("🔄 Using default conservative parameters")
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
        """Main optimization loop with better error handling"""
        print(f"""
🎯 CONTINUOUS MODEL OPTIMIZATION STARTED
📊 Max iterations: {self.max_iterations}
📝 Logging to: {self.log_file}
💾 Results saved to: {self.results_file}
🤖 LLM APIs: {'OpenAI' if self.openai_client else ''} {'Claude' if self.claude_api_key else ''}
""")
        
        # Start with default parameters
        current_params = ModelParams()
        consecutive_failures = 0
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            # Run training
            result = self.run_training(current_params)
            
            # Log and save results
            self.results_history.append(result)
            self.log_result(result)
            self.save_results()
            self.update_best_result(result)
            
            print(f"\n📈 ITERATION {self.iteration} SUMMARY:")
            print(f"   Success: {result.success}")
            if result.success:
                print(f"   Accuracy: {result.accuracy:.4f}")
                print(f"   F1 Score: {result.f1_score:.4f}")
                consecutive_failures = 0  # Reset failure count
            else:
                print(f"   Error: {result.error}")
                consecutive_failures += 1
            
            if self.best_result:
                print(f"   Best so far: {self.best_result.accuracy:.4f} accuracy (iteration {self.best_result.iteration})")
            
            # Check for too many consecutive failures
            if consecutive_failures >= 3:
                print("⚠️  Too many consecutive failures. Reverting to best known parameters...")
                if self.best_result:
                    current_params = ModelParams(**asdict(self.best_result.params))
                else:
                    current_params = ModelParams()  # Reset to defaults
                consecutive_failures = 0
            
            # Get LLM suggestions for next iteration
            if self.iteration < self.max_iterations:
                new_params = self.get_llm_suggestions(self.results_history)
                if new_params:
                    current_params = new_params
                else:
                    print("⚠️  Using fallback parameters")
                
                # Wait a bit between iterations
                print(f"\n⏸️  Waiting 30 seconds before next iteration...")
                time.sleep(30)
        
        # Final summary
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print(f"Total iterations: {self.max_iterations}")
        successful_runs = len([r for r in self.results_history if r.success])
        print(f"Successful runs: {successful_runs}/{self.max_iterations}")
        
        if self.best_result:
            print(f"\n🏆 BEST RESULT:")
            print(f"   Iteration: {self.best_result.iteration}")
            print(f"   Accuracy: {self.best_result.accuracy:.4f}")
            print(f"   F1 Score: {self.best_result.f1_score:.4f}")
            print(f"   Parameters: {asdict(self.best_result.params)}")

def main():
    """Main function to run continuous optimization"""
    # Configuration - Replace with your actual API key
    OPENAI_API_KEY = "sk-proj-PASTE_YOUR_KEY_HERE"
    
    # Create OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Optional Claude API key (can be None if you only want to use OpenAI)
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")  # Optional
    
    optimizer = ContinuousOptimizer(
        rolling_retrain_script="rolling_retrain.py",  # Your original training script
        max_iterations=25,  # Adjust as needed
        openai_client=openai_client,
        claude_api_key=CLAUDE_API_KEY
    )
    
    try:
        optimizer.run_optimization_loop()
    except KeyboardInterrupt:
        print(f"\n🛑 Optimization stopped by user")
        if optimizer.best_result:
            print(f"Best result so far: {optimizer.best_result.accuracy:.4f} accuracy")

if __name__ == "__main__":
    main()
