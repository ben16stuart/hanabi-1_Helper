# rolling_retrain.py
# Model retraining and replacement system for hanabi-1
# Maintains the single best performing model (best_financial_model.pt)

import os
import subprocess
import sys
import tempfile
import re
import shutil
from datetime import datetime
import argparse
from pathlib import Path

# === CONFIGURATION ===

## Training parameters
WINDOW_SIZE = 20
HORIZON = 1
BATCH_SIZE = 32
HIDDEN_DIM = 256
TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.15
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.001 
DIRECTION_WEIGHT = 0.7
FOCAL_GAMMA = 0.6
EPOCHS = 150
PATIENCE = 15
MIN_PRICE_CHANGE = 0.001
DIRECTION_THRESHOLD = 0.55
SEED = 7890

# Model management
BEST_MODEL_NAME = "best_financial_model.pt"
MIN_ACCURACY_THRESHOLD = 0.52  # 52% minimum directional accuracy

def train_new_candidate_model(MAIN_DIR, MODEL_DIR, TICKER, HOURLY_DATA, FNG_DATA, NEW_MODEL_SUFFIX, SAVE_PATH):
    """Train a new candidate model for evaluation"""
    print("[INFO] Training new candidate model...")
    
    candidate_model_name = f"financial_model_{NEW_MODEL_SUFFIX}.pt"
    candidate_model_path = os.path.join(MODEL_DIR, candidate_model_name)

    cmd = [
        "python", os.path.join(MAIN_DIR, "hanabi-1/train_model.py"),
        "--hourly_data", HOURLY_DATA,
        "--fear_greed_data", FNG_DATA,
        "--window_size", str(WINDOW_SIZE),
        "--horizon", str(HORIZON),
        "--batch_size", str(BATCH_SIZE),
        "--hidden_dim", str(HIDDEN_DIM),
        "--transformer_layers", str(TRANSFORMER_LAYERS),
        "--num_heads", str(NUM_HEADS),
        "--dropout", str(DROPOUT),
        "--learning_rate", str(LEARNING_RATE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--direction_weight", str(DIRECTION_WEIGHT),
        "--focal_gamma", str(FOCAL_GAMMA),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--min_price_change", str(MIN_PRICE_CHANGE),
        "--direction_threshold", str(DIRECTION_THRESHOLD),
        "--save_path", SAVE_PATH,
        "--seed", str(SEED),
        "--model_suffix", NEW_MODEL_SUFFIX
    ]
    
    print(f"[INFO] Training candidate model: {candidate_model_name}")
    print("-" * 80)
    
    try:
        # Show real-time training output
        subprocess.run(cmd, check=True)
        print("-" * 80)
        print(f"[INFO] Candidate model training completed")
        
        # Verify the model file was created
        if not os.path.exists(candidate_model_path):
            print(f"[ERROR] Expected candidate model not found: {candidate_model_path}")
            # Try to find the actual created file
            for file in os.listdir(MODEL_DIR):
                if NEW_MODEL_SUFFIX in file and file.endswith('.pt'):
                    actual_path = os.path.join(MODEL_DIR, file)
                    print(f"[INFO] Found candidate model: {actual_path}")
                    return actual_path
            raise FileNotFoundError(f"No candidate model created in {MODEL_DIR}")
            
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"[ERROR] Candidate model training failed with return code: {e.returncode}")
        raise

    return candidate_model_path

def evaluate_single_model(model_path, model_name, MAIN_DIR, HOURLY_DATA, FNG_DATA):
    """Evaluate a single model and return metrics"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file does not exist: {model_path}")
        return None
    
    print(f"[INFO] Evaluating {model_name}: {os.path.basename(model_path)}")
    
    cmd = [
        "python", os.path.join(MAIN_DIR, "hanabi-1/evaluate_ensemble.py"),
        "--model_path", model_path,
        "--hourly_data", HOURLY_DATA,
        "--fear_greed_data", FNG_DATA,
        "--window_size", str(WINDOW_SIZE),
        "--horizon", str(HORIZON),
        "--output_dir", os.path.join(MAIN_DIR, "hanabi-1/evaluation")
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse metrics from the combined output
        full_output = result.stdout + "\n" + result.stderr
        output_lines = full_output.split('\n')
        
        metrics = {
            'accuracy': None,
            'f1': None,
            'precision': None,
            'recall': None
        }
        
        # Extract all relevant metrics
        for line in output_lines:
            if "Direction Accuracy:" in line:
                match = re.search(r'Direction Accuracy:\s*([0-9]*\.?[0-9]+)', line)
                if match:
                    metrics['accuracy'] = float(match.group(1))
            elif "Direction F1 Score:" in line:
                match = re.search(r'Direction F1 Score:\s*([0-9]*\.?[0-9]+)', line)
                if match:
                    metrics['f1'] = float(match.group(1))
            elif "Direction Precision:" in line:
                match = re.search(r'Direction Precision:\s*([0-9]*\.?[0-9]+)', line)
                if match:
                    metrics['precision'] = float(match.group(1))
            elif "Direction Recall:" in line:
                match = re.search(r'Direction Recall:\s*([0-9]*\.?[0-9]+)', line)
                if match:
                    metrics['recall'] = float(match.group(1))
        
        # Validate we got the key metrics
        if metrics['accuracy'] is None or metrics['f1'] is None:
            print(f"[WARNING] Could not extract key metrics for {model_name}")
            print("[DEBUG] Full evaluation output:")
            print("=" * 50)
            print(full_output)
            print("=" * 50)
            return None
        
        print(f"[INFO] {model_name} metrics:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}" if metrics['precision'] else "  - Precision: N/A")
        print(f"  - Recall: {metrics['recall']:.4f}" if metrics['recall'] else "  - Recall: N/A")
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed for {model_name}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

def check_accuracy_threshold(metrics, model_name):
    """Check if model meets minimum accuracy threshold"""
    if metrics is None or metrics['accuracy'] is None:
        print(f"[WARNING] Cannot check threshold for {model_name} - missing accuracy metric")
        return False
    
    accuracy = metrics['accuracy']
    if accuracy < MIN_ACCURACY_THRESHOLD:
        print("=" * 80)
        print("🚨 ACCURACY WARNING 🚨")
        print(f"Model '{model_name}' accuracy ({accuracy:.4f}) is below threshold ({MIN_ACCURACY_THRESHOLD:.4f})")
        print("This model will likely NOT be profitable!")
        print("RECOMMENDATION: Adjust training parameters before using this model")
        print("=" * 80)
        return False
    else:
        print(f"✅ {model_name} accuracy ({accuracy:.4f}) meets threshold ({MIN_ACCURACY_THRESHOLD:.4f})")
        return True

def compare_models(best_metrics, candidate_metrics):
    """Compare two models and determine which is better"""
    if best_metrics is None:
        return "candidate"
    if candidate_metrics is None:
        return "best"
    
    # Get both F1 score and accuracy for comparison
    best_f1 = best_metrics['f1']
    candidate_f1 = candidate_metrics['f1']
    best_accuracy = best_metrics['accuracy']
    candidate_accuracy = candidate_metrics['accuracy']
    
    print(f"[INFO] Model comparison:")
    print(f"  - Best model F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")
    print(f"  - Candidate model F1: {candidate_f1:.4f}, Accuracy: {candidate_accuracy:.4f}")
    
    # Candidate must be better in BOTH F1 score AND directional accuracy
    if candidate_f1 > best_f1 and candidate_accuracy > best_accuracy:
        f1_improvement = candidate_f1 - best_f1
        accuracy_improvement = candidate_accuracy - best_accuracy
        print(f"  ✅ Candidate is better in both metrics:")
        print(f"     - F1 improvement: +{f1_improvement:.4f}")
        print(f"     - Accuracy improvement: +{accuracy_improvement:.4f}")
        return "candidate"
    elif candidate_f1 > best_f1 and candidate_accuracy <= best_accuracy:
        f1_improvement = candidate_f1 - best_f1
        accuracy_difference = best_accuracy - candidate_accuracy
        print(f"  ❌ Candidate has better F1 (+{f1_improvement:.4f}) but worse accuracy (-{accuracy_difference:.4f})")
        print(f"     REJECTED: Both metrics must improve")
        return "best"
    elif candidate_f1 <= best_f1 and candidate_accuracy > best_accuracy:
        f1_difference = best_f1 - candidate_f1
        accuracy_improvement = candidate_accuracy - best_accuracy
        print(f"  ❌ Candidate has better accuracy (+{accuracy_improvement:.4f}) but worse F1 (-{f1_difference:.4f})")
        print(f"     REJECTED: Both metrics must improve")
        return "best"
    else:
        f1_difference = best_f1 - candidate_f1
        accuracy_difference = best_accuracy - candidate_accuracy
        print(f"  ❌ Best model is better in both metrics:")
        print(f"     - F1 advantage: +{f1_difference:.4f}")
        print(f"     - Accuracy advantage: +{accuracy_difference:.4f}")
        return "best"

def backup_and_replace_best_model(candidate_path, BEST_MODEL_PATH, MODEL_DIR):
    """Backup current best model and replace with candidate"""
    if os.path.exists(BEST_MODEL_PATH):
        # Create backup with timestamp
        backup_name = f"best_financial_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        backup_path = os.path.join(MODEL_DIR, backup_name)
        shutil.copy2(BEST_MODEL_PATH, backup_path)
        print(f"[INFO] Backed up previous best model to: {backup_name}")
    
    # Replace with candidate
    shutil.copy2(candidate_path, BEST_MODEL_PATH)
    print(f"[INFO] ✅ NEW BEST MODEL: {os.path.basename(candidate_path)} -> {BEST_MODEL_NAME}")

def cleanup_candidate_model(candidate_path):
    """Remove candidate model if not selected"""
    try:
        if os.path.exists(candidate_path):
            os.remove(candidate_path)
            print(f"[INFO] Removed candidate model: {os.path.basename(candidate_path)}")
    except OSError as e:
        print(f"[WARNING] Could not remove candidate model {candidate_path}: {e}")


def main():
    """Main daily model update workflow"""
    parser = argparse.ArgumentParser(description="Daily Model Retraining for Stock Predictions")
    parser.add_argument("--TICKER", type=str, required=True, help="Stock symbol to train")
    
    args = parser.parse_args()
    
    # Now define all the path-dependent variables
    TICKER = args.TICKER
    DATE_STR = datetime.now().strftime("%Y%m%d")
    MAIN_DIR = str(Path.home() / "Desktop" / "hanabi")
    MODEL_DIR = os.path.join(MAIN_DIR, f"{TICKER}/trained_models")
    HOURLY_DATA = os.path.join(MAIN_DIR, TICKER, "hourly_data.csv")
    FNG_DATA = os.path.join(MAIN_DIR, "sentiment-fear-and-greed/fear_greed_data/fear_greed_index_enhanced.csv")
    SAVE_PATH = os.path.join(MAIN_DIR, f"{TICKER}/trained_models")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
    NEW_MODEL_SUFFIX = f"candidate_{DATE_STR}"

    print("=" * 80)
    print("🔄 DAILY MODEL UPDATE WORKFLOW")
    print(f"📅 Date: {DATE_STR}")
    print(f"🎯 Ticker: {TICKER}")
    print(f"📊 Min Accuracy Threshold: {MIN_ACCURACY_THRESHOLD:.1%}")
    print("=" * 80)
    
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MAIN_DIR, "hanabi-1/evaluation"), exist_ok=True)
    
    try:
        # Step 1: Train new candidate model
        candidate_path = train_new_candidate_model(MAIN_DIR, MODEL_DIR, TICKER, HOURLY_DATA, FNG_DATA, NEW_MODEL_SUFFIX, SAVE_PATH)
        
        # Step 2: Evaluate candidate model
        print("\n" + "=" * 40 + " EVALUATION " + "=" * 40)
        candidate_metrics = evaluate_single_model(candidate_path, "Candidate Model", MAIN_DIR, HOURLY_DATA, FNG_DATA)
        
        if candidate_metrics is None:
            print("[ERROR] Failed to evaluate candidate model. Aborting.")
            cleanup_candidate_model(candidate_path)
            return
        
        # Step 3: Check candidate accuracy threshold
        candidate_passes_threshold = check_accuracy_threshold(candidate_metrics, "Candidate Model")
        
        # Step 4: Evaluate current best model (if it exists)
        best_metrics = None
        if os.path.exists(BEST_MODEL_PATH):
            best_metrics = evaluate_single_model(BEST_MODEL_PATH, "Current Best Model", MAIN_DIR, HOURLY_DATA, FNG_DATA)
            if best_metrics:
                check_accuracy_threshold(best_metrics, "Current Best Model")
        else:
            print("[INFO] No existing best model found. Candidate will become the new best model.")
        
        # Step 5: Compare models and make decision
        print("\n" + "=" * 40 + " DECISION " + "=" * 40)
        
        if not candidate_passes_threshold:
            print("❌ REJECTED: Candidate model does not meet accuracy threshold")
            cleanup_candidate_model(candidate_path)
            if best_metrics and best_metrics['accuracy'] < MIN_ACCURACY_THRESHOLD:
                print("⚠️  WARNING: Your current best model ALSO doesn't meet the threshold!")
                print("🛠️  URGENT: Consider adjusting training parameters")
        else:
            winner = compare_models(best_metrics, candidate_metrics)
            
            if winner == "candidate":
                print("🏆 DECISION: Replacing best model with candidate")
                backup_and_replace_best_model(candidate_path, BEST_MODEL_PATH, MODEL_DIR)
                cleanup_candidate_model(candidate_path)
                print("✅ Model update completed successfully!")
            else:
                print("🏆 DECISION: Keeping current best model")
                cleanup_candidate_model(candidate_path)
                print("✅ Current best model retained")
        
        # Step 6: Final summary
        print("\n" + "=" * 40 + " SUMMARY " + "=" * 40)
        if os.path.exists(BEST_MODEL_PATH):
            final_metrics = evaluate_single_model(BEST_MODEL_PATH, "Active Best Model", MAIN_DIR, HOURLY_DATA, FNG_DATA)
            if final_metrics:
                print(f"🎯 Active model performance: {final_metrics['accuracy']:.4f} accuracy, {final_metrics['f1']:.4f} F1")
                if final_metrics['accuracy'] >= MIN_ACCURACY_THRESHOLD:
                    print("✅ Ready for trading!")
                else:
                    print("⚠️  Model needs improvement before trading!")
            else:
                print("❌ Could not evaluate final active model")
        else:
            print("❌ No active best model available!")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        candidate_files = [f for f in os.listdir(MODEL_DIR) if NEW_MODEL_SUFFIX in f and f.endswith('.pt')]
        for candidate_file in candidate_files:
            candidate_path = os.path.join(MODEL_DIR, candidate_file)
            cleanup_candidate_model(candidate_path)

if __name__ == "__main__":
    main()
