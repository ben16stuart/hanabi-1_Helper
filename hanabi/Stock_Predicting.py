import gradio as gr
import subprocess
import os
import sys
import json
import pandas as pd
import re
from datetime import datetime, date
import pytz
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Base configuration
BASE_PATH = str(Path.home() / "Desktop" / "hanabi")
FEAR_GREED_DATA_PATH = os.path.join(BASE_PATH, "sentiment-fear-and-greed/fear_greed_data/fear_greed_index_enhanced.csv")   
OUTPUT_DIR = "./predictions"

def get_stock_paths(ticker):
    """Get the model and data paths for a specific stock ticker"""
    return {
        "model_path": f"{BASE_PATH}/{ticker}/trained_models/best_financial_model.pt",
        "hourly_data_path": f"{BASE_PATH}/{ticker}/hourly_data.csv",
        "download_script": f"{BASE_PATH}/{ticker}/download_data.py"
    }

def get_latest_data_info(csv_path):
    """Extract latest epoch time and stock price from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return "No data found in CSV file", None, None
        
        epoch_columns = ['epoch', 'timestamp', 'Epoch', 'Timestamp', 'time', 'Time']
        epoch_col = None
        
        for col in epoch_columns:
            if col in df.columns:
                epoch_col = col
                break
        
        if epoch_col is None:
            return f"No epoch/timestamp column found. Available columns: {list(df.columns)}", None, None
        
        df_sorted = df.sort_values(by=epoch_col, ascending=False)
        latest_row = df_sorted.iloc[0]
        latest_epoch = latest_row[epoch_col]
        
        price_columns = ['close', 'Close', 'price', 'Price', 'last', 'Last']
        price_col = None
        
        for col in price_columns:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            if price_cols:
                price_col = price_cols[0]
        
        latest_price = latest_row[price_col] if price_col else "Price column not found"
        mst_time = convert_epoch_to_mst(latest_epoch)
        
        return mst_time, latest_price, latest_epoch
        
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None, None

def convert_epoch_to_mst(epoch_timestamp):
    """Convert epoch timestamp to MST HH:MM AM/PM format"""
    try:
        if epoch_timestamp > 10000000000:
            epoch_timestamp = epoch_timestamp / 1000
        
        utc_dt = datetime.fromtimestamp(epoch_timestamp, tz=pytz.UTC)
        mst_tz = pytz.timezone('US/Mountain')
        mst_dt = utc_dt.astimezone(mst_tz)
        formatted_time = mst_dt.strftime("%I:%M %p")
        formatted_date = mst_dt.strftime("%Y-%m-%d")
        
        return f"{formatted_date} {formatted_time} MST"
        
    except Exception as e:
        return f"Error converting time: {str(e)}"

def interpret_future_prediction(prediction_data):
    """Interpret the future prediction and provide trading suggestion"""
    try:
        direction_prob = prediction_data.get('direction_probability', 0.5)
        predicted_direction = prediction_data.get('predicted_direction', 'NEUTRAL')
        confidence = prediction_data.get('confidence', 0)
        expected_volatility = prediction_data.get('expected_volatility', 0)
        expected_price_change = prediction_data.get('expected_price_change', 0)
        expected_spread = prediction_data.get('expected_spread', 0)
        
        signal_strength = confidence * abs(direction_prob - 0.5) * 2
        
        if predicted_direction == 'UP':
            if signal_strength > 0.4 and confidence > 0.4:
                if direction_prob > 0.8:
                    suggestion = "🟢 STRONG BUY"
                    reasoning = f"Strong upward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
                elif direction_prob > 0.65:
                    suggestion = "🟢 BUY"
                    reasoning = f"Good upward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
                else:
                    suggestion = "🟢 SOFT BUY"
                    reasoning = f"Moderate upward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
            elif signal_strength > 0.2:
                suggestion = "🟡 WEAK BUY"
                reasoning = f"Weak upward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
            else:
                suggestion = "🟡 HOLD"
                reasoning = f"Unclear upward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
                
        elif predicted_direction == 'DOWN':
            if signal_strength > 0.4 and confidence > 0.4:
                if direction_prob < 0.2:
                    suggestion = "🔴 STRONG SELL"
                    reasoning = f"Strong downward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
                elif direction_prob < 0.35:
                    suggestion = "🔴 SELL"
                    reasoning = f"Good downward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
                else:
                    suggestion = "🔴 SOFT SELL"
                    reasoning = f"Moderate downward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
            elif signal_strength > 0.2:
                suggestion = "🟡 WEAK SELL"
                reasoning = f"Weak downward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
            else:
                suggestion = "🟡 HOLD"
                reasoning = f"Unclear downward signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
        else:
            suggestion = "🟡 HOLD"
            reasoning = f"Neutral/unclear signal: {direction_prob:.1%} probability, {confidence:.1%} confidence"
        
        context_flags = []
        
        if abs(expected_volatility) > 0.5:
            context_flags.append("⚠️ High volatility expected")
        elif abs(expected_volatility) > 0.3:
            context_flags.append("📊 Moderate volatility expected")
        else:
            context_flags.append("📉 Low volatility expected")
        
        if abs(expected_price_change) > 0.05:
            context_flags.append(f"📈 Large price move expected ({expected_price_change:+.2%})")
        elif abs(expected_price_change) > 0.02:
            context_flags.append(f"📊 Moderate price move expected ({expected_price_change:+.2%})")
        else:
            context_flags.append(f"📉 Small price move expected ({expected_price_change:+.2%})")
        
        if abs(expected_spread) > 0.03:
            context_flags.append("💰 Wide spread expected (higher trading costs)")
        
        if context_flags:
            reasoning += " | " + " | ".join(context_flags)
        
        return suggestion, reasoning, signal_strength
        
    except Exception as e:
        return "🟡 HOLD", f"Error interpreting prediction: {str(e)}", 0

def parse_future_prediction_output(output_text):
    """Parse the future prediction output from the log format"""
    try:
        prediction_data = {}
        
        timestamp_match = re.search(r'Timestamp:\s*([\d\-:\s]+)', output_text)
        if timestamp_match:
            prediction_data['timestamp'] = timestamp_match.group(1).strip()
        
        prob_match = re.search(r'Direction probability:\s*([\d\.]+)', output_text)
        if prob_match:
            prediction_data['direction_probability'] = float(prob_match.group(1))
        
        direction_match = re.search(r'Predicted direction:\s*(\w+)', output_text)
        if direction_match:
            prediction_data['predicted_direction'] = direction_match.group(1)
        
        confidence_match = re.search(r'Confidence:\s*([\d\.\-]+)', output_text)
        if confidence_match:
            prediction_data['confidence'] = float(confidence_match.group(1))
        
        volatility_match = re.search(r'Expected volatility:\s*([\d\.\-]+)', output_text)
        if volatility_match:
            prediction_data['expected_volatility'] = float(volatility_match.group(1))
        
        price_change_match = re.search(r'Expected price change:\s*([\d\.\-]+)', output_text)
        if price_change_match:
            prediction_data['expected_price_change'] = float(price_change_match.group(1))
        
        spread_match = re.search(r'Expected spread:\s*([\d\.\-]+)', output_text)
        if spread_match:
            prediction_data['expected_spread'] = float(spread_match.group(1))
        
        print(f"Debug - Parsed prediction data: {prediction_data}")
        
        if 'direction_probability' in prediction_data and 'predicted_direction' in prediction_data:
            return prediction_data
        else:
            print(f"Debug - Missing essential fields. Found keys: {list(prediction_data.keys())}")
            print(f"Debug - Output text snippet: {output_text[:500]}...")
            return {}
        
    except Exception as e:
        print(f"Error parsing prediction output: {e}")
        print(f"Debug - Output text: {output_text[:200]}...")
        return {}

def create_prediction_dataframe(prediction_data, suggestion, reasoning):
    """Create a formatted DataFrame for the prediction results"""
    try:
        df_data = {
            'Metric': [
                'Predicted Direction',
                'Direction Probability', 
                'Model Confidence',
                'Expected Price Change',
                'Expected Volatility',
                'Expected Spread',
                'Trading Suggestion',
                'Reasoning'
            ],
            'Value': [
                prediction_data.get('predicted_direction', 'N/A'),
                f"{prediction_data.get('direction_probability', 0):.1%}",
                f"{prediction_data.get('confidence', 0):.1%}",
                f"{prediction_data.get('expected_price_change', 0):+.2%}",
                f"{prediction_data.get('expected_volatility', 0):.4f}",
                f"{prediction_data.get('expected_spread', 0):.4f}",
                suggestion,
                reasoning
            ]
        }
        
        return pd.DataFrame(df_data)
        
    except Exception as e:
        return pd.DataFrame({'Error': [f"Failed to create dataframe: {str(e)}"]})

def parse_evaluation_output(output_text):
    """Parse and format the evaluation output from the log format"""
    try:
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', output_text)
        timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metrics = {}
        
        patterns = {
            'Direction Accuracy': r'Direction Accuracy:\s*([\d\.]+)',
            'Direction Precision': r'Direction Precision:\s*([\d\.]+)',
            'Direction Recall': r'Direction Recall:\s*([\d\.]+)',
            'Direction F1': r'Direction F1:\s*([\d\.]+)',
            'Positive Predictions': r'Prediction distribution:\s*([\d\.]+)%\s*positive',
            'Volatility MAE': r'Volatility MAE:\s*([\d\.]+)',
            'Price Change MAE': r'Price Change MAE:\s*([\d\.]+)',
            'Spread MAE': r'Spread MAE:\s*([\d\.]+)'
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                metrics[metric_name] = float(match.group(1))
        
        return timestamp, metrics
        
    except Exception as e:
        print(f"Error parsing evaluation output: {e}")
        return None, {}

def format_evaluation_results(stock_symbol, timestamp, metrics, output_dir):
    """Format evaluation results in a clean, readable format"""
    if not metrics:
        return None
    
    formatted = f"✅ Model Evaluation Results for {stock_symbol}\n"
    formatted += f"🕐 {timestamp}\n"
    formatted += "=" * 60 + "\n\n"
    
    formatted += "📊 Direction Prediction Metrics:\n"
    if 'Direction Accuracy' in metrics:
        formatted += f"   Accuracy:   {metrics['Direction Accuracy']:.4f} ({metrics['Direction Accuracy']*100:.2f}%)\n"
    if 'Direction Precision' in metrics:
        formatted += f"   Precision:  {metrics['Direction Precision']:.4f} ({metrics['Direction Precision']*100:.2f}%)\n"
    if 'Direction Recall' in metrics:
        formatted += f"   Recall:     {metrics['Direction Recall']:.4f} ({metrics['Direction Recall']*100:.2f}%)\n"
    if 'Direction F1' in metrics:
        formatted += f"   F1 Score:   {metrics['Direction F1']:.4f} ({metrics['Direction F1']*100:.2f}%)\n"
    if 'Positive Predictions' in metrics:
        formatted += f"   Positive:   {metrics['Positive Predictions']:.2f}%\n"
    
    formatted += "\n"
    
    formatted += "📉 Prediction Error Metrics (MAE):\n"
    if 'Volatility MAE' in metrics:
        formatted += f"   Volatility:     {metrics['Volatility MAE']:.6f}\n"
    if 'Price Change MAE' in metrics:
        formatted += f"   Price Change:   {metrics['Price Change MAE']:.6f}\n"
    if 'Spread MAE' in metrics:
        formatted += f"   Spread:         {metrics['Spread MAE']:.6f}\n"
    
    formatted += "\n" + "=" * 60 + "\n"
    formatted += f"📁 Full results saved to: {output_dir}\n"
    
    return formatted

def run_prediction(stock_symbol):
    """Run the prediction script for a specific stock with future prediction"""
    try:
        stock_output_dir = f"{BASE_PATH}/{stock_symbol}/predictions"
        os.makedirs(stock_output_dir, exist_ok=True)
        paths = get_stock_paths(stock_symbol)
        predict_path = f"{BASE_PATH}/hanabi-1/predict.py"
        
        cmd = [
            sys.executable, predict_path,
            "--model_path", paths["model_path"],
            "--hourly_data", paths["hourly_data_path"],
            "--fear_greed_data", FEAR_GREED_DATA_PATH,
            "--calibrate_threshold",
            "--future",
            "--output_dir", stock_output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            output_text = result.stdout + "\n" + result.stderr
            prediction_data = parse_future_prediction_output(output_text)
            
            if prediction_data:
                suggestion, reasoning, signal_strength = interpret_future_prediction(prediction_data)
                df = create_prediction_dataframe(prediction_data, suggestion, reasoning)
                full_message = f"🔮 [{timestamp}] Future prediction completed for {stock_symbol}.\n{suggestion}\n{reasoning}"
                return full_message, df
            else:
                return f"⚠️ [{timestamp}] Prediction completed for {stock_symbol} but could not parse future prediction data.\n\nRaw Output:\n{output_text}\n\nPlease check the log format.", pd.DataFrame({'Debug': ['Could not parse prediction data', 'Check raw output above']})
        else:
            error_msg = result.stderr
            
            if "ModuleNotFoundError" in error_msg and "sklearn" in error_msg:
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol} prediction.\n\n🔧 To fix this, run: pip install scikit-learn\n\nFull error:\n{error_msg}", pd.DataFrame()
            elif "ModuleNotFoundError" in error_msg and "torch" in error_msg:
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol} prediction.\n\n🔧 To fix this, run: pip install torch\n\nFull error:\n{error_msg}", pd.DataFrame()
            elif "ModuleNotFoundError" in error_msg:
                missing_module = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol} prediction.\n\n🔧 To fix this, run: pip install {missing_module}\n\nFull error:\n{error_msg}", pd.DataFrame()
            else:
                return f"❌ [{timestamp}] Prediction failed for {stock_symbol}.\n\nError:\n{error_msg}\n\nStdout:\n{result.stdout}", pd.DataFrame()
            
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception occurred for {stock_symbol}: {str(e)}", pd.DataFrame()

def run_model_evaluation(stock_symbol):
    """Run the evaluate_ensemble.py script for a specific stock"""
    try:
        paths = get_stock_paths(stock_symbol)
        
        # ✅ Use stock-specific output directory only
        eval_output_dir = os.path.abspath(f"{BASE_PATH}/{stock_symbol}/evaluations")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        # ✅ Check if model file exists
        if not os.path.exists(paths["model_path"]):
            return f"❌ Model file not found: {paths['model_path']}\n\nPlease train the model first using 'Train Model' button."
        
        model_mtime = os.path.getmtime(paths["model_path"])
        model_time_str = datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # ✅ Try multiple methods to get window_size
        window_size = None
        
        # Method 1: Check for best_params JSON file (from optimization)
        params_file = f"{BASE_PATH}/{stock_symbol}/best_params_{stock_symbol}.json"
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params_data = json.load(f)
                    window_size = str(params_data.get('parameters', {}).get('WINDOW_SIZE'))
                    #if window_size and window_size != 'None':
                        #print(f"✅ Found window_size from best_params file: {window_size}")
            except Exception as e:
                print(f"⚠️  Could not read best_params file: {e}")
        
        # Method 2: Load from model checkpoint
        if not window_size or window_size == 'None':
            import torch
            try:
                checkpoint = torch.load(paths["model_path"], map_location='cpu')
                
                # Debug: Print what's in the checkpoint
                print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
                
                model_hyperparams = checkpoint.get('hyperparameters', {})
                print(f"🔍 Hyperparameters in model: {model_hyperparams}")
                
                # Try different possible keys
                for key in ['window_size', 'WINDOW_SIZE', 'seq_length', 'sequence_length']:
                    if key in model_hyperparams:
                        window_size = str(model_hyperparams[key])
                        #print(f"✅ Found window_size in model as '{key}': {window_size}")
                        break
                
                # Also check if it's stored at top level
                if not window_size or window_size == 'None':
                    for key in ['window_size', 'WINDOW_SIZE']:
                        if key in checkpoint:
                            window_size = str(checkpoint[key])
                            #print(f"✅ Found window_size at checkpoint root: {window_size}")
                            break
                        
            except Exception as e:
                print(f"⚠️  Could not load model hyperparameters: {e}")
        
        # Method 3: Fallback to optimization results
        if not window_size or window_size == 'None':
            opt_results_file = f"{BASE_PATH}/optimization_results.json"
            if os.path.exists(opt_results_file):
                try:
                    with open(opt_results_file, 'r') as f:
                        opt_results = json.load(f)
                        # Find the best result
                        best_result = max(opt_results, key=lambda x: x.get('accuracy', 0) if x.get('success') else 0)
                        if best_result.get('success'):
                            window_size = str(best_result.get('params', {}).get('WINDOW_SIZE'))
                            #if window_size and window_size != 'None':
                                #print(f"✅ Found window_size from optimization results: {window_size}")
                except Exception as e:
                    print(f"⚠️  Could not read optimization results: {e}")
        
        # Method 4: Final fallback - use default
        if not window_size or window_size == 'None':
            window_size = "20"
            print(f"⚠️  Could not find window_size anywhere, using default: {window_size}")
        
        print(f"🔍 Evaluating model for {stock_symbol}")
        print(f"   Model: {paths['model_path']}")
        print(f"   Model last modified: {model_time_str}")
        #print(f"   Using window_size: {window_size}")
        print(f"   Output: {eval_output_dir}")
        
        ee_path = f"{BASE_PATH}/hanabi-1/evaluate_ensemble.py"

        cmd = [
            sys.executable, ee_path,
            "--model_path", paths["model_path"],
            "--hourly_data", paths["hourly_data_path"],
            "--fear_greed_data", FEAR_GREED_DATA_PATH,
            "--window_size", window_size,
            "--output_dir", eval_output_dir
        ]
        
        wd_path = f"{BASE_PATH}/hanabi-1"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=wd_path
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            # Parse evaluation output from stdout/stderr
            eval_timestamp, metrics = parse_evaluation_output(result.stdout + "\n" + result.stderr)
            
            if metrics:
                formatted_result = format_evaluation_results(stock_symbol, eval_timestamp, metrics, eval_output_dir)
                if formatted_result:
                    # ✅ Add model timestamp to output
                    model_info = f"\n📦 Model file: {os.path.basename(paths['model_path'])}\n📅 Model updated: {model_time_str}\n\n"
                    return model_info + formatted_result
            
            # ✅ Look for latest evaluation JSON in the stock-specific directory
            if os.path.exists(eval_output_dir):
                eval_files = [f for f in os.listdir(eval_output_dir) 
                             if f.startswith("evaluation_") and f.endswith(".json")]
                
                if eval_files:
                    latest_eval = sorted(eval_files)[-1]
                    eval_path = os.path.join(eval_output_dir, latest_eval)
                    
                    try:
                        with open(eval_path, 'r') as f:
                            eval_data = json.load(f)
                        
                        formatted_output = f"✅ [{timestamp}] Model evaluation completed for {stock_symbol}\n"
                        formatted_output += f"📦 Model: {os.path.basename(paths['model_path'])}\n"
                        formatted_output += f"📅 Model updated: {model_time_str}\n\n"
                        formatted_output += f"📊 Evaluation Results from {latest_eval}:\n"
                        formatted_output += "=" * 50 + "\n\n"
                        
                        for key, value in eval_data.items():
                            if isinstance(value, float):
                                formatted_output += f"{key}: {value:.4f}\n"
                            else:
                                formatted_output += f"{key}: {value}\n"
                        
                        formatted_output += "\n" + "=" * 50 + "\n"
                        formatted_output += f"\n🔍 Full results saved to: {eval_output_dir}"
                        
                        return formatted_output
                    except Exception as e:
                        return f"✅ [{timestamp}] Model evaluation completed for {stock_symbol}.\n\nResults saved to: {eval_output_dir}\n\nNote: Could not parse JSON results: {str(e)}\n\nRaw Output:\n{result.stdout}"
            
            return f"✅ [{timestamp}] Model evaluation completed for {stock_symbol}.\n\nOutput:\n{result.stdout}\n\nStderr:\n{result.stderr}"
        else:
            error_msg = result.stderr
            
            if "ModuleNotFoundError" in error_msg:
                missing_module = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol} evaluation.\n\n🔧 To fix this, run: pip install {missing_module}\n\nFull error:\n{error_msg}"
            else:
                return f"❌ [{timestamp}] Model evaluation failed for {stock_symbol}.\n\n🔍 Command run:\n{' '.join(cmd)}\n\n❌ Error:\n{error_msg}\n\n📤 Stdout:\n{result.stdout}"
            
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception occurred during {stock_symbol} model evaluation: {str(e)}"


def run_optimization(stock_symbol):
    """Launch continuous_optimization.py in a new terminal window."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script_path = os.path.join(os.path.dirname(__file__), "continuous_optimization.py")

    try:
        if sys.platform == "darwin":  # ✅ macOS FIXED
            subprocess.Popen([
                "osascript", "-e",
                f'tell application "Terminal" to do script "python3 \'{script_path}\' --TICKER {stock_symbol}"'
            ])
        else:
            subprocess.Popen(["x-terminal-emulator", "-e", f"python3 {script_path} --TICKER {stock_symbol}"])
        
        msg = f"🚀 [{timestamp}] Optimization process for {stock_symbol} started in a new terminal window."
    except Exception as e:
        msg = f"❌ [{timestamp}] Failed to start optimization for {stock_symbol}: {e}"

    return msg, pd.DataFrame()

def predict_t():
    message, df = run_prediction("T")
    return message, df

def predict_msft():
    message, df = run_prediction("MSFT")
    return message, df

def predict_gd():
    message, df = run_prediction("GD")
    return message, df

def check_model_t():
    return run_model_evaluation("T")

def check_model_msft():
    return run_model_evaluation("MSFT")

def check_model_gd():
    return run_model_evaluation("GD")

def roll_retrain_t():
    return run_optimization("T")

def roll_retrain_msft():
    return run_optimization("MSFT")

def roll_retrain_gd():
    return run_optimization("GD")

def run_get_data(stock_symbol):
    """Run the download_data.py script for a specific stock"""
    try:
        paths = get_stock_paths(stock_symbol)
        script_path = paths["download_script"]
        
        if not os.path.exists(script_path):
            return f"❌ Script not found: {script_path}"
        
        cmd = [sys.executable, script_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            csv_path = paths["hourly_data_path"]
            mst_time, latest_price, latest_epoch = get_latest_data_info(csv_path)
            
            base_message = f"✅ [{timestamp}] Data collection completed for {stock_symbol}!"
            
            if mst_time and latest_price is not None:
                data_info = f"\n🕐 Latest Data: {mst_time}\n💰 Price: ${latest_price:.2f}"
            else:
                data_info = f"\n⚠️ Could not extract latest data info: {mst_time}"
            
            return f"{base_message}{data_info}\n\nOutput:\n{result.stdout}"
        else:
            error_msg = result.stderr
            
            if "ModuleNotFoundError" in error_msg and "yfinance" in error_msg:
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol}.\n\n🔧 To fix this, run: pip install yfinance\n\nFull error:\n{error_msg}"
            elif "ModuleNotFoundError" in error_msg:
                missing_module = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                return f"❌ [{timestamp}] Missing dependency for {stock_symbol}.\n\n🔧 To fix this, run: pip install {missing_module}\n\nFull error:\n{error_msg}"
            else:
                return f"❌ [{timestamp}] Data collection failed for {stock_symbol}.\n\nError:\n{error_msg}\n\nStdout:\n{result.stdout}"
            
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception occurred for {stock_symbol} data collection: {str(e)}"


def run_sentiment_data():
    """Run the getFGData.py script for sentiment data"""
    try:
        fg_script_path = f"{BASE_PATH}/sentiment-fear-and-greed/getFGData.py"
        
        if not os.path.exists(fg_script_path):
            return f"❌ Script not found: {fg_script_path}"
        
        cmd = [sys.executable, fg_script_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        enhance_script_path = f"{BASE_PATH}/sentiment-fear-and-greed/fear_greed_enhancer.py"
        
        if not os.path.exists(enhance_script_path):
            return f"❌ Script not found: {enhance_script_path}"

        cmd = [sys.executable, enhance_script_path]
        result_1 = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    

        if result.returncode == 0 and result_1.returncode == 0:

            return f"✅ [{timestamp}] Sentiment data collection and enhancment completed!\n\nOutput:\n{result.stdout}"
        else:
            return f"❌ [{timestamp}] Sentiment data collection and enhancment failed.\n\nError:\n{result.stderr}\n\nStdout:\n{result.stdout}\n{result_1.stderr}\n\nStdout:\n{result_1.stdout}"

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception occurred during sentiment data collection: {str(e)}"

def get_data_t():
    return run_get_data("T")

def get_data_msft():
    return run_get_data("MSFT")

def get_data_gd():
    return run_get_data("GD")

# Create the Gradio interface
with gr.Blocks(title="Stock Predictor", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🔮 Stock Price Future Predictor")
    gr.Markdown("Collect data and run **future predictions** for each stock, or update sentiment data globally.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌐 Global Sentiment Data")
            sentiment_button = gr.Button("Update Sentiment Data", variant="secondary", size="lg")
            sentiment_output = gr.Textbox(
                label="Sentiment Data Results",
                lines=5,
                placeholder="Click 'Update Sentiment Data' to run getFGData.py...",
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        # T Column
        with gr.Column():
            gr.Markdown("### 📱 T")
            
            with gr.Row():
                get_data_t_btn = gr.Button("Get Data", variant="secondary")
                check_model_t_btn = gr.Button("Check Model", variant="secondary")
            
            with gr.Row():
                t_button = gr.Button("Predict Future", variant="primary")
                roll_retrain_t_btn = gr.Button("Train Model", variant="primary")
            
            t_output = gr.Textbox(
                label="T Status",
                lines=5,
                placeholder="Use buttons above to collect data or run future predictions...",
                interactive=False
            )
            
            t_results = gr.DataFrame(
                label="T Future Predictions & Trading Suggestions",
                interactive=False,
                wrap=True,
                column_widths=["1fr", "2fr"]
            )
        
        # MSFT Column
        with gr.Column():
            gr.Markdown("### 🪟 MSFT")
            
            with gr.Row():
                get_data_msft_btn = gr.Button("Get Data", variant="secondary")
                check_model_msft_btn = gr.Button("Check Model", variant="secondary")
            
            with gr.Row():
                msft_button = gr.Button("Predict Future", variant="primary")
                roll_retrain_msft_btn = gr.Button("Train Model", variant="primary")
            
            msft_output = gr.Textbox(
                label="MSFT Status",
                lines=5,
                placeholder="Use buttons above to collect data or run future predictions...",
                interactive=False
            )
            
            msft_results = gr.DataFrame(
                label="MSFT Future Predictions & Trading Suggestions",
                interactive=False,
                wrap=True,
                column_widths=["1fr", "2fr"]
            )
        
        # GD Column
        with gr.Column():
            gr.Markdown("### 💣 GD")
            
            with gr.Row():
                get_data_gd_btn = gr.Button("Get Data", variant="secondary")
                check_model_gd_btn = gr.Button("Check Model", variant="secondary")
            
            with gr.Row():
                gd_button = gr.Button("Predict Future", variant="primary")
                roll_retrain_gd_btn = gr.Button("Train Model", variant="primary")
            
            gd_output = gr.Textbox(
                label="GD Status",
                lines=5,
                placeholder="Use buttons above to collect data or run future predictions...",
                interactive=False
            )
            
            gd_results = gr.DataFrame(
                label="GD Future Predictions & Trading Suggestions",
                interactive=False,
                wrap=True,
                column_widths=["1fr", "2fr"]
            )
    
    # Configuration info
    with gr.Accordion("Configuration", open=False):
        gr.Markdown(f"""
        **Base Path:** `{BASE_PATH}`
        
        **Fear & Greed Data:** `{FEAR_GREED_DATA_PATH}`
        
        **Output Directory:** `{OUTPUT_DIR}`
        
        **New Features:**
        - All predictions now use the `--future` flag for actual future predictions!
        - **Check Model** button runs `evaluate_ensemble.py` to evaluate the current model
        - **Train Model** button runs `continous_optimization.py to retrain / train better
        
        **Stock-specific paths are generated dynamically:**
        - Model: `{BASE_PATH}/{{TICKER}}/trained_models/best_financial_model.pt`
        - Data: `{BASE_PATH}/{{TICKER}}/hourly_data.csv`
        - Script: `{BASE_PATH}/{{TICKER}}/download_data.py`
        
        **Trading Suggestions Legend:**
        - 🟢 **STRONG BUY/BUY/SOFT BUY**: Bullish signals with varying confidence levels
        - 🔴 **STRONG SELL/SELL/SOFT SELL**: Bearish signals with varying confidence levels  
        - 🟡 **HOLD/WEAK HOLD**: Neutral or low-confidence signals
        """)
    
    # Connect buttons to functions
    sentiment_button.click(run_sentiment_data, outputs=sentiment_output)
    
    # T buttons
    get_data_t_btn.click(get_data_t, outputs=t_output)
    check_model_t_btn.click(check_model_t, outputs=t_output)
    t_button.click(predict_t, outputs=[t_output, t_results])
    roll_retrain_t_btn.click(roll_retrain_t, outputs=[t_output, t_results])
    
    # MSFT buttons
    get_data_msft_btn.click(get_data_msft, outputs=msft_output)
    check_model_msft_btn.click(check_model_msft, outputs=msft_output)
    msft_button.click(predict_msft, outputs=[msft_output, msft_results])
    roll_retrain_msft_btn.click(roll_retrain_msft, outputs=[msft_output, msft_results])
    
    # GD buttons
    get_data_gd_btn.click(get_data_gd, outputs=gd_output)
    check_model_gd_btn.click(check_model_gd, outputs=gd_output)
    gd_button.click(predict_gd, outputs=[gd_output, gd_results])
    roll_retrain_gd_btn.click(roll_retrain_gd, outputs=[gd_output, gd_results])

if __name__ == "__main__":
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
