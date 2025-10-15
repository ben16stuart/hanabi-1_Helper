import gradio as gr
import subprocess
import os
import sys
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, time
import pytz
import warnings
from pathlib import Path
from put_call_integration import fetch_put_call_ratio, get_put_call_signal, combine_model_and_putcall
import threading
import time as time_module

warnings.filterwarnings("ignore", category=FutureWarning)

# Base configuration
BASE_PATH = str(Path.home() / "Desktop" / "hanabi")
FEAR_GREED_DATA_PATH = os.path.join(BASE_PATH, "sentiment-fear-and-greed/fear_greed_data/fear_greed_index_enhanced.csv")   
OUTPUT_DIR = "./predictions"
AUTO_LOG_FILE = os.path.join(OUTPUT_DIR, "auto_live_log.csv")

# Global state for auto-live mode
auto_live_state = {
    "enabled": False,
    "thread": None,
    "last_run": None,
    "next_run": None
}

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
        
        if 'direction_probability' in prediction_data and 'predicted_direction' in prediction_data:
            return prediction_data
        else:
            return {}
        
    except Exception as e:
        print(f"Error parsing prediction output: {e}")
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
        
        eval_output_dir = os.path.abspath(f"{BASE_PATH}/{stock_symbol}/evaluations")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        if not os.path.exists(paths["model_path"]):
            return f"❌ Model file not found: {paths['model_path']}\n\nPlease train the model first using 'Train Model' button."
        
        model_mtime = os.path.getmtime(paths["model_path"])
        model_time_str = datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        window_size = None

        # Canonical params location: trained_models/best_params_{TICKER}.json
        candidate_param_paths = [
            os.path.join(BASE_PATH, stock_symbol, 'trained_models', f"best_params_{stock_symbol}.json"),
            os.path.join(BASE_PATH, stock_symbol, 'trained_models', 'best_params.json')
        ]

        for p in candidate_param_paths:
            try:
                if not os.path.exists(p):
                    continue
                if os.path.isdir(p):
                    # Skip directories (avoid iterating strings as paths)
                    print(f"⚠️  Found directory instead of params file: '{p}'")
                    continue

                with open(p, 'r') as f:
                    params_data = json.load(f)
                    ws = None
                    if isinstance(params_data, dict):
                        ws = params_data.get('parameters', {}).get('WINDOW_SIZE') if params_data.get('parameters') else None
                        if ws is None:
                            for key in ('WINDOW_SIZE', 'window_size', 'seq_length', 'sequence_length'):
                                if key in params_data:
                                    ws = params_data.get(key)
                                    break
                    if ws is not None:
                        window_size = str(ws)
                        print(f"ℹ️ Using window_size from params file: {p} -> {window_size}")
                        break
            except Exception as e:
                print(f"⚠️  Could not read best_params file '{p}': {e}")

        # fallback: read from model checkpoint hyperparameters
        if not window_size or window_size == 'None':
            try:
                import torch
                checkpoint = torch.load(paths["model_path"], map_location='cpu')
                model_hyperparams = {}
                if isinstance(checkpoint, dict):
                    model_hyperparams = checkpoint.get('hyperparameters', {}) or checkpoint.get('hyperparams', {}) or {}

                for key in ['window_size', 'WINDOW_SIZE', 'seq_length', 'sequence_length']:
                    if key in model_hyperparams:
                        window_size = str(model_hyperparams[key])
                        break

                if not window_size or window_size == 'None':
                    for key in ['window_size', 'WINDOW_SIZE']:
                        if key in checkpoint:
                            window_size = str(checkpoint[key])
                            break
            except Exception as e:
                print(f"⚠️  Could not load model hyperparameters: {e}")
        
        if not window_size or window_size == 'None':
            opt_results_file = f"{BASE_PATH}/optimization_results.json"
            if os.path.exists(opt_results_file):
                try:
                    with open(opt_results_file, 'r') as f:
                        opt_results = json.load(f)
                        best_result = max(opt_results, key=lambda x: x.get('accuracy', 0) if x.get('success') else 0)
                        if best_result.get('success'):
                            window_size = str(best_result.get('params', {}).get('WINDOW_SIZE'))
                except Exception as e:
                    print(f"⚠️  Could not read optimization results: {e}")
        
        if not window_size or window_size == 'None':
            window_size = "24"
            print(f"⚠️  Could not find window_size anywhere, using default: {window_size}")
        
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
            eval_timestamp, metrics = parse_evaluation_output(result.stdout + "\n" + result.stderr)
            
            if metrics:
                formatted_result = format_evaluation_results(stock_symbol, eval_timestamp, metrics, eval_output_dir)
                if formatted_result:
                    model_info = f"\n📦 Model file: {os.path.basename(paths['model_path'])}\n📅 Model updated: {model_time_str}\n\n"
                    return model_info + formatted_result
            
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
                        formatted_output += f"\n📁 Full results saved to: {eval_output_dir}"
                        
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
        if sys.platform == "darwin":
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

def run_prediction_with_putcall(stock_symbol):
    """
    Enhanced prediction that combines model output with put/call ratio analysis
    """
    try:
        stock_output_dir = f"{BASE_PATH}/{stock_symbol}/predictions"
        os.makedirs(stock_output_dir, exist_ok=True)
        paths = get_stock_paths(stock_symbol)
        predict_path = f"{BASE_PATH}/hanabi-1/predict.py"
        
        print(f"📊 Fetching put/call ratio for {stock_symbol}...")
        pc_data_df = fetch_put_call_ratio(stock_symbol)
        
        pc_signal = None
        if not pc_data_df.empty:
            pc_ratio = pc_data_df['put_call_volume_ratio'].iloc[0]
            pc_signal = get_put_call_signal(pc_ratio)
            print(f"✅ P/C Ratio: {pc_ratio:.3f} | Signal: {pc_signal['signal']}")
        else:
            print(f"⚠️ No put/call data available for {stock_symbol}")
        
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
                base_suggestion, base_reasoning, signal_strength = interpret_future_prediction(prediction_data)
                
                if pc_signal:
                    combined = combine_model_and_putcall(prediction_data, pc_signal)
                    
                    if combined['agreement']:
                        if 'BUY' in base_suggestion and 'BUY' in pc_signal['signal']:
                            enhanced_suggestion = "🟢🟢 STRONG BUY (Model + P/C Confirmed)"
                        elif 'SELL' in base_suggestion and 'SELL' in pc_signal['signal']:
                            enhanced_suggestion = "🔴🔴 STRONG SELL (Model + P/C Confirmed)"
                        else:
                            enhanced_suggestion = base_suggestion + " ✅"
                    elif combined['conflict']:
                        enhanced_suggestion = "🟡 HOLD (Model/P/C Conflict)"
                    else:
                        enhanced_suggestion = base_suggestion
                    
                    enhanced_reasoning = f"{base_reasoning}\n\n📊 Put/Call Analysis:\n"
                    enhanced_reasoning += f"   Ratio: {pc_signal['put_call_ratio']:.3f}\n"
                    enhanced_reasoning += f"   P/C Signal: {pc_signal['signal']}\n"
                    enhanced_reasoning += f"   {pc_signal['interpretation']}\n"
                    enhanced_reasoning += f"   {combined['reasoning']}"
                    
                    df_data = {
                        'Metric': [
                            '🎯 Final Recommendation',
                            'Model Direction',
                            'Model Probability', 
                            'Model Confidence',
                            'Put/Call Ratio',
                            'P/C Signal',
                            'Agreement Status',
                            'Combined Confidence',
                            'Expected Price Change',
                            'Expected Volatility',
                            'Reasoning'
                        ],
                        'Value': [
                            enhanced_suggestion,
                            prediction_data.get('predicted_direction', 'N/A'),
                            f"{prediction_data.get('direction_probability', 0):.1%}",
                            f"{prediction_data.get('confidence', 0):.1%}",
                            f"{pc_signal['put_call_ratio']:.3f}",
                            pc_signal['signal'],
                            '✅ Confirmed' if combined['agreement'] else ('⚠️ Conflict' if combined['conflict'] else '🟡 Neutral'),
                            f"{combined['combined_confidence']:.1%}",
                            f"{prediction_data.get('expected_price_change', 0):+.2%}",
                            f"{prediction_data.get('expected_volatility', 0):.4f}",
                            enhanced_reasoning
                        ]
                    }
                    
                    df = pd.DataFrame(df_data)
                    full_message = f"🔮 [{timestamp}] Enhanced prediction for {stock_symbol}\n{enhanced_suggestion}\n\n{enhanced_reasoning}"
                    
                else:
                    df = create_prediction_dataframe(prediction_data, base_suggestion, base_reasoning)
                    full_message = f"🔮 [{timestamp}] Future prediction for {stock_symbol} (no P/C data)\n{base_suggestion}\n{base_reasoning}"
                
                return full_message, df
            else:
                return f"⚠️ [{timestamp}] Could not parse prediction data for {stock_symbol}", pd.DataFrame()
        else:
            return f"❌ [{timestamp}] Prediction failed for {stock_symbol}\n{result.stderr}", pd.DataFrame()
            
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception: {str(e)}", pd.DataFrame()

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
            return f"✅ [{timestamp}] Sentiment data collection and enhancement completed!\n\nOutput:\n{result.stdout}"
        else:
            return f"❌ [{timestamp}] Sentiment data collection and enhancement failed.\n\nError:\n{result.stderr}\n\nStdout:\n{result.stdout}\n{result_1.stderr}\n\nStdout:\n{result_1.stdout}"

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"❌ [{timestamp}] Exception occurred during sentiment data collection: {str(e)}"


# ============= AUTO-LIVE FUNCTIONALITY =============

def get_next_market_run_time():
    """Calculate the next run time during market hours (9:30 AM - 4:30 PM EST, every hour)"""
    est_tz = pytz.timezone('US/Eastern')
    now_est = datetime.now(est_tz)
    
    # Market hours: 9:30 AM to 4:30 PM EST
    market_open = time(9, 30)
    market_close = time(16, 30)
    
    # Run times: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30, 16:30
    run_times = [time(h, 30) for h in range(9, 17)]
    
    current_time = now_est.time()
    current_date = now_est.date()
    
    # Find next run time today
    for run_time in run_times:
        if current_time < run_time:
            next_run = datetime.combine(current_date, run_time)
            next_run = est_tz.localize(next_run)
            return next_run
    
    # If no more runs today, schedule for 9:30 AM tomorrow
    next_day = current_date + pd.Timedelta(days=1)
    next_run = datetime.combine(next_day, market_open)
    next_run = est_tz.localize(next_run)
    return next_run


def log_auto_run_results(stock_symbol, price, suggestion, timestamp_str):
    """Log auto-live results to CSV file"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        log_entry = {
            'Timestamp': timestamp_str,
            'Stock': stock_symbol,
            'Price': f"${price:.2f}" if price else "N/A",
            'Suggestion': suggestion
        }
        
        # Create or append to CSV
        if os.path.exists(AUTO_LOG_FILE):
            df = pd.read_csv(AUTO_LOG_FILE)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([log_entry])
        
        df.to_csv(AUTO_LOG_FILE, index=False)
        print(f"✅ Logged: {stock_symbol} @ ${price:.2f} - {suggestion}")
        
    except Exception as e:
        print(f"❌ Failed to log results: {e}")


def format_auto_log_display():
    """Format the auto-live log for display"""
    try:
        if not os.path.exists(AUTO_LOG_FILE):
            return "📋 No auto-live runs yet. Enable Auto-Live to start logging."
        
        df = pd.read_csv(AUTO_LOG_FILE)
        
        if df.empty:
            return "📋 No auto-live runs yet."
        
        # Get last 20 entries
        df_display = df.tail(20)
        
        output = "📋 Auto-Live Run History (Last 20 runs)\n"
        output += "=" * 80 + "\n\n"
        
        for _, row in df_display.iterrows():
            output += f"🕐 {row['Timestamp']} | {row['Stock']:5s} | {row['Price']:10s} | {row['Suggestion']}\n"
        
        output += "\n" + "=" * 80 + "\n"
        output += f"📁 Full log: {AUTO_LOG_FILE}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error reading log: {str(e)}"


def run_auto_cycle():
    """Execute one complete auto-live cycle"""
    est_tz = pytz.timezone('US/Eastern')
    timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M %p EST")
    
    print(f"\n{'='*60}")
    print(f"🤖 AUTO-LIVE CYCLE STARTED: {timestamp}")
    print(f"{'='*60}\n")
    
    results_summary = f"🤖 AUTO-LIVE CYCLE: {timestamp}\n\n"
    
    # Step 1: Update Sentiment Data
    print("📊 Step 1: Updating sentiment data...")
    sentiment_result = run_sentiment_data()
    results_summary += "1️⃣ SENTIMENT UPDATE:\n"
    if "✅" in sentiment_result:
        results_summary += "   ✅ Completed\n\n"
    else:
        results_summary += f"   ❌ Failed\n\n"
    
    time_module.sleep(2)
    
    # Step 2: Get Data for all stocks
    stocks = ["T", "MSFT", "GD"]
    stock_prices = {}
    
    print("📥 Step 2: Collecting stock data...")
    results_summary += "2️⃣ DATA COLLECTION:\n"
    
    for stock in stocks:
        print(f"   Getting data for {stock}...")
        data_result = run_get_data(stock)
        
        if "✅" in data_result and "Price:" in data_result:
            price_match = re.search(r'Price: \$?([\d\.]+)', data_result)
            if price_match:
                stock_prices[stock] = float(price_match.group(1))
                results_summary += f"   ✅ {stock}: ${stock_prices[stock]:.2f}\n"
        else:
            results_summary += f"   ❌ {stock}: Failed\n"
        
        time_module.sleep(1)
    
    results_summary += "\n"
    
    # Step 3: Run Predictions
    print("🔮 Step 3: Running predictions...")
    results_summary += "3️⃣ PREDICTIONS:\n"
    
    for stock in stocks:
        print(f"   Predicting {stock}...")
        pred_message, pred_df = run_prediction_with_putcall(stock)
        
        # Extract suggestion from message or dataframe
        suggestion = "UNKNOWN"
        if not pred_df.empty:
            rec_row = pred_df[pred_df['Metric'].str.contains('Recommendation|Suggestion', case=False, na=False)]
            if not rec_row.empty:
                suggestion = rec_row['Value'].iloc[0]
        
        price = stock_prices.get(stock, 0)
        results_summary += f"   {stock}: ${price:.2f} → {suggestion}\n"
        
        # Log to CSV
        log_auto_run_results(stock, price, suggestion, timestamp)
        
        time_module.sleep(1)
    
    results_summary += "\n" + "="*60 + "\n"
    results_summary += "✅ AUTO-LIVE CYCLE COMPLETED\n"
    
    print(f"\n{'='*60}")
    print("✅ AUTO-LIVE CYCLE COMPLETED")
    print(f"{'='*60}\n")
    
    auto_live_state["last_run"] = timestamp
    
    return results_summary


def auto_live_scheduler():
    """Background thread that runs auto-live cycles"""
    while auto_live_state["enabled"]:
        try:
            next_run = get_next_market_run_time()
            auto_live_state["next_run"] = next_run.strftime("%Y-%m-%d %I:%M %p EST")
            
            est_tz = pytz.timezone('US/Eastern')
            now_est = datetime.now(est_tz)
            
            sleep_seconds = (next_run - now_est).total_seconds()
            
            if sleep_seconds > 0:
                print(f"⏰ Next auto-live run scheduled for: {auto_live_state['next_run']}")
                print(f"   Sleeping for {sleep_seconds/60:.1f} minutes...")
                
                # Sleep in small intervals to allow quick shutdown
                sleep_intervals = int(sleep_seconds / 10) + 1
                for _ in range(sleep_intervals):
                    if not auto_live_state["enabled"]:
                        print("🛑 Auto-live disabled, stopping scheduler")
                        return
                    time_module.sleep(min(10, sleep_seconds))
                    sleep_seconds -= 10
                    if sleep_seconds <= 0:
                        break
            
            if auto_live_state["enabled"]:
                run_auto_cycle()
            
        except Exception as e:
            print(f"❌ Auto-live scheduler error: {e}")
            time_module.sleep(60)


def toggle_auto_live(enabled):
    """Toggle auto-live mode on/off"""
    auto_live_state["enabled"] = enabled
    
    if enabled:
        if auto_live_state["thread"] is None or not auto_live_state["thread"].is_alive():
            auto_live_state["thread"] = threading.Thread(target=auto_live_scheduler, daemon=True)
            auto_live_state["thread"].start()
            
            next_run = get_next_market_run_time()
            auto_live_state["next_run"] = next_run.strftime("%Y-%m-%d %I:%M %p EST")
            
            status = f"✅ Auto-Live ENABLED\n\n"
            status += f"🕐 Next run: {auto_live_state['next_run']}\n"
            status += f"📍 Last run: {auto_live_state['last_run'] or 'Never'}\n\n"
            status += "The system will automatically:\n"
            status += "  1. Update sentiment data\n"
            status += "  2. Collect stock data (T, MSFT, GD)\n"
            status += "  3. Run predictions\n"
            status += "  4. Log results\n\n"
            status += "Runs every hour from 9:30 AM to 4:30 PM EST"
            
            return status
    else:
        auto_live_state["next_run"] = None
        status = f"🛑 Auto-Live DISABLED\n\n"
        status += f"📍 Last run: {auto_live_state['last_run'] or 'Never'}\n\n"
        status += "Manual mode active. Use buttons to run predictions."
        
        return status


def get_auto_live_status():
    """Get current auto-live status"""
    if auto_live_state["enabled"]:
        status = "🟢 Auto-Live: ACTIVE\n\n"
        status += f"🕐 Next run: {auto_live_state['next_run']}\n"
        status += f"📍 Last run: {auto_live_state['last_run'] or 'Never'}"
    else:
        status = "⚪ Auto-Live: INACTIVE\n\n"
        status += f"📍 Last run: {auto_live_state['last_run'] or 'Never'}"
    
    return status


# ============= BUTTON HANDLERS =============

def predict_t_enhanced():
    message, df = run_prediction_with_putcall("T")
    return message, df

def predict_msft_enhanced():
    message, df = run_prediction_with_putcall("MSFT")
    return message, df

def predict_gd_enhanced():
    message, df = run_prediction_with_putcall("GD")
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

def get_data_t():
    return run_get_data("T")

def get_data_msft():
    return run_get_data("MSFT")

def get_data_gd():
    return run_get_data("GD")


# ============= GRADIO INTERFACE =============

with gr.Blocks(title="Stock Predictor", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🔮 Stock Price Future Predictor")
    gr.Markdown("Collect data and run **future predictions** for each stock, or enable **Auto-Live** for automated trading.")
    
    # Auto-Live Section
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Auto-Live Mode")
            #gr.Markdown("Automatically runs predictions every hour during market hours (9:30 AM - 4:30 PM EST)")
            
            auto_live_toggle = gr.Checkbox(
                label="Enable Auto-Live",
                value=False,
                info="Turn on to automate data collection and predictions"
            )
            
            auto_live_status = gr.Textbox(
                label="Auto-Live Status",
                lines=12,
                value="⚪ Auto-Live: INACTIVE\n\nManual mode active. Use buttons below to run predictions.",
                interactive=False
            )
        
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Run History")
            auto_log_display = gr.Textbox(
                label="Auto-Live Run History",
                lines=10,
                value=format_auto_log_display(),
                interactive=False
            )
            
            refresh_log_btn = gr.Button("🔄 Refresh Log", variant="secondary")
    
    gr.Markdown("---")
    
    # Manual Controls Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌍 Global Sentiment Data")
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
        
        **Auto-Live Log:** `{AUTO_LOG_FILE}`
        
        **Auto-Live Schedule:**
        - Runs every hour during market hours (9:30 AM - 4:30 PM EST)
        - Execution times: 9:30, 10:30, 11:30, 12:30, 1:30, 2:30, 3:30, 4:30
        - Each cycle: Updates sentiment → Collects data → Runs predictions → Logs results
        
        **Manual Features:**
        - All predictions use the `--future` flag for actual future predictions
        - **Check Model** button runs `evaluate_ensemble.py` to evaluate the current model
        - **Train Model** button runs `continuous_optimization.py` to retrain/optimize
        
        **Stock-specific paths are generated dynamically:**
        - Model: `{BASE_PATH}/{{TICKER}}/trained_models/best_financial_model.pt`
        - Data: `{BASE_PATH}/{{TICKER}}/hourly_data.csv`
        - Script: `{BASE_PATH}/{{TICKER}}/download_data.py`
        
        **Trading Suggestions Legend:**
        - 🟢 **STRONG BUY/BUY/SOFT BUY**: Bullish signals with varying confidence levels
        - 🔴 **STRONG SELL/SELL/SOFT SELL**: Bearish signals with varying confidence levels  
        - 🟡 **HOLD/WEAK HOLD**: Neutral or low-confidence signals
        """)
    
    # Connect Auto-Live controls
    auto_live_toggle.change(
        toggle_auto_live,
        inputs=[auto_live_toggle],
        outputs=[auto_live_status]
    )
    
    refresh_log_btn.click(
        format_auto_log_display,
        outputs=[auto_log_display]
    )
    
    # Connect manual buttons
    sentiment_button.click(run_sentiment_data, outputs=sentiment_output)
    
    # T buttons
    get_data_t_btn.click(get_data_t, outputs=t_output)
    check_model_t_btn.click(check_model_t, outputs=t_output)
    t_button.click(predict_t_enhanced, outputs=[t_output, t_results])
    roll_retrain_t_btn.click(roll_retrain_t, outputs=[t_output, t_results])
    
    # MSFT buttons
    get_data_msft_btn.click(get_data_msft, outputs=msft_output)
    check_model_msft_btn.click(check_model_msft, outputs=msft_output)
    msft_button.click(predict_msft_enhanced, outputs=[msft_output, msft_results])
    roll_retrain_msft_btn.click(roll_retrain_msft, outputs=[msft_output, msft_results])
    
    # GD buttons
    get_data_gd_btn.click(get_data_gd, outputs=gd_output)
    check_model_gd_btn.click(check_model_gd, outputs=gd_output)
    gd_button.click(predict_gd_enhanced, outputs=[gd_output, gd_results])
    roll_retrain_gd_btn.click(roll_retrain_gd, outputs=[gd_output, gd_results])

if __name__ == "__main__":
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
