import gradio as gr
import subprocess
import os
import sys
import json
import pandas as pd
import re
from datetime import datetime, date
import pytz

# Base configuration
BASE_PATH = "/<path to folder>/hanabi"
FEAR_GREED_DATA_PATH = "<path to file >/fear_greed_data/fear_greed_index_enhanced.csv"
WINDOW_SIZE = "4"
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
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if the dataframe is empty
        if df.empty:
            return "No data found in CSV file", None, None
        
        # Sort by epoch to get the latest entry (assuming epoch column exists)
        # Common column names for epoch time
        epoch_columns = ['epoch', 'timestamp', 'Epoch', 'Timestamp', 'time', 'Time']
        epoch_col = None
        
        for col in epoch_columns:
            if col in df.columns:
                epoch_col = col
                break
        
        if epoch_col is None:
            return f"No epoch/timestamp column found. Available columns: {list(df.columns)}", None, None
        
        # Sort by epoch and get the latest row
        df_sorted = df.sort_values(by=epoch_col, ascending=False)
        latest_row = df_sorted.iloc[0]
        
        latest_epoch = latest_row[epoch_col]
        
        # Find price column (common names)
        price_columns = ['close', 'Close', 'price', 'Price', 'last', 'Last']
        price_col = None
        
        for col in price_columns:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            # If no standard price column, look for any column with 'price' in the name
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            if price_cols:
                price_col = price_cols[0]
        
        latest_price = latest_row[price_col] if price_col else "Price column not found"
        
        # Convert epoch to MST time
        mst_time = convert_epoch_to_mst(latest_epoch)
        
        return mst_time, latest_price, latest_epoch
        
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None, None

def convert_epoch_to_mst(epoch_timestamp):
    """Convert epoch timestamp to MST HH:MM AM/PM format"""
    try:
        # Handle both seconds and milliseconds timestamps
        if epoch_timestamp > 10000000000:  # If more than 10 digits, it's milliseconds
            epoch_timestamp = epoch_timestamp / 1000
        
        # Convert epoch to UTC datetime
        utc_dt = datetime.fromtimestamp(epoch_timestamp, tz=pytz.UTC)
        
        # Convert to MST (Mountain Standard Time)
        mst_tz = pytz.timezone('US/Mountain')
        mst_dt = utc_dt.astimezone(mst_tz)
        
        # Format as HH:MM AM/PM
        formatted_time = mst_dt.strftime("%I:%M %p")
        
        # Also include the date for context
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
        
        # Calculate overall signal strength
        # Higher confidence and clearer direction = stronger signal
        signal_strength = confidence * abs(direction_prob - 0.5) * 2  # Scale direction prob difference
        
        # Trading suggestion logic
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
        
        # Add context based on other metrics
        context_flags = []
        
        # Volatility warning
        if abs(expected_volatility) > 0.5:
            context_flags.append("⚠️ High volatility expected")
        elif abs(expected_volatility) > 0.3:
            context_flags.append("📊 Moderate volatility expected")
        else:
            context_flags.append("📉 Low volatility expected")
        
        # Price change magnitude
        if abs(expected_price_change) > 0.05:
            context_flags.append(f"📈 Large price move expected ({expected_price_change:+.2%})")
        elif abs(expected_price_change) > 0.02:
            context_flags.append(f"📊 Moderate price move expected ({expected_price_change:+.2%})")
        else:
            context_flags.append(f"📉 Small price move expected ({expected_price_change:+.2%})")
        
        # Spread analysis
        if abs(expected_spread) > 0.03:
            context_flags.append("💰 Wide spread expected (higher trading costs)")
        
        # Add flags to reasoning
        if context_flags:
            reasoning += " | " + " | ".join(context_flags)
        
        return suggestion, reasoning, signal_strength
        
    except Exception as e:
        return "🟡 HOLD", f"Error interpreting prediction: {str(e)}", 0

def parse_future_prediction_output(output_text):
    """Parse the future prediction output from the log format"""
    try:
        prediction_data = {}
        
        # Extract timestamp
        timestamp_match = re.search(r'Timestamp:\s*([\d\-:\s]+)', output_text)
        if timestamp_match:
            prediction_data['timestamp'] = timestamp_match.group(1).strip()
        
        # Extract direction probability
        prob_match = re.search(r'Direction probability:\s*([\d\.]+)', output_text)
        if prob_match:
            prediction_data['direction_probability'] = float(prob_match.group(1))
        
        # Extract predicted direction
        direction_match = re.search(r'Predicted direction:\s*(\w+)', output_text)
        if direction_match:
            prediction_data['predicted_direction'] = direction_match.group(1)
        
        # Extract confidence
        confidence_match = re.search(r'Confidence:\s*([\d\.\-]+)', output_text)
        if confidence_match:
            prediction_data['confidence'] = float(confidence_match.group(1))
        
        # Extract expected volatility
        volatility_match = re.search(r'Expected volatility:\s*([\d\.\-]+)', output_text)
        if volatility_match:
            prediction_data['expected_volatility'] = float(volatility_match.group(1))
        
        # Extract expected price change
        price_change_match = re.search(r'Expected price change:\s*([\d\.\-]+)', output_text)
        if price_change_match:
            prediction_data['expected_price_change'] = float(price_change_match.group(1))
        
        # Extract expected spread
        spread_match = re.search(r'Expected spread:\s*([\d\.\-]+)', output_text)
        if spread_match:
            prediction_data['expected_spread'] = float(spread_match.group(1))
        
        # Debug: Print what we found
        print(f"Debug - Parsed prediction data: {prediction_data}")
        
        # Only return data if we found the essential fields
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

def interpret_metrics(metrics):
    """Interpret the model metrics and provide trading suggestion"""
    direction_accuracy = metrics.get('direction_accuracy', 0)
    direction_precision = metrics.get('direction_precision', 0)
    direction_recall = metrics.get('direction_recall', 0)
    direction_f1 = metrics.get('direction_f1', 0)
    positive_prediction_pct = metrics.get('positive_prediction_pct', 0)
    volatility_mae = metrics.get('volatility_mae', 0)
    price_change_mae = metrics.get('price_change_mae', 0)
    spread_mae = metrics.get('spread_mae', 0)
    
    # Calculate weighted model confidence score
    # Weight accuracy and F1 more heavily as they're more comprehensive metrics
    model_confidence = (
        direction_accuracy * 0.4 + 
        direction_f1 * 0.3 + 
        direction_precision * 0.15 + 
        direction_recall * 0.15
    )
    
    # More realistic thresholds for financial models
    high_confidence_threshold = 0.58  # 58% weighted average
    medium_confidence_threshold = 0.52  # 52% weighted average
    min_accuracy_high = 0.55  # 55% accuracy for high confidence
    min_accuracy_medium = 0.51  # 51% accuracy for medium confidence
    
    # Determine trading suggestion based on metrics
    if model_confidence >= high_confidence_threshold and direction_accuracy >= min_accuracy_high:
        if positive_prediction_pct > 55:
            suggestion = "🟢 STRONG BUY"
            reasoning = f"High model confidence ({model_confidence:.2f}) with bullish predictions ({positive_prediction_pct:.1f}%)"
        elif positive_prediction_pct < 45:
            suggestion = "🔴 STRONG SELL"
            reasoning = f"High model confidence ({model_confidence:.2f}) with bearish predictions ({positive_prediction_pct:.1f}%)"
        else:
            suggestion = "🟡 HOLD"
            reasoning = f"High model confidence ({model_confidence:.2f}) but neutral direction ({positive_prediction_pct:.1f}%)"
    
    elif model_confidence >= medium_confidence_threshold and direction_accuracy >= min_accuracy_medium:
        if positive_prediction_pct > 52:
            suggestion = "🟢 SOFT BUY"
            reasoning = f"Moderate confidence ({model_confidence:.2f}) with slight bullish bias ({positive_prediction_pct:.1f}%)"
        elif positive_prediction_pct < 48:
            suggestion = "🔴 SOFT SELL"
            reasoning = f"Moderate confidence ({model_confidence:.2f}) with slight bearish bias ({positive_prediction_pct:.1f}%)"
        else:
            suggestion = "🟡 HOLD"
            reasoning = f"Moderate confidence ({model_confidence:.2f}) but mixed signals ({positive_prediction_pct:.1f}%)"
    
    # Low confidence cases
    elif model_confidence >= 0.45:  # Still some signal
        suggestion = "🟡 WEAK HOLD"
        reasoning = f"Low confidence ({model_confidence:.2f}) - limited predictive power but some signal detected"
    else:
        suggestion = "🟡 HOLD"
        reasoning = f"Very low confidence ({model_confidence:.2f}) - insufficient data for reliable prediction"
    
    # Add additional context based on other metrics
    context_flags = []
    
    # Volatility warning
    if volatility_mae > 0.5:
        context_flags.append("⚠️ High volatility expected")
    elif volatility_mae > 0.3:
        context_flags.append("📊 Moderate volatility expected")
    
    # Precision/Recall imbalance warning
    if abs(direction_precision - direction_recall) > 0.15:
        if direction_precision > direction_recall:
            context_flags.append("🎯 Conservative predictions (high precision)")
        else:
            context_flags.append("🔄 Liberal predictions (high recall)")
    
    # Price change prediction quality
    if price_change_mae < 0.02:  # Very low MAE
        context_flags.append("📈 Excellent price change prediction")
    elif price_change_mae > 0.1:  # High MAE
        context_flags.append("⚠️ Poor price change prediction")
    
    # Add flags to reasoning
    if context_flags:
        reasoning += " | " + " | ".join(context_flags)
    
    return suggestion, reasoning, model_confidence

def parse_prediction_results(stock_symbol, output_dir):
    """Parse the JSON metrics results and return a formatted display"""
    try:
        # Look for the most recent metrics file for this stock
        predictions_dir = f"{BASE_PATH}/{stock_symbol}/predictions"
        if not os.path.exists(predictions_dir):
            return "No predictions directory found.", None, "", ""
        
        # Find JSON files matching the metrics pattern
        json_files = [f for f in os.listdir(predictions_dir) if f.startswith("metrics_w4_") and f.endswith(".json")]
        if not json_files:
            return "No metrics JSON files found.", None, "", ""
        
        # Get the most recent file
        json_files.sort(reverse=True)
        latest_file = os.path.join(predictions_dir, json_files[0])
        
        # Read and parse the JSON
        with open(latest_file, 'r') as f:
            metrics = json.load(f)
        
        # Create a formatted display DataFrame
        metrics_data = {
            'Metric': [
                'Direction Accuracy',
                'Direction Precision', 
                'Direction Recall',
                'Direction F1 Score',
                'Positive Predictions %',
                'Volatility MAE',
                'Price Change MAE',
                'Spread MAE'
            ],
            'Metric Value': [
                f"{metrics.get('direction_accuracy', 0):.4f}",
                f"{metrics.get('direction_precision', 0):.4f}",
                f"{metrics.get('direction_recall', 0):.4f}",
                f"{metrics.get('direction_f1', 0):.4f}",
                f"{metrics.get('positive_prediction_pct', 0):.2f}%",
                f"{metrics.get('volatility_mae', 0):.4f}",
                f"{metrics.get('price_change_mae', 0):.4f}",
                f"{metrics.get('spread_mae', 0):.4f}"
            ],
            'Interpretation': [
                '📈 Higher = Better prediction accuracy',
                '🎯 Higher = Better positive predictions',
                '✅ Higher = Better at catching moves',
                '⚖️ Higher = Better overall balance',
                '📊 % of bullish predictions',
                '📉 Lower = Better volatility prediction',
                '💰 Lower = Better price prediction',
                '📊 Lower = Better spread prediction'
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        
        # Get trading suggestion
        suggestion, reasoning, confidence = interpret_metrics(metrics)
        
        return f"✅ Model metrics loaded from {latest_file.split('/')[-1]}", df, suggestion, reasoning
        
    except Exception as e:
        return f"❌ Error parsing results: {str(e)}", None, "", ""

def run_prediction(stock_symbol):
    """Run the prediction script for a specific stock with future prediction"""
    try:
        # Create output directory if it doesn't exist
        stock_output_dir = f"{BASE_PATH}/{stock_symbol}/predictions"
        os.makedirs(stock_output_dir, exist_ok=True)
        
        # Get paths for this specific stock
        paths = get_stock_paths(stock_symbol)
        
        # Build the command with stock-specific paths and --future flag
        cmd = [
            sys.executable, f"{BASE_PATH}/hanabi-1/predict.py",
            "--model_path", paths["model_path"],
            "--hourly_data", paths["hourly_data_path"],
            "--fear_greed_data", FEAR_GREED_DATA_PATH,
            "--window_size", WINDOW_SIZE,
            "--calibrate_threshold",
            "--future",  # Add the future prediction flag
            "--output_dir", stock_output_dir
        ]
        
        # Run the prediction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            # Always parse future prediction from stdout/stderr when using --future flag
            output_text = result.stdout + "\n" + result.stderr
            
            # Parse the future prediction format from the log output
            prediction_data = parse_future_prediction_output(output_text)
            
            if prediction_data:
                # Get trading suggestion based on future prediction
                suggestion, reasoning, signal_strength = interpret_future_prediction(prediction_data)
                
                # Create DataFrame with future prediction data
                df = create_prediction_dataframe(prediction_data, suggestion, reasoning)
                
                full_message = f"🔮 [{timestamp}] Future prediction completed for {stock_symbol}.\n{suggestion}\n{reasoning}"
                return full_message, df
            else:
                # If we can't parse future prediction data, show the raw output for debugging
                return f"⚠️ [{timestamp}] Prediction completed for {stock_symbol} but could not parse future prediction data.\n\nRaw Output:\n{output_text}\n\nPlease check the log format.", pd.DataFrame({'Debug': ['Could not parse prediction data', 'Check raw output above']})
        else:
            error_msg = result.stderr
            
            # Check for common dependency issues and provide helpful suggestions
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

def predict_t():
    message, df = run_prediction("T")
    return message, df

def predict_msft():
    message, df = run_prediction("MSFT")
    return message, df

def predict_gd():
    message, df = run_prediction("GD")
    return message, df

def run_get_data(stock_symbol):
    """Run the download_data.py script for a specific stock"""
    try:
        paths = get_stock_paths(stock_symbol)
        script_path = paths["download_script"]
        
        if not os.path.exists(script_path):
            return f"❌ Script not found: {script_path}"
        
        cmd = [sys.executable, script_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            # Get latest data info from the CSV file
            csv_path = paths["hourly_data_path"]
            mst_time, latest_price, latest_epoch = get_latest_data_info(csv_path)
            
            # Format the response with latest data info
            base_message = f"✅ [{timestamp}] Data collection completed for {stock_symbol}!"
            
            if mst_time and latest_price is not None:
                data_info = f"\n🕐 Latest Data: {mst_time}\n💰 Price: ${latest_price:.2f}"
            else:
                data_info = f"\n⚠️ Could not extract latest data info: {mst_time}"
            
            return f"{base_message}{data_info}\n\nOutput:\n{result.stdout}"
        else:
            error_msg = result.stderr
            
            # Check for common dependency issues and provide helpful suggestions
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
        script_path = f"{BASE_PATH}/sentiment-fear-and-greed/getFGData.py"
        
        if not os.path.exists(script_path):
            return f"❌ Script not found: {script_path}"
        
        cmd = [sys.executable, script_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.returncode == 0:
            return f"✅ [{timestamp}] Sentiment data collection completed!\n\nOutput:\n{result.stdout}"
        else:
            return f"❌ [{timestamp}] Sentiment data collection failed.\n\nError:\n{result.stderr}\n\nStdout:\n{result.stdout}"
            
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
    
    # Global sentiment data section
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
                t_button = gr.Button("Predict Future", variant="primary")
            
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
                msft_button = gr.Button("Predict Future", variant="primary")
            
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
                gd_button = gr.Button("Predict Future", variant="primary")
            
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
        
        **Window Size:** `{WINDOW_SIZE}`
        
        **Output Directory:** `{OUTPUT_DIR}`
        
        **New Feature:** All predictions now use the `--future` flag for actual future predictions!
        
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
    t_button.click(predict_t, outputs=[t_output, t_results])
    
    # MSFT buttons
    get_data_msft_btn.click(get_data_msft, outputs=msft_output)
    msft_button.click(predict_msft, outputs=[msft_output, msft_results])
    
    # GD buttons
    get_data_gd_btn.click(get_data_gd, outputs=gd_output)
    gd_button.click(predict_gd, outputs=[gd_output, gd_results])

if __name__ == "__main__":
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
