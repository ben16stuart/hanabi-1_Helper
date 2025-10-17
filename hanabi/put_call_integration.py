# put_call_integration.py
# Functions to fetch and integrate put/call ratio data with your predictions

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def fetch_put_call_ratio(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch put/call ratio data for a given ticker.
    
    Note: yfinance doesn't directly provide put/call ratio, so we calculate it
    from options chain data. For production, consider using a paid data provider.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        exp_dates = stock.options
        
        if not exp_dates:
            print(f"⚠️ No options data available for {ticker}")
            return pd.DataFrame()
        
        # Use the nearest expiration
        nearest_exp = exp_dates[0]
        
        # Get options chain
        opt_chain = stock.option_chain(nearest_exp)
        
        # Calculate put/call ratio
        total_put_volume = opt_chain.puts['volume'].sum()
        total_call_volume = opt_chain.calls['volume'].sum()
        
        if total_call_volume > 0:
            put_call_ratio = total_put_volume / total_call_volume
        else:
            put_call_ratio = 0
        
        # Calculate put/call open interest ratio
        total_put_oi = opt_chain.puts['openInterest'].sum()
        total_call_oi = opt_chain.calls['openInterest'].sum()
        
        if total_call_oi > 0:
            put_call_oi_ratio = total_put_oi / total_call_oi
        else:
            put_call_oi_ratio = 0
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'put_call_volume_ratio': [put_call_ratio],
            'put_call_oi_ratio': [put_call_oi_ratio],
            'total_put_volume': [total_put_volume],
            'total_call_volume': [total_call_volume],
            'total_put_oi': [total_put_oi],
            'total_call_oi': [total_call_oi]
        })
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching put/call ratio for {ticker}: {e}")
        return pd.DataFrame()


def get_put_call_signal(put_call_ratio: float) -> dict:
    """
    Generate trading signal based on put/call ratio.
    
    Interpretation:
    - High ratio (>1.0): More puts than calls = Bearish sentiment = Contrarian BUY
    - Low ratio (<0.5): More calls than puts = Bullish sentiment = Contrarian SELL
    - Mid ratio (0.5-1.0): Neutral
    """
    signal_data = {
        'put_call_ratio': put_call_ratio,
        'signal': 'HOLD',
        'strength': 0.0,
        'interpretation': ''
    }
    
    if put_call_ratio > 1.2:
        signal_data['signal'] = 'STRONG_BUY'
        signal_data['strength'] = min((put_call_ratio - 1.0) / 1.0, 1.0)  # 0-1 scale
        signal_data['interpretation'] = 'Extreme bearish sentiment - Contrarian buy opportunity'
    elif put_call_ratio > 1.0:
        signal_data['signal'] = 'BUY'
        signal_data['strength'] = min((put_call_ratio - 0.8) / 0.4, 1.0)
        signal_data['interpretation'] = 'Elevated bearish sentiment - Consider buying'
    elif put_call_ratio < 0.45:
        signal_data['signal'] = 'STRONG_SELL'
        signal_data['strength'] = min((0.6 - put_call_ratio) / 0.6, 1.0)
        signal_data['interpretation'] = 'Extreme bullish sentiment - Contrarian sell signal'
    elif put_call_ratio < 0.6:
        signal_data['signal'] = 'SELL'
        signal_data['strength'] = min((0.8 - put_call_ratio) / 0.4, 1.0)
        signal_data['interpretation'] = 'Elevated bullish sentiment - Consider selling'
    else:
        signal_data['signal'] = 'HOLD'
        signal_data['strength'] = 0.3
        signal_data['interpretation'] = 'Neutral sentiment - No strong signal'
    
    return signal_data


def combine_model_and_putcall(model_prediction: dict, put_call_data: dict) -> dict:
    """
    Combine your transformer model prediction with put/call ratio signal.
    
    Args:
        model_prediction: Dict with 'predicted_direction', 'direction_probability', 'confidence'
        put_call_data: Dict from get_put_call_signal()
    
    Returns:
        Combined signal with adjusted confidence
    """
    model_dir = model_prediction.get('predicted_direction', 'NEUTRAL')
    model_prob = model_prediction.get('direction_probability', 0.5)
    model_conf = model_prediction.get('confidence', 0.5)
    
    pc_signal = put_call_data['signal']
    pc_strength = put_call_data['strength']
    
    # Determine agreement
    model_bullish = model_dir == 'UP'
    model_bearish = model_dir == 'DOWN'
    pc_bullish = 'BUY' in pc_signal
    pc_bearish = 'SELL' in pc_signal
    
    agreement = False
    conflict = False
    
    if (model_bullish and pc_bullish) or (model_bearish and pc_bearish):
        agreement = True
    elif (model_bullish and pc_bearish) or (model_bearish and pc_bullish):
        conflict = True
    
    # Adjust confidence based on agreement/conflict
    combined_confidence = model_conf
    reasoning = []
    
    if agreement:
        # Boost confidence when both agree
        combined_confidence = min(model_conf + (pc_strength * 0.2), 1.0)
        reasoning.append(f"✅ Put/Call ratio confirms model prediction ({put_call_data['interpretation']})")
        final_signal = model_dir
    elif conflict:
        # Reduce confidence when they conflict
        combined_confidence = max(model_conf - (pc_strength * 0.3), 0.0)
        reasoning.append(f"⚠️ Put/Call ratio conflicts with model ({put_call_data['interpretation']})")
        final_signal = 'HOLD' if combined_confidence < 0.4 else model_dir
    else:
        # Neutral put/call, keep model prediction
        reasoning.append(f"🟡 Put/Call ratio is neutral ({put_call_data['interpretation']})")
        final_signal = model_dir
    
    combined = {
        'final_direction': final_signal,
        'combined_confidence': combined_confidence,
        'model_prediction': model_dir,
        'model_probability': model_prob,
        'model_confidence': model_conf,
        'put_call_ratio': put_call_data['put_call_ratio'],
        'put_call_signal': pc_signal,
        'put_call_strength': pc_strength,
        'agreement': agreement,
        'conflict': conflict,
        'reasoning': ' | '.join(reasoning)
    }
    
    return combined


def add_put_call_to_csv(ticker: str, csv_path: str, output_path: str = None):
    """
    Add put/call ratio data to your existing hourly_data.csv
    
    Args:
        ticker: Stock symbol
        csv_path: Path to your hourly_data.csv
        output_path: Where to save enhanced CSV (defaults to same path)
    """
    try:
        # Load existing data
        df = pd.read_csv(csv_path)
        
        # Get put/call data
        pc_data = fetch_put_call_ratio(ticker)
        
        if pc_data.empty:
            print(f"⚠️ No put/call data available, keeping original CSV")
            return df
        
        # Add put/call columns (fill with latest value for all rows)
        latest_pc_ratio = pc_data['put_call_volume_ratio'].iloc[0]
        latest_pc_oi = pc_data['put_call_oi_ratio'].iloc[0]
        
        df['put_call_volume_ratio'] = latest_pc_ratio
        df['put_call_oi_ratio'] = latest_pc_oi
        
        # Save
        output_path = output_path or csv_path
        df.to_csv(output_path, index=False)
        
        print(f"✅ Added put/call ratio to {output_path}")
        print(f"   Current P/C Volume Ratio: {latest_pc_ratio:.3f}")
        print(f"   Current P/C OI Ratio: {latest_pc_oi:.3f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error adding put/call data: {e}")
        return None


# Example usage in your predict.py or as a standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    args = parser.parse_args()
    
    # Fetch and display put/call data
    print(f"\n📊 Fetching Put/Call Ratio for {args.ticker}...")
    pc_df = fetch_put_call_ratio(args.ticker)
    
    if not pc_df.empty:
        print(f"\n{pc_df.to_string()}")
        
        ratio = pc_df['put_call_volume_ratio'].iloc[0]
        signal = get_put_call_signal(ratio)
        
        print(f"\n🎯 Put/Call Trading Signal:")
        print(f"   Signal: {signal['signal']}")
        print(f"   Strength: {signal['strength']:.2f}")
        print(f"   Interpretation: {signal['interpretation']}")
        
        # Add to CSV
        add_put_call_to_csv(args.ticker, args.csv_path)
    else:
        print(f"❌ Could not fetch put/call data for {args.ticker}")