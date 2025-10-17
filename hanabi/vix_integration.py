"""
VIX Integration Module for Stock Prediction System

This module implements the VIX Strategy for market volatility analysis.
It fetches VIX data, generates contrarian trading signals based on fear/greed levels,
and combines these signals with model predictions and put/call ratios.

VIX Strategy Logic:
- High VIX (>35): Extreme fear → Contrarian BUY signal
- Low VIX (<10): Extreme complacency → Contrarian SELL signal
- Normal VIX (15-25): HOLD/Neutral
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz


def fetch_vix_data():
    """
    Fetch current VIX (Volatility Index) data from Yahoo Finance
    
    The VIX measures expected market volatility based on S&P 500 options.
    Higher VIX = Higher expected volatility/fear
    Lower VIX = Lower expected volatility/complacency
    
    Returns:
        pd.DataFrame: DataFrame with VIX data including current level, changes, and metrics
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        
        if hist.empty:
            print("⚠️ No VIX data available")
            return pd.DataFrame()
        
        # Get latest and previous VIX levels
        latest_vix = hist['Close'].iloc[-1]
        prev_vix = hist['Close'].iloc[-2] if len(hist) > 1 else latest_vix
        vix_change = latest_vix - prev_vix
        vix_change_pct = (vix_change / prev_vix) * 100 if prev_vix != 0 else 0
        
        # Get additional metrics
        vix_high = hist['High'].iloc[-1]
        vix_low = hist['Low'].iloc[-1]
        vix_volume = hist['Volume'].iloc[-1]
        
        result = pd.DataFrame({
            'vix_level': [latest_vix],
            'vix_prev': [prev_vix],
            'vix_change': [vix_change],
            'vix_change_pct': [vix_change_pct],
            'vix_high': [vix_high],
            'vix_low': [vix_low],
            'vix_volume': [vix_volume],
            'timestamp': [datetime.now(pytz.UTC)]
        })
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching VIX data: {e}")
        return pd.DataFrame()


def get_vix_signal(vix_level):
    """
    Generate trading signal based on VIX level using contrarian VIX strategy
    
    VIX Interpretation (Contrarian Approach):
    - VIX > 35: High fear/volatility → Contrarian BUY signal (market oversold)
    - VIX 25-35: Elevated fear → Caution, potential buying opportunity
    - VIX 15-25: Normal volatility → Neutral
    - VIX 10-15: Low fear → Caution, market may be complacent
    - VIX < 10: Extreme complacency → Contrarian SELL signal (market overbought)
    
    Args:
        vix_level (float): Current VIX level
        
    Returns:
        dict: Signal information including:
            - vix_level: The input VIX level
            - signal: Trading signal (STRONG BUY, BUY, HOLD, SELL, etc.)
            - strength: Signal strength from 0 to 1
            - interpretation: Human-readable interpretation
            - market_sentiment: Current market sentiment description
    """
    signal_data = {
        'vix_level': vix_level,
        'signal': 'HOLD',
        'strength': 0.5,
        'interpretation': '',
        'market_sentiment': ''
    }
    
    # VIX > 35: Extreme Fear - Strong Buy Signal (contrarian)
    if vix_level > 35:
        signal_data['signal'] = 'STRONG BUY'
        signal_data['strength'] = min(0.95, 0.7 + (vix_level - 35) * 0.01)
        signal_data['market_sentiment'] = 'EXTREME FEAR'
        signal_data['interpretation'] = 'Market panic - Contrarian buying opportunity'
    
    # VIX 30-35: High Fear - Buy Signal
    elif vix_level > 30:
        signal_data['signal'] = 'BUY'
        signal_data['strength'] = 0.65 + (vix_level - 30) * 0.01
        signal_data['market_sentiment'] = 'HIGH FEAR'
        signal_data['interpretation'] = 'Elevated fear - Good buying opportunity'
    
    # VIX 25-30: Moderate Fear - Soft Buy
    elif vix_level > 25:
        signal_data['signal'] = 'SOFT BUY'
        signal_data['strength'] = 0.55 + (vix_level - 25) * 0.02
        signal_data['market_sentiment'] = 'MODERATE FEAR'
        signal_data['interpretation'] = 'Above-average fear - Consider buying'
    
    # VIX 20-25: Slightly Elevated - Neutral/Hold
    elif vix_level > 20:
        signal_data['signal'] = 'HOLD'
        signal_data['strength'] = 0.5
        signal_data['market_sentiment'] = 'SLIGHTLY ELEVATED'
        signal_data['interpretation'] = 'Normal to slightly elevated volatility'
    
    # VIX 15-20: Normal - Hold
    elif vix_level > 15:
        signal_data['signal'] = 'HOLD'
        signal_data['strength'] = 0.5
        signal_data['market_sentiment'] = 'NORMAL'
        signal_data['interpretation'] = 'Normal market volatility'
    
    # VIX 12-15: Low Fear - Caution
    elif vix_level > 12:
        signal_data['signal'] = 'WEAK SELL'
        signal_data['strength'] = 0.45 - (15 - vix_level) * 0.02
        signal_data['market_sentiment'] = 'LOW FEAR'
        signal_data['interpretation'] = 'Market complacency - Consider taking profits'
    
    # VIX 10-12: Very Low Fear - Soft Sell
    elif vix_level > 10:
        signal_data['signal'] = 'SOFT SELL'
        signal_data['strength'] = 0.35 - (12 - vix_level) * 0.03
        signal_data['market_sentiment'] = 'VERY LOW FEAR'
        signal_data['interpretation'] = 'High complacency - Reduce exposure'
    
    # VIX < 10: Extreme Complacency - Strong Sell Signal (contrarian)
    else:
        signal_data['signal'] = 'STRONG SELL'
        signal_data['strength'] = max(0.05, 0.25 - (10 - vix_level) * 0.05)
        signal_data['market_sentiment'] = 'EXTREME COMPLACENCY'
        signal_data['interpretation'] = 'Extreme complacency - Contrarian selling opportunity'
    
    return signal_data


def combine_signals(model_prediction, putcall_signal, vix_signal):
    """
    Combine model prediction, put/call ratio, and VIX signals into a unified recommendation
    
    This function weighs the three signals and produces an overall trading recommendation
    with confidence levels and agreement analysis.
    
    Weighting:
    - Model Prediction: 40%
    - Put/Call Ratio: 30%
    - VIX Signal: 30%
    
    Args:
        model_prediction (dict): Model prediction data with keys:
            - predicted_direction: 'UP', 'DOWN', or 'NEUTRAL'
            - direction_probability: float (0-1)
            - confidence: float (0-1)
        putcall_signal (dict): Put/call ratio signal data from put_call_integration
        vix_signal (dict): VIX signal data from get_vix_signal()
        
    Returns:
        dict: Combined analysis with:
            - overall_signal: Final trading recommendation with emoji
            - combined_value: Numeric signal value (0-1)
            - overall_confidence: 'HIGH', 'GOOD', 'MODERATE', or 'NEUTRAL'
            - agreement_status: Agreement between signals
            - reasoning: Detailed reasoning string
            - signal_breakdown: Individual signal values
    """
    # Extract model direction and confidence
    model_dir = model_prediction.get('predicted_direction', 'NEUTRAL')
    model_prob = model_prediction.get('direction_probability', 0.5)
    model_conf = model_prediction.get('confidence', 0.5)
    
    # Map text signals to numeric values (0 = STRONG SELL, 1 = STRONG BUY)
    signal_map = {
        'STRONG BUY': 1.0,
        'BUY': 0.75,
        'SOFT BUY': 0.6,
        'WEAK BUY': 0.55,
        'HOLD': 0.5,
        'WEAK SELL': 0.45,
        'SOFT SELL': 0.4,
        'SELL': 0.25,
        'STRONG SELL': 0.0
    }
    
    # Convert model prediction to signal value
    if model_dir == 'UP':
        model_signal_value = model_prob
    elif model_dir == 'DOWN':
        model_signal_value = 1 - model_prob
    else:
        model_signal_value = 0.5
    
    # Get signal values for P/C and VIX
    pc_signal_value = signal_map.get(putcall_signal.get('signal', 'HOLD'), 0.5)
    vix_signal_value = signal_map.get(vix_signal.get('signal', 'HOLD'), 0.5)
    
    # Weighted combination (Model: 40%, P/C: 30%, VIX: 30%)
    combined_value = (
        model_signal_value * 0.4 +
        pc_signal_value * 0.3 +
        vix_signal_value * 0.3
    )
    
    # Determine overall signal based on combined value
    if combined_value >= 0.75:
        overall_signal = '🟢🟢 STRONG BUY'
        overall_conf = 'HIGH'
    elif combined_value >= 0.65:
        overall_signal = '🟢 BUY'
        overall_conf = 'GOOD'
    elif combined_value >= 0.55:
        overall_signal = '🟢 SOFT BUY'
        overall_conf = 'MODERATE'
    elif combined_value >= 0.45:
        overall_signal = '🟡 HOLD'
        overall_conf = 'NEUTRAL'
    elif combined_value >= 0.35:
        overall_signal = '🔴 SOFT SELL'
        overall_conf = 'MODERATE'
    elif combined_value >= 0.25:
        overall_signal = '🔴 SELL'
        overall_conf = 'GOOD'
    else:
        overall_signal = '🔴🔴 STRONG SELL'
        overall_conf = 'HIGH'
    
    # Check for agreement/conflict between signals
    signals = [
        putcall_signal.get('signal', 'HOLD'),
        vix_signal.get('signal', 'HOLD'),
        'BUY' if model_dir == 'UP' else ('SELL' if model_dir == 'DOWN' else 'HOLD')
    ]
    
    buy_signals = sum(1 for s in signals if 'BUY' in s)
    sell_signals = sum(1 for s in signals if 'SELL' in s)
    
    # Determine agreement status
    agreement_status = 'NEUTRAL'
    if buy_signals >= 2:
        agreement_status = '✅ BULLISH CONSENSUS'
    elif sell_signals >= 2:
        agreement_status = '✅ BEARISH CONSENSUS'
    elif buy_signals >= 1 and sell_signals >= 1:
        agreement_status = '⚠️ MIXED SIGNALS'
    
    # Build detailed reasoning
    reasoning_parts = []
    
    # Model analysis
    model_desc = f"Model: {model_dir} ({model_prob:.1%} prob, {model_conf:.1%} conf)"
    reasoning_parts.append(model_desc)
    
    # P/C analysis
    pc_desc = f"P/C: {putcall_signal.get('signal', 'N/A')} (ratio: {putcall_signal.get('put_call_ratio', 0):.3f})"
    reasoning_parts.append(pc_desc)
    
    # VIX analysis
    vix_desc = f"VIX: {vix_signal.get('signal', 'N/A')} (level: {vix_signal.get('vix_level', 0):.2f} - {vix_signal.get('market_sentiment', 'N/A')})"
    reasoning_parts.append(vix_desc)
    
    combined_reasoning = " | ".join(reasoning_parts)
    
    return {
        'overall_signal': overall_signal,
        'combined_value': combined_value,
        'overall_confidence': overall_conf,
        'agreement_status': agreement_status,
        'reasoning': combined_reasoning,
        'signal_breakdown': {
            'model': model_signal_value,
            'putcall': pc_signal_value,
            'vix': vix_signal_value
        }
    }


if __name__ == "__main__":
    # Test the VIX integration
    print("Testing VIX Integration...")
    print("=" * 60)
    
    # Fetch current VIX data
    vix_data = fetch_vix_data()
    
    if not vix_data.empty:
        vix_level = vix_data['vix_level'].iloc[0]
        print(f"\n📊 Current VIX Level: {vix_level:.2f}")
        print(f"📈 Change: {vix_data['vix_change'].iloc[0]:+.2f} ({vix_data['vix_change_pct'].iloc[0]:+.2f}%)")
        print(f"📊 High: {vix_data['vix_high'].iloc[0]:.2f} | Low: {vix_data['vix_low'].iloc[0]:.2f}")
        
        signal = get_vix_signal(vix_level)
        print(f"\n🎯 VIX Signal: {signal['signal']}")
        print(f"💪 Strength: {signal['strength']:.2f}")
        print(f"🌡️ Market Sentiment: {signal['market_sentiment']}")
        print(f"📝 Interpretation: {signal['interpretation']}")
        
        print("\n" + "=" * 60)
        print("Testing different VIX levels:")
        print("=" * 60)
        
        test_levels = [8, 12, 18, 28, 40]
        for level in test_levels:
            test_signal = get_vix_signal(level)
            print(f"\nVIX {level:5.1f}: {test_signal['signal']:15s} | {test_signal['market_sentiment']:20s}")
            print(f"          {test_signal['interpretation']}")
        
    else:
        print("❌ Failed to fetch VIX data")
        print("\nMake sure you have internet connection and yfinance is installed:")
        print("pip install yfinance")