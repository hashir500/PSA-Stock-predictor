import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import os
import json
import numpy as np

# Page Config
st.set_page_config(page_title="PSX Pulse", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Anti-Gravity Glassmorphism
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #121212 !important;
        background-image: radial-gradient(circle at 15% 50%, rgba(0, 255, 194, 0.05), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(252, 92, 101, 0.05), transparent 25%);
        color: #ffffff;
    }

    /* Floating Glassmorphic Cards */
    div[data-testid="stVerticalBlock"] > div > div.metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Hover effect for Anti-Gravity */
    div[data-testid="stVerticalBlock"] > div > div.metric-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 45px 0 rgba(0, 255, 194, 0.1);
        border: 1px solid rgba(0, 255, 194, 0.2);
    }
    
    /* Typography inside cards */
    .metric-card h2 {
        color: #ffffff;
        font-size: 1.8rem;
        margin-bottom: 5px;
        margin-top: 0;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .metric-card .price {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-card .prediction {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 5px;
    }
    
    /* Accents */
    .carbon-mint { color: #00FFC2; }
    .lava-core { color: #FC5C65; }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(0, 255, 194, 0.1);
        color: #00FFC2;
        border: 1px solid rgba(0, 255, 194, 0.3);
    }

    /* Helper container for grid */
    .st-emotion-cache-1wmy9hl { 
        gap: 2rem !important; 
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 PSX Pulse \n *Anti-Gravity Precision Tracking*")
st.markdown("---")

TICKERS = {
    'HUBC': 'HUBC.KA', 'MARI': 'MARI.KA', 'PSO': 'PSO.KA', 'OGDC': 'OGDC.KA',
    'PPL': 'PPL.KA', 'SEARL': 'SEARL.KA', 'EFERT': 'EFERT.KA', 'FFC': 'FFC.KA',
    'MEZAN': 'MEBL.KA', 'NATF': 'NATF.KA', 'TGL': 'TGL.KA', 'SYS': 'SYS.KA'
}

MODELS_DIR = '../models'
DATA_DIR = '../data'

@st.cache_data(ttl=300) # Cache for 5 mins
def fetch_target_data(target_date):
    target_data = {}
    import pandas as pd
    target_dt = pd.to_datetime(target_date)
    start_date = (target_dt - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
    end_date = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                
                # Filter to only rows on or before the target date
                df = df[df.index <= pd.Timestamp(target_date)]
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    old_prev = df.iloc[-3] if len(df) > 2 else prev
                    actual_date = df.index[-1].strftime('%Y-%m-%d')
                    
                    target_data[name] = {
                        'Actual_Date': actual_date,
                        'Open': float(latest['Open']),
                        'High': float(latest['High']),
                        'Low': float(latest['Low']),
                        'Close': float(latest['Close']),
                        'Volume': float(latest['Volume']),
                        'Prev_Close': float(prev['Close']),
                        'Prev_High': float(prev['High']),
                        'Prev_Low': float(prev['Low']),
                        'Old_Prev_Close': float(old_prev['Close'])
                    }
        except Exception:
            pass
    return target_data

@st.cache_resource
def load_models():
    models = {}
    for name in TICKERS.keys():
        try:
            m_h = joblib.load(f"{MODELS_DIR}/{name}_high.pkl")
            m_l = joblib.load(f"{MODELS_DIR}/{name}_low.pkl")
            m_c = joblib.load(f"{MODELS_DIR}/{name}_close.pkl")
            models[name] = {'high': m_h, 'low': m_l, 'close': m_c}
        except:
            pass
    return models

@st.cache_data
def load_metrics():
    try:
        with open(f"{MODELS_DIR}/metrics.json", "r") as f:
            return json.load(f)
    except:
        return {}

def make_prediction(models, data, name):
    try:
        if name in models and data is not None:
            gap_open = (data['Open'] / data['Prev_Close']) - 1
            prev_range = (data['Prev_High'] - data['Prev_Low']) / data['Prev_Close']
            return_1d = (data['Prev_Close'] / data['Old_Prev_Close']) - 1
            
            X = pd.DataFrame([{
                'Gap_Open': gap_open,
                'Prev_Range': prev_range,
                'Return_1d': return_1d
            }])
            
            pred_h_pct = models[name]['high'].predict(X)[0]
            pred_l_pct = models[name]['low'].predict(X)[0]
            pred_c_pct = models[name]['close'].predict(X)[0]
            
            open_price = data['Open']
            return {
                'High': open_price * (1 + pred_h_pct),
                'Low': open_price * (1 + pred_l_pct),
                'Close': open_price * (1 + pred_c_pct)
            }
    except Exception:
        pass
    return None

def main():
    import datetime
    
    col_title, col_picker = st.columns([3, 1])
    with col_title:
        st.markdown("### Market Dashboard")
    with col_picker:
        min_date = datetime.date.today() - datetime.timedelta(days=700)
        target_date = st.date_input("Select Target Date", datetime.date.today(), min_value=min_date, max_value=datetime.date.today())
        
    today_data = fetch_target_data(target_date)
    models = load_models()
    metrics = load_metrics()
    
    if not models:
        st.warning("Models are not trained yet. Run the training script first.")
        return
    
    # Create rows of 3 columns
    cols = st.columns(3)
    
    for idx, (name, ticker) in enumerate(TICKERS.items()):
        col = cols[idx % 3]
        
        # Data logic
        current_data = today_data.get(name)
        preds = make_prediction(models, current_data, name)
        m_info = metrics.get(name, {"MAE_Pct": "--"})
        
        with col:
            if current_data and preds:
                # Decide if actuals are available (if the market is deep in the day, High/Low track Actuals)
                # We'll just show actual vs predicted simply
                actual_close = current_data['Close']
                pred_close = preds['Close']
                diff = actual_close - pred_close
                color_class = "carbon-mint" if diff >= 0 else "lava-core"
                sign = "+" if diff >= 0 else ""
                
                h_diff = current_data['High'] - preds['High']
                h_sign = "+" if h_diff >= 0 else ""
                h_color = "#00FFC2" if h_diff >= 0 else "#FC5C65"

                l_diff = current_data['Low'] - preds['Low']
                l_sign = "+" if l_diff >= 0 else ""
                l_color = "#00FFC2" if l_diff >= 0 else "#FC5C65"
                
                # Check accuracy badge
                mae_str = m_info["MAE_Pct"]
                badge_html = f"<span class='badge'>MAE: {mae_str}%</span>" if mae_str != "--" else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{name} {badge_html}</h2>
                    <div style="font-size: 0.9rem; color: #888;">{ticker} | Date: {current_data.get('Actual_Date', '')}</div>
                    <div class="price {color_class}">
                        Rs {actual_close:.2f}
                        <span style="font-size: 1rem; opacity: 0.8;">({sign}{diff:.2f})</span>
                    </div>
                    <table style="width:100%; text-align:left; font-size:0.85rem; border-collapse:collapse; margin-top:15px; background: rgba(0,0,0,0.2); border-radius:8px; overflow:hidden;">
                        <tr style="background: rgba(255,255,255,0.05); border-bottom:1px solid rgba(255,255,255,0.1);">
                            <th style="padding:6px 10px; color:#aaa; font-weight:500;">Metric</th>
                            <th style="padding:6px 10px; color:#aaa; font-weight:500;">Predicted</th>
                            <th style="padding:6px 10px; color:#aaa; font-weight:500;">Actual</th>
                            <th style="padding:6px 10px; color:#aaa; font-weight:500;">Diff</th>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px 10px; color:#ddd;">High</td>
                            <td style="padding:6px 10px;">{preds['High']:.2f}</td>
                            <td style="padding:6px 10px;">{current_data['High']:.2f}</td>
                            <td style="padding:6px 10px; color:{h_color};">{h_sign}{h_diff:.2f}</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px 10px; color:#ddd;">Low</td>
                            <td style="padding:6px 10px;">{preds['Low']:.2f}</td>
                            <td style="padding:6px 10px;">{current_data['Low']:.2f}</td>
                            <td style="padding:6px 10px; color:{l_color};">{l_sign}{l_diff:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 10px; color:#ddd;">Open</td>
                            <td style="padding:6px 10px; color:#888;">--</td>
                            <td style="padding:6px 10px;">{current_data['Open']:.2f}</td>
                            <td style="padding:6px 10px; color:#888;">--</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{name}</h2>
                    <div style="font-size: 0.9rem; color: #888;">{ticker}</div>
                    <div style="margin-top: 20px; color: #555;">Data Unavailable</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
