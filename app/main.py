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
                        'Prev_Low': float(prev['Low'])
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
            X = pd.DataFrame([{
                'Shifted_Open': data['Open'],
                'Prev_Close': data['Prev_Close'],
                'Prev_High': data['Prev_High'],
                'Prev_Low': data['Prev_Low']
            }])
            return {
                'High': models[name]['high'].predict(X)[0],
                'Low': models[name]['low'].predict(X)[0],
                'Close': models[name]['close'].predict(X)[0]
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
                
                # Check accuracy badge
                mae_str = m_info["MAE_Pct"]
                badge_html = f"<span class='badge'>MAE: {mae_str}%</span>" if mae_str != "--" else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{name} {badge_html}</h2>
                    <div style="font-size: 0.9rem; color: #888;">{ticker} | Data for: {current_data.get('Actual_Date', '')}</div>
                    <div class="price {color_class}">
                        Rs {actual_close:.2f}
                        <span style="font-size: 1rem; opacity: 0.8;">({sign}{diff:.2f})</span>
                    </div>
                    <div class="prediction">Predicted Close: <strong>{pred_close:.2f}</strong></div>
                    <div class="prediction">Predicted High: <strong>{preds['High']:.2f}</strong> | Actual High: {current_data['High']:.2f}</div>
                    <div class="prediction">Predicted Low: <strong>{preds['Low']:.2f}</strong> | Actual Low: {current_data['Low']:.2f}</div>
                    <div class="prediction" style="margin-top:5px; border-top:1px solid rgba(255,255,255,0.1); padding-top:5px;">Open: {current_data['Open']:.2f}</div>
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
