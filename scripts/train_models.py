import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os
import json

# Ticker mapping for PSX
TICKERS = {
    'HUBC': 'HUBC.KA',
    'MARI': 'MARI.KA',
    'PSO': 'PSO.KA',
    'OGDC': 'OGDC.KA',
    'PPL': 'PPL.KA',
    'SEARL': 'SEARL.KA',
    'EFERT': 'EFERT.KA',
    'FFC': 'FFC.KA',
    'MEZAN': 'MEBL.KA',
    'NATF': 'NATF.KA',
    'TGL': 'TGL.KA',
    'SYS': 'SYS.KA'
}

DATA_DIR = '../data'
MODELS_DIR = '../models'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_and_train():
    metrics = {}
    
    for name, ticker in TICKERS.items():
        print(f"Processing {name} ({ticker})...")
        try:
            # Fetch 2 years of daily data
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                print(f"WARNING: No data for {name}. Skipping.")
                continue
            
            # Since yfinance returns MultiIndex columns sometimes if you request multiple or just formatting
            # Let's clean the columns just in case
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten the MultiIndex if present
                df.columns = [col[0] for col in df.columns]

            # Feature engineering
            df['Target_High'] = df['High']
            df['Target_Low'] = df['Low']
            df['Target_Close'] = df['Close']
            
            # Features available at beginning of day
            df['Shifted_Open'] = df['Open'] # Open for predicting the same day's H/L/C
            df['Prev_Close'] = df['Close'].shift(1)
            df['Prev_High'] = df['High'].shift(1)
            df['Prev_Low'] = df['Low'].shift(1)
            
            df = df.dropna()
            if len(df) < 50:
                print(f"WARNING: Not enough data for {name}. Skipping.")
                continue

            df.to_csv(f"{DATA_DIR}/{name}_historical.csv")
            
            # Prepare data
            X = df[['Shifted_Open', 'Prev_Close', 'Prev_High', 'Prev_Low']]
            y_high = df['Target_High']
            y_low = df['Target_Low']
            y_close = df['Target_Close']
            
            # Train test split (time series simple split for metric info)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            
            # Models
            model_h = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            model_l = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            model_c = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            
            model_h.fit(X_train, y_high.iloc[:split_idx])
            model_l.fit(X_train, y_low.iloc[:split_idx])
            model_c.fit(X_train, y_close.iloc[:split_idx])
            
            # Simple eval
            pred_c = model_c.predict(X_test)
            mae_c = np.mean(np.abs(pred_c - y_close.iloc[split_idx:]))
            mae_pct = (mae_c / np.mean(y_close.iloc[split_idx:])) * 100
            
            # Re-train on full dataset for deployment
            model_h.fit(X, y_high)
            model_l.fit(X, y_low)
            model_c.fit(X, y_close)
            
            # Save models
            joblib.dump(model_h, f"{MODELS_DIR}/{name}_high.pkl")
            joblib.dump(model_l, f"{MODELS_DIR}/{name}_low.pkl")
            joblib.dump(model_c, f"{MODELS_DIR}/{name}_close.pkl")
            
            metrics[name] = {"MAE_Close": round(mae_c, 2), "MAE_Pct": round(float(mae_pct), 2)}
            print(f"Successfully trained models for {name}. Close MAE: {mae_pct:.2f}%")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    fetch_and_train()
