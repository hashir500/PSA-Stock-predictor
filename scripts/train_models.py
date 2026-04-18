import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os
import json

TICKERS = {
    'HUBC': 'HUBC.KA', 'MARI': 'MARI.KA', 'PSO': 'PSO.KA', 'OGDC': 'OGDC.KA',
    'PPL': 'PPL.KA', 'SEARL': 'SEARL.KA', 'EFERT': 'EFERT.KA', 'FFC': 'FFC.KA',
    'MEZAN': 'MEBL.KA', 'NATF': 'NATF.KA', 'TGL': 'TGL.KA', 'SYS': 'SYS.KA'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_and_train():
    metrics = {}
    
    for name, ticker in TICKERS.items():
        print(f"Processing {name} ({ticker})...")
        try:
            # Fetch 5 years of daily data for vast training scale
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            if df.empty:
                print(f"WARNING: No data for {name}. Skipping.")
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Relative Target Percentages (Against Today's Open)
            df['Target_High_Pct'] = (df['High'] - df['Open']) / df['Open']
            df['Target_Low_Pct'] = (df['Low'] - df['Open']) / df['Open']
            df['Target_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Relative Training Features (Available early morning based on prevailing context)
            df['Prev_Close'] = df['Close'].shift(1)
            df['Prev_High'] = df['High'].shift(1)
            df['Prev_Low'] = df['Low'].shift(1)
            
            df['Gap_Open'] = (df['Open'] / df['Prev_Close']) - 1
            df['Prev_Range'] = (df['Prev_High'] - df['Prev_Low']) / df['Prev_Close']
            df['Return_1d'] = df['Close'].shift(1) / df['Close'].shift(2) - 1
            
            df = df.dropna()
            if len(df) < 50:
                print(f"WARNING: Not enough data for {name}. Skipping.")
                continue

            df.to_csv(f"{DATA_DIR}/{name}_historical.csv")
            
            # Prepare independent relative arrays
            X = df[['Gap_Open', 'Prev_Range', 'Return_1d']]
            y_high = df['Target_High_Pct']
            y_low = df['Target_Low_Pct']
            y_close = df['Target_Close_Pct']
            
            # Train test split for pseudo-evaluation metric tracking
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            
            # Subsampled XGBoost Regressor tuned to ignore extreme market cap noise
            model_h = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8)
            model_l = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8)
            model_c = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8)
            
            model_h.fit(X_train, y_high.iloc[:split_idx])
            model_l.fit(X_train, y_low.iloc[:split_idx])
            model_c.fit(X_train, y_close.iloc[:split_idx])
            
            # Evaluate back in absolute prices
            pred_c_pct = model_c.predict(X_test)
            open_prices = df['Open'].iloc[split_idx:]
            actual_close = df['Close'].iloc[split_idx:]
            
            pred_close_reconstructed = open_prices * (1 + pred_c_pct)
            mae_c = np.mean(np.abs(pred_close_reconstructed - actual_close))
            mae_pct = (mae_c / np.mean(actual_close)) * 100
            
            # Re-train on full array for absolute max data ingestion deployment
            model_h.fit(X, y_high)
            model_l.fit(X, y_low)
            model_c.fit(X, y_close)
            
            # Write optimized sub-assemblies back into memory
            joblib.dump(model_h, f"{MODELS_DIR}/{name}_high.pkl")
            joblib.dump(model_l, f"{MODELS_DIR}/{name}_low.pkl")
            joblib.dump(model_c, f"{MODELS_DIR}/{name}_close.pkl")
            
            metrics[name] = {"MAE_Close": float(round(mae_c, 2)), "MAE_Pct": float(round(mae_pct, 2))}
            print(f"Successfully trained models for {name}. Close MAE: {mae_pct:.2f}%")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    fetch_and_train()
