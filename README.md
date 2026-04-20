# PSX Pulse: Advanced Stock Predictor

**PSX Pulse** is a minimalist, ultra-premium stock prediction dashboard designed specifically for the Pakistan Stock Exchange (PSX). It utilizes an advanced XGBoost Machine Learning pipeline to dynamically predict the End-of-Day Trajectories for 12 major PSX heavyweight tickers.

### Features
*   **"Anti-Gravity" UI:** Built entirely on a custom-injected CSS glassmorphic aesthetic (Carbon-Mint and Lava-Core visual mapping).
*   **Predictive Engine:** An XGBoost Regressor trained across 5 years of daily market contexts, mapping relative variance features (`Gap_Open`, `Return_1d`, `Prev_Range`) to achieve an extreme mathematical noise floor (cross-board Model MAE of ~1.4%).
*   **Real-time Fundamental Context:** Instantly fetches the live Market Capitalization, Trailing P/E Ratios, and Dividend Yields using integrated `yfinance` bridging.
*   **Timeframe Backtesting:** A "Detailed Analysis" framework supporting multi-axis Time Series charts overlaying the **Actual** historical trails dynamically against the **Model's Simulated Predictions**.

---

## Local Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hashir500/PSA-Stock-predictor.git
   cd PSA-Stock-predictor
   ```

2. **Install Dependencies**
   It's highly recommended to use a virtual environment (`python -m venv venv`).
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-Train the ML Models**
   The repository may already include the pre-trained `.pkl` models. However, to synchronize the algorithmic weights with the latest market data, execute the backend training pipeline:
   ```bash
   python scripts/train_models.py
   ```

4. **Launch the Dashboard**
   Spin up the frontend using Streamlit.
   ```bash
   streamlit run app/main.py
   ```

---

## Deploying to Streamlit Community Cloud

Hosting this application natively on the web for free is incredibly easy using Streamlit Community Cloud:

1. **Push your code to GitHub:**
   Ensure all your latest files (especially the `requirements.txt` and everything inside the `models/` directory) are pushed to your GitHub repository.
   ```bash
   git add .
   git commit -m "Deploy Ready"
   git push origin main
   ```

2. **Deploy via Streamlit:**
   * Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
   * Click **New App**.
   * Select the `PSA-Stock-predictor` repository and the `main` branch.
   * Important: Because the Main File is inside the app folder, set the **Main file path** exactly to: `app/main.py`
   * Click **Deploy!**

Streamlit Cloud will automatically detect your `requirements.txt`, install all dependencies into the server container, and securely map your application to a live public URL!
