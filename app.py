import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Predictor")
st.write("Enter a stock symbol below and click **Predict** to see the forecast.")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
start  = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end    = st.sidebar.date_input("End Date",   pd.to_datetime("2024-01-01"))
n_days = st.sidebar.slider("Lookback Days (window)", 10, 60, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Try these symbols:**")
st.sidebar.markdown("🇺🇸 `AAPL` `TSLA` `GOOGL` `AMZN`")
st.sidebar.markdown("🇮🇳 `INFY.NS` `TCS.NS` `RELIANCE.NS`")

# ── Main Button ───────────────────────────────────────────
if st.button("🔮 Predict Next Day Price"):

    with st.spinner("Fetching data and training model..."):

        # 1. Download data
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            st.error("❌ Invalid stock symbol or no data found. Try AAPL or INFY.NS")
            st.stop()

        # 2. Show raw data
        st.subheader(f"📊 Raw Data — {ticker}")
        st.dataframe(df.tail(10), use_container_width=True)

        # 3. Scale closing prices — FIX: flatten to 1D properly
        close_prices = df['Close'].values.flatten()  # ✅ ensure 1D
        prices = close_prices.reshape(-1, 1)          # ✅ reshape to 2D for scaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)         # scaled is now (n, 1)

        # 4. Create sequences
        X, y = [], []
        for i in range(n_days, len(scaled)):
            X.append(scaled[i - n_days:i].flatten())
            y.append(scaled[i][0])

        X, y = np.array(X), np.array(y)

        # 5. Train/test split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 6. Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 7. Score
        score = model.score(X_test, y_test)

        # 8. Predict next day — FIX: reshape correctly
        last_window     = scaled[-n_days:].flatten().reshape(1, -1)  # (1, n_days)
        pred_scaled     = model.predict(last_window)                  # (1,)
        predicted_price = scaler.inverse_transform(                   # ✅ 2D input
                            pred_scaled.reshape(-1, 1)
                          )[0][0]

        # 9. Predict on test set
        test_preds_scaled = model.predict(X_test).reshape(-1, 1)     # ✅ 2D
        test_preds        = scaler.inverse_transform(test_preds_scaled).flatten()
        actual_test       = scaler.inverse_transform(
                              y_test.reshape(-1, 1)
                            ).flatten()

    # ── Results ───────────────────────────────────────────
    last_close = float(df['Close'].iloc[-1].values[0])
    st.success(f"✅ Predicted Next Day Closing Price for **{ticker}**: **${predicted_price:.2f}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Last Close",  f"${last_close:.2f}")
    col2.metric("🔮 Predicted",   f"${predicted_price:.2f}")
    col3.metric("🎯 Model Score", f"{score * 100:.1f}%")

    # ── Chart 1: Historical prices ────────────────────────
    st.subheader("📉 Historical Closing Price")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, close_prices, color='royalblue', linewidth=1.5, label='Close Price')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # ── Chart 2: Actual vs Predicted ──────────────────────
    st.subheader("🔁 Actual vs Predicted (Test Set)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(actual_test, label='Actual',    color='green', linewidth=1.5)
    ax2.plot(test_preds,  label='Predicted', color='red',   linewidth=1.5, linestyle='--')
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.info("⚠️ This is for educational purposes only. Do not use for real trading decisions.")