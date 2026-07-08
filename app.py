import os
from typing import TypedDict, List, Optional

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="AI Stock Market Predictor", page_icon="📈", layout="centered")

st.title("📈 AI Stock Market Predictor")
st.write(
    "Enter a stock symbol to pull **live market data + recent news**, then get a "
    "quantitative next-day price prediction combined with an AI-generated sentiment outlook."
)


# ── Secrets / API keys — resolved server-side, never shown in the UI ───────
def get_secret(name: str) -> Optional[str]:
    """Read from Streamlit secrets first (for Streamlit Cloud), then env vars (for local/.env)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
start = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))
n_days = st.sidebar.slider("Lookback Days (window)", 10, 60, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Analysis")
llm_provider = st.sidebar.radio("LLM Provider", ["Groq (Llama 3.3)", "OpenAI (GPT-4o-mini)"])
use_llm = st.sidebar.checkbox("Include AI news/sentiment analysis", value=True)

if use_llm:
    if llm_provider.startswith("Groq") and not GROQ_API_KEY:
        st.sidebar.warning("GROQ_API_KEY not found in secrets. AI analysis will be skipped.")
    if llm_provider.startswith("OpenAI") and not OPENAI_API_KEY:
        st.sidebar.warning("OPENAI_API_KEY not found in secrets. AI analysis will be skipped.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Try these symbols:**")
st.sidebar.markdown("🇺🇸 `AAPL` `TSLA` `GOOGL` `AMZN`")
st.sidebar.markdown("🇮🇳 `INFY.NS` `TCS.NS` `RELIANCE.NS`")


# ── Cached, time-boxed data fetchers (keep things "live" without hammering the API) ──
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol: str, start_date, end_date) -> pd.DataFrame:
    return yf.download(symbol, start=start_date, end=end_date, progress=False)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_price(symbol: str) -> Optional[float]:
    try:
        fast = yf.Ticker(symbol).fast_info
        return float(fast["last_price"])
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(symbol: str, limit: int = 6) -> List[dict]:
    """Pull recent headlines for the ticker. Handles both old and new yfinance news schemas."""
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        raw = []

    cleaned = []
    for item in raw[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else item.get("publisher")
        canonical = content.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else item.get("link")
        if title:
            cleaned.append({"title": title, "publisher": publisher or "Unknown", "link": link or ""})
    return cleaned


# ── LangGraph pipeline ───────────────────────────────────────
# All per-request inputs (n_days, use_llm, provider, key) travel inside the state dict
# rather than as module-level globals, so concurrent Streamlit sessions never cross-talk.
class PipelineState(TypedDict, total=False):
    ticker: str
    df: pd.DataFrame
    news: List[dict]
    live_price: Optional[float]
    n_days: int
    use_llm: bool
    llm_provider: str
    api_key: Optional[str]

    close_prices: np.ndarray
    last_close: float
    predicted_price: float
    model_score: float
    test_preds: np.ndarray
    actual_test: np.ndarray
    llm_summary: str


def node_technical_model(state: PipelineState) -> PipelineState:
    df = state["df"]
    window = state["n_days"]

    close_prices = df["Close"].values.flatten()
    prices = close_prices.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i].flatten())
        y.append(scaled[i][0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    last_window = scaled[-window:].flatten().reshape(1, -1)
    pred_scaled = model.predict(last_window)
    predicted_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    test_preds_scaled = model.predict(X_test).reshape(-1, 1)
    test_preds = scaler.inverse_transform(test_preds_scaled).flatten()
    actual_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    state["close_prices"] = close_prices
    state["last_close"] = float(close_prices[-1])
    state["predicted_price"] = float(predicted_price)
    state["model_score"] = float(score)
    state["test_preds"] = test_preds
    state["actual_test"] = actual_test
    return state


def node_llm_analysis(state: PipelineState) -> PipelineState:
    if not state.get("use_llm"):
        state["llm_summary"] = ""
        return state

    api_key = state.get("api_key")
    if not api_key:
        state["llm_summary"] = "_AI analysis skipped — no API key configured in secrets._"
        return state

    news = state.get("news") or []
    news_lines = "\n".join(f"- {n['title']} ({n['publisher']})" for n in news) or "No recent headlines found."
    current_price = state.get("live_price") or state["last_close"]

    prompt = (
        f"You are a financial analyst assistant. Ticker: {state['ticker']}.\n"
        f"Last close price: ${state['last_close']:.2f}\n"
        f"Current/live price: ${current_price:.2f}\n"
        f"Quantitative model's next-day prediction: ${state['predicted_price']:.2f} "
        f"(holdout R^2: {state['model_score']:.2f})\n\n"
        f"Recent news headlines:\n{news_lines}\n\n"
        "In under 150 words, summarize current sentiment based on the headlines, flag any "
        "notable risks, and state whether the news sentiment supports or contradicts the "
        "quantitative prediction. This is educational analysis only, not financial advice."
    )

    try:
        if state["llm_provider"].startswith("Groq"):
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.3)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)

        response = llm.invoke([
            SystemMessage(content="You are a concise, careful financial analyst assistant."),
            HumanMessage(content=prompt),
        ])
        state["llm_summary"] = response.content
    except Exception as e:
        state["llm_summary"] = f"_AI analysis failed: {e}_"

    return state


@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("technical_model", node_technical_model)
    graph.add_node("llm_analysis", node_llm_analysis)
    graph.set_entry_point("technical_model")
    graph.add_edge("technical_model", "llm_analysis")
    graph.add_edge("llm_analysis", END)
    return graph.compile()


# ── Main Button ───────────────────────────────────────────
if st.button("🔮 Predict Next Day Price"):

    with st.spinner("Fetching live data, news, and running the models..."):
        df = fetch_history(ticker, start, end)

        if df.empty:
            st.error("❌ Invalid stock symbol or no data found. Try AAPL or INFY.NS")
            st.stop()

        live_price = fetch_live_price(ticker)
        news = fetch_news(ticker)

        api_key = GROQ_API_KEY if llm_provider.startswith("Groq") else OPENAI_API_KEY

        app_graph = build_graph()
        result = app_graph.invoke({
            "ticker": ticker,
            "df": df,
            "news": news,
            "live_price": live_price,
            "n_days": n_days,
            "use_llm": use_llm,
            "llm_provider": llm_provider,
            "api_key": api_key,
        })

    # ── Raw data ─────────────────────────────────────────
    st.subheader(f"📊 Raw Data — {ticker}")
    st.dataframe(df.tail(10), use_container_width=True)

    # ── Results ──────────────────────────────────────────
    st.success(f"✅ Predicted Next Day Closing Price for **{ticker}**: **${result['predicted_price']:.2f}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Last Close", f"${result['last_close']:.2f}")
    col2.metric("🔴 Live Price", f"${live_price:.2f}" if live_price else "N/A")
    col3.metric("🔮 Predicted", f"${result['predicted_price']:.2f}")
    col4.metric("🎯 Model Score", f"{result['model_score'] * 100:.1f}%")

    # ── Chart 1: Historical prices ────────────────────────
    st.subheader("📉 Historical Closing Price")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, result["close_prices"], color="royalblue", linewidth=1.5, label="Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # ── Chart 2: Actual vs Predicted ──────────────────────
    st.subheader("🔁 Actual vs Predicted (Test Set)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(result["actual_test"], label="Actual", color="green", linewidth=1.5)
    ax2.plot(result["test_preds"], label="Predicted", color="red", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # ── News & AI Analysis ─────────────────────────────────
    st.subheader("📰 Recent News")
    if news:
        for n in news:
            if n["link"]:
                st.markdown(f"- [{n['title']}]({n['link']}) — *{n['publisher']}*")
            else:
                st.markdown(f"- {n['title']} — *{n['publisher']}*")
    else:
        st.write("No recent news found for this symbol.")

    if use_llm:
        st.subheader("🤖 AI Sentiment & Outlook")
        st.write(result.get("llm_summary", ""))

    
