from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from src.predictor import train_and_predict

st.set_page_config(page_title="Taiwan Stock AI — 5-Day Forecast", page_icon="📈", layout="wide")
st.title("Taiwan Stock AI — 5-Day Forecast")
st.caption("3-year OHLCV + technical indicators + Taiwan/global market context + BERTopic news signals + BiLSTM self-attention")

with st.sidebar:
    st.header("Model settings")
    epochs=st.slider("Maximum training epochs",15,80,35,5)
    news_days=st.slider("News lookback",90,365,365,30)
    use_news=st.toggle("Use news / BERTopic",value=True)
    st.caption("A ticker is trained on demand. Models use chronological train/validation/test splits and a random-walk benchmark.")

code=st.text_input("Taiwan stock code",value="2330",placeholder="2330, 2317, 2454, 5347 …")
company=st.text_input("Company name (optional; improves news search)",value="",placeholder="e.g. 台積電 / TSMC")
run=st.button("Predict next 5 trading days",type="primary",use_container_width=True)

if run:
    try:
        with st.status(f"Training / evaluating {code} …",expanded=True) as status:
            st.write("1/4 Downloading ~3 years of stock and cross-market data")
            st.write("2/4 Collecting public company + Taiwan/global market headlines")
            st.write("3/4 Converting news into BERTopic topic proportions, shocks and variances")
            st.write("4/4 Training BiLSTM + self-attention with chronological validation")
            r=train_and_predict(code,epochs=epochs,news_days=news_days,company_name=company or None,use_news=use_news)
            status.update(label="Forecast completed",state="complete")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Resolved ticker",r.symbol)
        c2.metric("Latest close",f"NT$ {r.latest_close:,.2f}")
        c3.metric("Day-1 direction accuracy",f"{r.metrics['validation_direction_accuracy_day1']:.1%}")
        c4.metric("News headlines",f"{r.topic_meta.get('headline_count',0):,}")
        st.caption(f"Price source: {r.source} · Topic engine: {r.topic_meta.get('topic_engine','none')}")

        last=pd.Timestamp(r.history.index[-1])
        dates=pd.bdate_range(last+pd.Timedelta(days=1),periods=5)
        table=pd.DataFrame({
            "Trading day":dates.date,
            "Predicted price (NT$)":np.round(r.predicted_prices,2),
            "80% empirical lower":np.round(r.lower_prices,2),
            "80% empirical upper":np.round(r.upper_prices,2),
            "Expected cumulative return":[f"{np.expm1(x):.2%}" for x in r.predicted_returns],
        })
        st.subheader("5-trading-day forecast")
        st.dataframe(table,use_container_width=True,hide_index=True)

        hist=r.history[["Close"]].tail(120).rename(columns={"Close":"Historical close"})
        pred=pd.DataFrame({"Forecast":r.predicted_prices,"Lower":r.lower_prices,"Upper":r.upper_prices},index=dates)
        st.line_chart(pd.concat([hist,pred],axis=1))

        st.subheader("Out-of-sample diagnostics")
        a,b,c=st.columns(3)
        a.metric("Validation MAE (log return)",f"{r.metrics['validation_mae_logreturn']:.5f}")
        b.metric("Random-walk MAE",f"{r.metrics['random_walk_mae_logreturn']:.5f}")
        c.metric("Test day-1 direction",f"{r.metrics['test_validation_direction_accuracy_day1']:.1%}")
        if r.metrics.get("beats_random_walk_validation"):
            st.success("The neural model beat the zero-return/random-walk baseline on this validation split.")
        else:
            st.warning("The neural model did NOT beat the zero-return/random-walk baseline on this validation split. Treat this forecast as experimental.")
        with st.expander("Full metrics"):
            st.json(r.metrics)
        with st.expander("BERTopic / recent news"):
            st.json(r.topic_meta)
            if not r.recent_news.empty:
                st.dataframe(r.recent_news[[c for c in ["datetime","title","source","query"] if c in r.recent_news]],use_container_width=True,hide_index=True)
        st.info("Research/educational model only. A short-horizon stock-price forecast is highly uncertain and is not investment advice.")
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.exception(e)
