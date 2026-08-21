# Taiwan Stock AI — 5-Day Predictor (Final build)

Research application for **TWSE and TPEx** stocks. Type a Taiwan stock code (for example `2330`, `2317`, `2454`, `5347`) and the app resolves `.TW` / `.TWO`, obtains roughly three years of daily price history, builds technical and cross-market features, converts recent market/company news into BERTopic features, trains a leakage-aware BiLSTM+self-attention network, and outputs direct 1–5 trading-day price forecasts.

## Data design

- **Stock history:** official TWSE monthly `STOCK_DAY` first for listed stocks; Yahoo Chart API is a resilient `.TW` / `.TWO` fallback and is used for TPEx history.
- **Market context:** TAIEX, S&P 500, NASDAQ, Philadelphia Semiconductor Index, VIX, USD/TWD, U.S. 10-year yield, WTI, and gold where available.
- **News:** company/ticker + Taiwan-market + semiconductor/AI + Fed/inflation + US/China/trade + FX + oil + geopolitics queries using Google News RSS and GDELT DOC. Public headline archives are imperfect; the code records how many headlines it actually received.
- **News representation:** multilingual BERTopic (preferred) produces topic proportions. The model adds daily topic shocks and rolling topic variances. If BERTopic dependencies/model download are unavailable, it transparently falls back to TF-IDF + MiniBatchKMeans and labels that fact in the UI.

## Model design

- Predicts **five cumulative future log returns directly**, then converts them to TWD prices.
- 60-trading-day input sequence.
- Robust scaling fitted on training data only.
- BiLSTM + multi-head self-attention.
- Chronological 70/15/15 train/validation/test split, with a 5-day purge at boundaries.
- Early stopping, Huber loss, gradient clipping.
- Validation against a zero-return/random-walk baseline and Ridge baseline.
- Empirical 80% prediction range from validation residuals (not a calibrated probabilistic guarantee).

## Install and run

Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

CLI:

```bash
python scripts/predict_cli.py 2330 --epochs 35
```

Offline core smoke test (does not require web access):

```bash
PYTHONPATH=. python tests/smoke_test.py
```

## Deploy to Streamlit Community Cloud

1. Put this project in a GitHub repository.
2. In Streamlit Community Cloud choose **Create app**, select the repo, and set the entry point to `app.py`.
3. Use Python 3.11 if selectable.
4. First BERTopic startup can be memory-heavy because the multilingual sentence-transformer is loaded/downloaded. For a low-memory free host, run with news disabled first or precompute/cache topic features.

## Limitations

No model can guarantee five-day stock prices. Market/news archives can be incomplete, corporate actions can distort raw closes, and structural breaks can invalidate historical relationships. The website exposes validation/test diagnostics instead of hiding poor results. It is for research/education, not financial advice.
