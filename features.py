from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def make_features(stock: pd.DataFrame, market: pd.DataFrame | None = None, news: pd.DataFrame | None = None) -> pd.DataFrame:
    d = stock.copy().sort_index()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    close = d["Adj Close"] if "Adj Close" in d else d["Close"]
    close = close.fillna(d["Close"])
    f = pd.DataFrame(index=d.index)
    f["ret1"] = np.log(close).diff()
    f["ret2"] = np.log(close).diff(2)
    f["ret5"] = np.log(close).diff(5)
    f["ret10"] = np.log(close).diff(10)
    f["overnight"] = np.log(d["Open"] / d["Close"].shift(1))
    f["intraday"] = np.log(d["Close"] / d["Open"])
    f["range"] = (d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)
    f["gap_high"] = d["High"] / d["Close"].shift(1) - 1
    f["gap_low"] = d["Low"] / d["Close"].shift(1) - 1
    for n in (5,10,20,60):
        ma = close.rolling(n).mean()
        f[f"price_sma{n}"] = close / ma - 1
        f[f"volatility{n}"] = f["ret1"].rolling(n).std()
    f["rsi14"] = rsi(close,14)/100
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12-ema26
    f["macd_norm"] = macd/close
    f["macd_signal"] = (macd-macd.ewm(span=9,adjust=False).mean())/close
    mid = close.rolling(20).mean(); sd=close.rolling(20).std()
    f["bollinger_z"] = (close-mid)/(2*sd.replace(0,np.nan))
    tr = pd.concat([(d.High-d.Low).abs(),(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean()/close
    lv = np.log1p(d.Volume.clip(lower=0))
    f["volume_z20"] = (lv-lv.rolling(20).mean())/lv.rolling(20).std().replace(0,np.nan)
    f["volume_change"] = lv.diff()
    if "Turnover" in d:
        f["turnover_log"] = np.log1p(pd.to_numeric(d.Turnover,errors="coerce")).diff()
    if "Transactions" in d:
        f["transactions_log"] = np.log1p(pd.to_numeric(d.Transactions,errors="coerce")).diff()
    f["dow_sin"] = np.sin(2*np.pi*f.index.dayofweek/5)
    f["dow_cos"] = np.cos(2*np.pi*f.index.dayofweek/5)
    f["month_sin"] = np.sin(2*np.pi*(f.index.month-1)/12)
    f["month_cos"] = np.cos(2*np.pi*(f.index.month-1)/12)

    if market is not None and not market.empty:
        m = market.reindex(f.index).ffill(limit=5)
        for c in m.columns:
            s = pd.to_numeric(m[c],errors="coerce")
            if c in {"vix","us10y"}:
                f[f"mkt_{c}_level"] = s.pct_change(20).clip(-2,2)
            f[f"mkt_{c}_ret1"] = np.log(s.where(s>0)).diff()
            f[f"mkt_{c}_ret5"] = np.log(s.where(s>0)).diff(5)
        if "taiex" in m:
            tr1=np.log(m.taiex).diff()
            f["beta20_taiex"]=(f.ret1.rolling(20).cov(tr1)/tr1.rolling(20).var().replace(0,np.nan)).clip(-5,5)
            f["relstrength20_taiex"]=f.ret1.rolling(20).sum()-tr1.rolling(20).sum()

    if news is not None and not news.empty:
        n = news.reindex(f.index).fillna(0)
        f = f.join(n.add_prefix("news_"), how="left")

    f = f.replace([np.inf,-np.inf],np.nan)
    return f


def make_targets(close: pd.Series, horizon=5) -> pd.DataFrame:
    lc = np.log(pd.to_numeric(close,errors="coerce"))
    return pd.DataFrame({f"y{i}": lc.shift(-i)-lc for i in range(1,horizon+1)}, index=close.index)
