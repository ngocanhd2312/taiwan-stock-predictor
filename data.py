from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import HISTORY_YEARS, MARKET_SYMBOLS

UA = {"User-Agent": "Mozilla/5.0 TaiwanStockResearch/1.0"}


class DataError(RuntimeError):
    pass


def _to_num(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in {"", "--", "---", "X", "除權", "除息"}:
        return np.nan
    s = s.replace("+", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _roc_to_date(s: str) -> pd.Timestamp:
    parts = str(s).strip().split("/")
    if len(parts) == 3 and len(parts[0]) <= 3:
        return pd.Timestamp(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
    return pd.to_datetime(s)


def _session_get(url: str, params=None, timeout=20):
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            time.sleep(0.7 * (attempt + 1))
    raise DataError(f"Network request failed: {url}: {last}")


def yahoo_chart(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='^=.')}"
    params = {
        "period1": p1, "period2": p2, "interval": "1d",
        "events": "div,splits", "includeAdjustedClose": "true",
    }
    raw = _session_get(url, params=params).json()
    err = raw.get("chart", {}).get("error")
    results = raw.get("chart", {}).get("result")
    if err or not results:
        raise DataError(f"Yahoo has no data for {symbol}: {err}")
    x = results[0]
    q = x["indicators"]["quote"][0]
    adj = x["indicators"].get("adjclose", [{}])[0].get("adjclose", q.get("close"))
    idx = pd.to_datetime(x["timestamp"], unit="s", utc=True).tz_convert("Asia/Taipei").tz_localize(None).normalize()
    df = pd.DataFrame({
        "Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
        "Close": q.get("close"), "Adj Close": adj, "Volume": q.get("volume"),
    }, index=idx)
    df.index.name = "Date"
    return df.dropna(subset=["Close"]).sort_index()


def twse_month(stock_code: str, year: int, month: int) -> pd.DataFrame:
    date = f"{year:04d}{month:02d}01"
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    raw = _session_get(url, params={"response": "json", "date": date, "stockNo": stock_code}).json()
    if raw.get("stat") != "OK" or not raw.get("data"):
        return pd.DataFrame()
    rows = []
    for r in raw["data"]:
        # date, shares, amount, open, high, low, close, change, transactions
        rows.append({
            "Date": _roc_to_date(r[0]), "Volume": _to_num(r[1]), "Turnover": _to_num(r[2]),
            "Open": _to_num(r[3]), "High": _to_num(r[4]), "Low": _to_num(r[5]),
            "Close": _to_num(r[6]), "Change": _to_num(r[7]), "Transactions": _to_num(r[8]),
        })
    return pd.DataFrame(rows).set_index("Date")


def twse_history(stock_code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    parts = []
    for p in months:
        x = twse_month(stock_code, p.year, p.month)
        if not x.empty:
            parts.append(x)
        time.sleep(0.04)
    if not parts:
        raise DataError(f"TWSE returned no history for {stock_code}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[(df.index.date >= start) & (df.index.date <= end)]


def resolve_symbol(stock_code: str, start: Optional[dt.date] = None, end: Optional[dt.date] = None):
    code = "".join(ch for ch in str(stock_code).strip().upper() if ch.isalnum())
    if code.endswith("TW") and "." in str(stock_code):
        return str(stock_code).upper(), "yahoo"
    end = end or dt.date.today()
    start = start or (end - dt.timedelta(days=45))
    for suffix, market in [(".TW", "TWSE"), (".TWO", "TPEx")]:
        sym = code + suffix
        try:
            d = yahoo_chart(sym, start, end)
            if len(d) >= 5:
                return sym, market
        except Exception:
            pass
    raise DataError(f"Could not resolve Taiwan stock code '{stock_code}' as TWSE (.TW) or TPEx (.TWO).")


def stock_history(stock_code: str, years: int = HISTORY_YEARS) -> tuple[str, str, pd.DataFrame]:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365.25 * years) + 120)
    # Resolve with Yahoo because it handles both exchange suffixes.
    sym, market = resolve_symbol(stock_code, end=end)
    code = sym.split(".")[0]
    errors = []
    if market == "TWSE":
        try:
            d = twse_history(code, start, end)
            if len(d) > 500:
                d["Adj Close"] = d["Close"]
                return sym, "TWSE-official", d.tail(900)
        except Exception as e:
            errors.append(str(e))
    try:
        d = yahoo_chart(sym, start, end)
        if len(d) < 500:
            raise DataError(f"Only {len(d)} daily observations were returned")
        return sym, f"Yahoo-{market}", d.tail(900)
    except Exception as e:
        errors.append(str(e))
    raise DataError("Unable to obtain 3-year history. " + " | ".join(errors))


def market_context(start: dt.date, end: dt.date) -> pd.DataFrame:
    out = []
    for name, sym in MARKET_SYMBOLS.items():
        try:
            x = yahoo_chart(sym, start, end)[["Close"]].rename(columns={"Close": name})
            out.append(x)
        except Exception:
            # Missing a proxy must not kill the whole forecast.
            continue
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1).sort_index()


def twse_latest_all() -> pd.DataFrame:
    """Official TWSE latest daily table; used for discovery / display, not training history."""
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    raw = _session_get(url).json()
    return pd.DataFrame(raw)


def tpex_latest_all() -> pd.DataFrame:
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    raw = _session_get(url).json()
    return pd.DataFrame(raw)
