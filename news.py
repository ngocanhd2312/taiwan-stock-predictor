from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

from .config import CACHE, MARKET_NEWS_QUERIES, RANDOM_SEED

UA = {"User-Agent": "Mozilla/5.0 TaiwanStockResearch/1.0"}


def _get(url, params=None, timeout=20):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


def google_news_rss(query: str, days: int = 365) -> pd.DataFrame:
    # Google News RSS supports relative query operators. Coverage is not guaranteed to be complete.
    q = f"({query}) when:{min(days,365)}d"
    url = "https://news.google.com/rss/search?q=" + quote_plus(q) + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    text = _get(url).text
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    rows = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        pub = item.findtext("pubDate") or ""
        link = item.findtext("link") or ""
        source_node = item.find("source")
        source = source_node.text if source_node is not None else ""
        try:
            ts = pd.to_datetime(pub, utc=True).tz_convert("Asia/Taipei").tz_localize(None)
        except Exception:
            continue
        rows.append({"datetime": ts, "title": title.strip(), "source": source, "url": link, "query": query})
    return pd.DataFrame(rows)


def gdelt_news(query: str, days: int = 365, max_records: int = 250) -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        "timespan": f"{min(days,365)}d",
        "sort": "HybridRel",
    }
    raw = _get(url, params=params, timeout=30).json()
    rows = []
    for a in raw.get("articles", []):
        title = a.get("title") or ""
        seen = a.get("seendate") or ""
        try:
            ts = pd.to_datetime(seen, utc=True).tz_convert("Asia/Taipei").tz_localize(None)
        except Exception:
            continue
        rows.append({
            "datetime": ts, "title": title.strip(), "source": a.get("domain", ""),
            "url": a.get("url", ""), "query": query,
        })
    return pd.DataFrame(rows)


def collect_news(stock_code: str, company_name: str | None = None, days: int = 365) -> pd.DataFrame:
    terms = [f"{stock_code} 台股"]
    if company_name:
        terms.append(company_name)
    terms.extend(MARKET_NEWS_QUERIES)
    frames = []
    for q in terms:
        for fn in (google_news_rss, gdelt_news):
            try:
                x = fn(q, days=days)
                if not x.empty:
                    frames.append(x)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["datetime", "title", "source", "url", "query"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["title"].astype(str).str.len() > 4]
    df["key"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates("key").drop(columns="key").sort_values("datetime")
    cutoff = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None) - pd.Timedelta(days=days)
    return df[df["datetime"] >= cutoff].reset_index(drop=True)


def _clean_title(s: str) -> str:
    s = re.sub(r"\s+-\s+[^-]{1,60}$", "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def topic_features(news: pd.DataFrame, trading_index: pd.DatetimeIndex, n_topics: int = 10, cache_key: str = "market") -> tuple[pd.DataFrame, dict]:
    """Fit BERTopic when available; fall back to deterministic TF-IDF clustering.

    Returns daily topic proportions and metadata. The fallback makes the app runnable on
    lightweight hosts, while metadata tells the user which engine actually ran.
    """
    idx = pd.DatetimeIndex(trading_index).normalize()
    cols = [f"news_topic_{i}" for i in range(n_topics)]
    base = pd.DataFrame(0.0, index=idx, columns=cols + ["news_count", "news_entropy", "news_outlier_ratio"])
    if news is None or news.empty or len(news) < 12:
        return base, {"topic_engine": "none", "headline_count": 0, "topic_labels": {}}

    x = news.copy()
    x["text"] = x["title"].map(_clean_title)
    x = x[x["text"].str.len() > 4].reset_index(drop=True)
    texts = x["text"].tolist()
    labels = None
    probs = None
    engine = "BERTopic"
    topic_names = {}

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        model = BERTopic(
            embedding_model=embedder,
            language="multilingual",
            min_topic_size=max(5, min(20, len(texts)//20)),
            calculate_probabilities=True,
            verbose=False,
        )
        labels, probabilities = model.fit_transform(texts)
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities) if probabilities is not None else None
        info = model.get_topic_info()
        ranked = [int(t) for t in info.Topic.tolist() if int(t) >= 0][:n_topics]
        remap = {t: i for i, t in enumerate(ranked)}
        topic_names = {str(remap[t]): str(info.loc[info.Topic == t, "Name"].iloc[0]) for t in ranked}
        mapped = np.array([remap.get(int(t), -1) for t in labels])
    except Exception:
        engine = "TFIDF-MiniBatchKMeans fallback"
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
        k = max(2, min(n_topics, max(2, len(texts)//8)))
        vec = TfidfVectorizer(max_features=2500, ngram_range=(1,2), min_df=2, max_df=.98)
        z = vec.fit_transform(texts)
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto", batch_size=128)
        mapped = km.fit_predict(z)
        d = km.transform(z)
        inv = np.exp(-d / np.maximum(np.nanmedian(d), 1e-6))
        probs = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
        terms = np.asarray(vec.get_feature_names_out())
        for i in range(k):
            top = terms[km.cluster_centers_[i].argsort()[-5:][::-1]]
            topic_names[str(i)] = " · ".join(top.tolist())

    x["topic"] = mapped
    x["date"] = pd.to_datetime(x["datetime"]).dt.normalize()
    for date, g in x.groupby("date"):
        eligible = idx[idx >= date]
        if len(eligible) == 0:
            continue
        # News after a non-trading day is assigned to next trading day; same-day news is
        # assumed known by end-of-day. The model forecasts after close, avoiding future leakage.
        td = eligible[0]
        if td not in base.index:
            continue
        valid = g[g["topic"] >= 0]
        base.loc[td, "news_count"] += len(g)
        base.loc[td, "news_outlier_ratio"] = float((g["topic"] < 0).mean())
        counts = valid["topic"].value_counts()
        total = max(int(counts.sum()), 1)
        pvals = []
        for t, c in counts.items():
            if 0 <= int(t) < n_topics:
                p = float(c) / total
                base.loc[td, f"news_topic_{int(t)}"] = p
                pvals.append(p)
        if pvals:
            base.loc[td, "news_entropy"] = float(-sum(p * math.log(p + 1e-12) for p in pvals))

    # Useful variance/shock signals requested by the user.
    for c in cols:
        base[f"{c}_shock5"] = base[c] - base[c].rolling(5, min_periods=1).mean()
        base[f"{c}_var20"] = base[c].rolling(20, min_periods=2).var().fillna(0)
    base["news_count_z20"] = (base.news_count - base.news_count.rolling(20, min_periods=5).mean()) / base.news_count.rolling(20, min_periods=5).std().replace(0, np.nan)
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0)
    return base, {"topic_engine": engine, "headline_count": len(x), "topic_labels": topic_names}
