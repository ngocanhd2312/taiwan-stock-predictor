from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS, FORECAST_HORIZON, LOOKBACK_DAYS, RANDOM_SEED
from .data import market_context, stock_history
from .features import make_features, make_targets
from .model import BiLSTMAttention
from .news import collect_news, topic_features


def seed_all(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@dataclass
class PredictionResult:
    symbol: str
    source: str
    latest_close: float
    predicted_returns: list[float]
    predicted_prices: list[float]
    lower_prices: list[float]
    upper_prices: list[float]
    history: pd.DataFrame
    metrics: dict
    topic_meta: dict
    recent_news: pd.DataFrame


def _windows(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, lookback=LOOKBACK_DAYS):
    xs=[]; ys=[]; ds=[]
    for i in range(lookback-1, len(X)):
        if np.isfinite(y[i]).all() and np.isfinite(X[i-lookback+1:i+1]).all():
            xs.append(X[i-lookback+1:i+1]); ys.append(y[i]); ds.append(dates[i])
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32), pd.DatetimeIndex(ds)


def _metrics(y, p, close_map, dates):
    e = p-y
    out = {
        "validation_samples": int(len(y)),
        "validation_mae_logreturn": float(np.mean(np.abs(e))),
        "validation_rmse_logreturn": float(np.sqrt(np.mean(e**2))),
        "validation_direction_accuracy_day1": float(np.mean(np.sign(p[:,0])==np.sign(y[:,0]))),
        "random_walk_mae_logreturn": float(np.mean(np.abs(y))),
    }
    for h in range(y.shape[1]):
        out[f"mae_h{h+1}"] = float(np.mean(np.abs(e[:,h])))
        out[f"direction_h{h+1}"] = float(np.mean(np.sign(p[:,h])==np.sign(y[:,h])))
    if dates is not None and len(dates):
        bases=np.array([close_map.get(pd.Timestamp(d), np.nan) for d in dates],float)
        valid=np.isfinite(bases)
        if valid.any():
            pp=bases[valid,None]*np.exp(p[valid]); yy=bases[valid,None]*np.exp(y[valid])
            out["validation_price_mae_ntd"] = float(np.mean(np.abs(pp-yy)))
    return out


def train_model_from_frames(stock: pd.DataFrame, market: pd.DataFrame, news_daily: pd.DataFrame, epochs=35, lookback=LOOKBACK_DAYS):
    seed_all()
    feats=make_features(stock,market,news_daily)
    close=stock["Close"].reindex(feats.index)
    targets=make_targets(close,FORECAST_HORIZON)
    # Keep rows with stable feature history; imputation values are learned from train only below.
    usable=feats.index[feats.notna().sum(axis=1) >= max(5, int(feats.shape[1]*.55))]
    feats=feats.loc[usable]; targets=targets.loc[usable]; close=close.loc[usable]
    if len(feats)<320:
        raise RuntimeError(f"Not enough usable daily history ({len(feats)} rows). Need at least ~320.")

    n=len(feats); train_end=int(n*.70); val_end=int(n*.85)
    # Purge the 5 rows before boundaries to avoid target horizon crossing splits.
    train_idx=np.arange(0,max(1,train_end-FORECAST_HORIZON))
    train_frame=feats.iloc[train_idx]
    med=train_frame.median(numeric_only=True).reindex(feats.columns).fillna(0)
    feats=feats.fillna(med).fillna(0).clip(-1e6,1e6)
    scaler=RobustScaler(quantile_range=(10,90)).fit(feats.iloc[train_idx])
    z=scaler.transform(feats)
    y=targets.to_numpy(float)
    Xw,yw,dw=_windows(z,y,feats.index,lookback)
    # Map window end date back to original positional split.
    trmask=dw < feats.index[train_end-FORECAST_HORIZON]
    vamask=(dw >= feats.index[train_end]) & (dw < feats.index[val_end-FORECAST_HORIZON])
    temask=dw >= feats.index[val_end]
    if trmask.sum()<100 or vamask.sum()<30 or temask.sum()<30:
        raise RuntimeError("Chronological split is too small after feature construction.")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BiLSTMAttention(z.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    lossfn=nn.HuberLoss(delta=.02)
    loader=DataLoader(TensorDataset(torch.tensor(Xw[trmask]),torch.tensor(yw[trmask])),batch_size=64,shuffle=True)
    xv=torch.tensor(Xw[vamask],device=device); yv=torch.tensor(yw[vamask],device=device)
    best=None; best_loss=float("inf"); wait=0
    for ep in range(int(epochs)):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True)
            pred=model(xb); loss=lossfn(pred,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(lossfn(model(xv),yv).item())
        if vl < best_loss-1e-6:
            best_loss=vl; best=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=7: break
    if best is not None: model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pv=model(torch.tensor(Xw[vamask],device=device)).cpu().numpy()
        pt=model(torch.tensor(Xw[temask],device=device)).cpu().numpy()
    close_map=close.to_dict()
    metrics=_metrics(yw[vamask],pv,close_map,dw[vamask])
    tm=_metrics(yw[temask],pt,close_map,dw[temask])
    metrics.update({f"test_{k}":v for k,v in tm.items()})
    # Ridge baseline on last observation of each window.
    ridge=Ridge(alpha=10).fit(Xw[trmask][:,-1,:],yw[trmask])
    rp=ridge.predict(Xw[vamask][:,-1,:])
    metrics["ridge_validation_mae_logreturn"]=float(np.mean(np.abs(rp-yw[vamask])))
    metrics["beats_random_walk_validation"] = bool(metrics["validation_mae_logreturn"] < metrics["random_walk_mae_logreturn"])

    val_resid=yw[vamask]-pv
    qlo=np.quantile(val_resid,.10,axis=0); qhi=np.quantile(val_resid,.90,axis=0)
    last_seq=torch.tensor(z[-lookback:][None].astype(np.float32),device=device)
    with torch.no_grad(): forecast=model(last_seq).cpu().numpy()[0]
    return model,scaler,med,feats.columns.tolist(),forecast,qlo,qhi,metrics


def train_and_predict(stock_code: str, epochs=35, news_days=365, company_name: str | None=None, use_news=True) -> PredictionResult:
    symbol,source,stock=stock_history(stock_code,3)
    start=stock.index.min().date()-pd.Timedelta(days=10)
    end=stock.index.max().date()+pd.Timedelta(days=1)
    market=market_context(pd.Timestamp(start).date(),pd.Timestamp(end).date())
    raw_news=collect_news(symbol.split('.')[0],company_name=company_name,days=min(int(news_days),365)) if use_news else pd.DataFrame()
    news_daily,topic_meta=topic_features(raw_news,stock.index,n_topics=10,cache_key=symbol)
    model,scaler,med,columns,fr,qlo,qhi,metrics=train_model_from_frames(stock,market,news_daily,epochs=epochs)
    last=float(stock.Close.dropna().iloc[-1])
    prices=(last*np.exp(fr)).tolist(); lo=(last*np.exp(fr+qlo)).tolist(); hi=(last*np.exp(fr+qhi)).tolist()
    # Save artifacts for repeatable inference / audit.
    out=ARTIFACTS/symbol.replace('.','_'); out.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"n_features":len(columns),"columns":columns},out/"model.pt")
    joblib.dump({"scaler":scaler,"median":med},out/"preprocessor.joblib")
    with open(out/"metadata.json","w",encoding="utf-8") as f:
        json.dump({"symbol":symbol,"source":source,"latest_date":str(stock.index[-1].date()),"metrics":metrics,"topic_meta":topic_meta},f,ensure_ascii=False,indent=2)
    return PredictionResult(symbol,source,last,fr.tolist(),prices,lo,hi,stock,metrics,topic_meta,raw_news.tail(30))
