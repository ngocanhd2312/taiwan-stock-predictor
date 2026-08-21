"""Offline smoke test: proves feature->training->5-output inference works without network."""
import numpy as np, pandas as pd
from src.features import make_features
from src.predictor import train_model_from_frames

rng=np.random.default_rng(42); n=780; idx=pd.bdate_range('2023-01-02',periods=n)
r=rng.normal(.0005,.015,n); close=500*np.exp(np.cumsum(r)); open_=close*np.exp(rng.normal(0,.004,n)); high=np.maximum(open_,close)*(1+abs(rng.normal(0,.006,n))); low=np.minimum(open_,close)*(1-abs(rng.normal(0,.006,n)))
stock=pd.DataFrame({'Open':open_,'High':high,'Low':low,'Close':close,'Adj Close':close,'Volume':rng.lognormal(16,0.5,n)},index=idx)
market=pd.DataFrame({'taiex':18000*np.exp(np.cumsum(rng.normal(.0003,.01,n))),'vix':20*np.exp(np.cumsum(rng.normal(0,.02,n)))},index=idx)
news=pd.DataFrame({'news_topic_0':rng.beta(1,8,n),'news_count':rng.poisson(3,n),'news_entropy':rng.random(n),'news_outlier_ratio':rng.random(n)*.2},index=idx)
m,sc,med,cols,fr,lo,hi,metrics=train_model_from_frames(stock,market,news,epochs=2)
assert len(fr)==5 and np.isfinite(fr).all() and metrics['validation_samples']>0
print('PASS',len(cols),'features',fr.tolist())
