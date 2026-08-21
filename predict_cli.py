import argparse
from src.predictor import train_and_predict

p=argparse.ArgumentParser(); p.add_argument('code'); p.add_argument('--epochs',type=int,default=35); p.add_argument('--news-days',type=int,default=365); p.add_argument('--no-news',action='store_true')
a=p.parse_args(); r=train_and_predict(a.code,epochs=a.epochs,news_days=a.news_days,use_news=not a.no_news)
print(r.symbol,r.source,'latest',r.latest_close)
for i,(ret,price,lo,hi) in enumerate(zip(r.predicted_returns,r.predicted_prices,r.lower_prices,r.upper_prices),1):
    print(f'Day +{i}: NT${price:.2f}  ({(pow(2.718281828,ret)-1):+.2%})  interval {lo:.2f}–{hi:.2f}')
print(r.metrics)
