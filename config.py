from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / ".cache"
ARTIFACTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

LOOKBACK_DAYS = 60
FORECAST_HORIZON = 5
HISTORY_YEARS = 3
RANDOM_SEED = 42

MARKET_SYMBOLS = {
    "taiex": "^TWII",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "sox": "^SOX",
    "vix": "^VIX",
    "usdtwd": "TWD=X",
    "us10y": "^TNX",
    "wti": "CL=F",
    "gold": "GC=F",
}

# Broad topics that can materially affect Taiwan equities.
MARKET_NEWS_QUERIES = [
    '台股 OR 加權指數 OR TAIEX',
    '台灣 半導體 OR 晶片 OR AI',
    'Federal Reserve OR Fed interest rates inflation stocks',
    'US China Taiwan trade tariffs semiconductor',
    'Taiwan dollar USD TWD exchange rate',
    'oil prices OR crude oil market',
    'geopolitics Taiwan China market',
]
