import time
import pandas as pd
import yfinance as yf


def fetch(symbol, period, start, end, retries=3):
    error = None
    for attempt in range(retries):
        try:
            options = dict(interval="1d", auto_adjust=False, progress=False,
                           threads=False, timeout=30, multi_level_index=False)
            if start:
                options.update(start=start, end=end)
            else:
                options.update(period=period or "2y", end=end)
            df = yf.download(symbol, **options)
            if df is None or df.empty:
                raise ValueError("無數據：請檢查代號、日期、網絡或供應商限流")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if not set(required).issubset(df.columns):
                raise ValueError("供應商回傳欄位不完整")
            df = df[required].copy().sort_index()
            df = df.loc[~df.index.duplicated(keep="last")]
            for col in required:
                df[col] = pd.to_numeric(df[col], errors="raise")
            df = df.dropna(subset=["Close", "Adj Close"])
            if df.empty or (df[["Close", "Adj Close"]] <= 0).any().any():
                raise ValueError("無有效收市價")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            return df
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"{symbol}: {error}")


def enrich(df):
    df = df.copy()
    price = df["Adj Close"]
    df["Return_1D"] = price.pct_change(fill_method=None)
    for n in (20, 50, 200):
        df[f"SMA_{n}_Adj"] = price.rolling(n, min_periods=n).mean()
    df["Drawdown_From_Observed_Peak"] = price / price.cummax() - 1
    return df
