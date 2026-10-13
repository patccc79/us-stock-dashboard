import numpy as np
import pandas as pd


def rsi14(price):
    d = price.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr14(df):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def evaluate(df):
    p = df["Adj Close"]
    last = p.iloc[-1]
    s50, s200 = df["SMA_50_Adj"].iloc[-1], df["SMA_200_Adj"].iloc[-1]
    rsi = rsi14(p).iloc[-1]
    ret63 = p.iloc[-1] / p.iloc[-64] - 1 if len(p) > 63 else np.nan
    atr = atr14(df).iloc[-1]
    atr_pct = atr / df["Close"].iloc[-1]
    dd = df["Drawdown_From_Observed_Peak"].iloc[-1]
    notes, trend, mom, risk = [], 0, 0, 0
    if pd.notna(s200):
        trend += 25 if last > s200 else 0
        notes.append("價格在 200 日均線之上" if last > s200 else "價格在 200 日均線之下")
    else:
        notes.append("不足 200 個交易日，長期趨勢無法評分")
    if pd.notna(s50):
        trend += 15 if last > s50 else 0
        notes.append("價格在 50 日均線之上" if last > s50 else "價格在 50 日均線之下")
    if pd.notna(s50) and pd.notna(s200) and s50 > s200:
        trend += 10
    if pd.notna(ret63):
        mom += 25 if ret63 > 0 else 0
        notes.append(f"近 63 個交易日回報 {ret63:.1%}")
    if pd.notna(rsi):
        mom += 25 if 40 <= rsi <= 70 else (10 if rsi < 40 else 0)
        notes.append(f"RSI14 為 {rsi:.0f}" + ("（偏熱）" if rsi > 70 else "（偏弱）" if rsi < 30 else ""))
    if pd.notna(atr_pct) and atr_pct < 0.04:
        risk += 12
    if dd > -0.20:
        risk += 13
    total = trend + mom + risk
    label = "偏強" if total >= 75 else "中性" if total >= 45 else "偏弱"
    stop = df["Close"].iloc[-1] - 2 * atr if pd.notna(atr) else np.nan
    return dict(total=int(total), label=label, parts={"趨勢 (0-50)": trend, "動能 (0-50)": mom, "風險 (0-25)": risk},
                notes=notes, rsi=rsi, atr_pct=atr_pct, stop=stop)
