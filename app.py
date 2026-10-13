from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from fetch_us_stocks import fetch, enrich
from signals import evaluate

st.set_page_config(page_title="美股研究儀表板", page_icon="📈", layout="wide")
st.title("美股研究儀表板")
st.caption("日線研究原型｜訊號只是規則化參考，不提供自動落單或個人化投資建議")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol, start, end):
    return enrich(fetch(symbol, None, start, end, 3)), datetime.now(timezone.utc).isoformat()


def fmt(v, pct=False):
    if pd.isna(v):
        return "資料不足"
    return f"{v:.2%}" if pct else f"{v:,.2f}"


def csv_bytes(df, index=True):
    return df.to_csv(index=index).encode("utf-8-sig")


with st.sidebar:
    st.header("設定")
    raw = st.text_input("美股代號（用逗號分隔，最多 8 隻）", "VOO, NBIS, AAOI")
    symbols = []
    for token in raw.upper().replace("，", ",").replace(" ", ",").split(","):
        token = token.strip()
        if token and token not in symbols:
            if all(c.isalnum() or c in ".-" for c in token) and len(token) <= 10:
                symbols.append(token)
            else:
                st.warning(f"已略過無效代號：{token}")
    if len(symbols) > 8:
        st.warning("最多分析 8 隻，已取前 8 隻。")
        symbols = symbols[:8]
    today = datetime.now(timezone.utc).date()
    start = st.date_input("開始日期", today - timedelta(days=730))
    end_inclusive = st.date_input("結束日期（包含當日）", today)
    chart_mode = st.radio("價格圖", ["未調整 K 線", "調整收市價及均線"])
    if st.button("重新抓取資料"):
        load_prices.clear()
    st.caption("資料快取 15 分鐘。")

if not symbols:
    st.info("請先輸入至少一個美股代號。")
    st.stop()
if start > end_inclusive:
    st.error("開始日期不能晚於結束日期。")
    st.stop()
end = (end_inclusive + timedelta(days=1)).isoformat()
frames, stamps = {}, {}
with st.spinner("正在抓取日線資料…"):
    for s in symbols:
        try:
            frames[s], stamps[s] = load_prices(s, start.isoformat(), end)
        except Exception as exc:
            st.warning(f"{s} 下載失敗：{exc}")
if not frames:
    st.error("所有股票均下載失敗。請檢查代號、網絡或稍後重試。")
    st.stop()

st.warning("最新日線可能尚未收市，並非即時報價。")
rows = []
for s, df in frames.items():
    last = df.iloc[-1]
    rows.append({"股票": s, "首筆日期": str(df.index[0].date()), "最新日期": str(df.index[-1].date()),
                 "交易日數": len(df), "未調整收市價": last["Close"], "調整收市價": last["Adj Close"],
                 "每日回報": last["Return_1D"], "區間最大回撤": df["Drawdown_From_Observed_Peak"].min()})
summary = pd.DataFrame(rows)
st.subheader("市場概覽")
for col, (s, df) in zip(st.columns(len(frames)), frames.items()):
    last = df.iloc[-1]
    col.metric(f"{s} 收市價（USD）", fmt(last["Close"]),
               delta=f"日回報 {fmt(last['Return_1D'], True)}" if pd.notna(last["Return_1D"]) else None)
    col.caption(f"資料日期：{df.index[-1].date()}")
st.dataframe(summary, use_container_width=True, hide_index=True)

single, comparison, signals, exports = st.tabs(["個股圖表", "同區間比較", "買賣參考訊號", "資料與下載"])
with single:
    sym = st.selectbox("分析股票", list(frames))
    df = frames[sym]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])
    if chart_mode == "未調整 K 線":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="未調整價格"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="調整收市價"), row=1, col=1)
        for n in (20, 50, 200):
            fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{n}_Adj"], name=f"{n} 日均線"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量"), row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    dd = go.Figure(go.Scatter(x=df.index, y=df["Drawdown_From_Observed_Peak"] * 100, fill="tozeroy", name="回撤"))
    dd.update_layout(title="相對區間內高點的回撤 (%)", height=280)
    st.plotly_chart(dd, use_container_width=True)

with comparison:
    common = pd.concat({s: d["Adj Close"] for s, d in frames.items()}, axis=1).dropna()
    st.caption("只使用所有股票共同有資料的日期；回報以調整收市價計算，未扣成本或稅項。")
    if len(common) < 2:
        st.info("共同日期不足，無法比較。")
    else:
        rebased = common.div(common.iloc[0]).mul(100)
        fig = go.Figure([go.Scatter(x=rebased.index, y=rebased[s], name=s) for s in rebased.columns])
        fig.update_layout(title="共同起點 = 100", height=430)
        st.plotly_chart(fig, use_container_width=True)
        daily = common.pct_change(fill_method=None).dropna()
        st.dataframe(pd.DataFrame([{"股票": s, "共同區間回報": common[s].iloc[-1] / common[s].iloc[0] - 1,
                                    "年化波動率": daily[s].std(ddof=1) * np.sqrt(252),
                                    "最大回撤": (common[s] / common[s].cummax() - 1).min()}
                                   for s in common.columns]), use_container_width=True, hide_index=True)
        if len(common.columns) > 1:
            c = daily.corr()
            h = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1,
                                     colorscale="RdBu", reversescale=True))
            h.update_layout(title="回報相關性", height=350)
            st.plotly_chart(h, use_container_width=True)

with signals:
    st.caption("規則化研究評分（滿分 125）：趨勢、動能、風險。這不是預測或買賣建議；未計入基本面、估值、新聞、財報日、交易成本。")
    res = {s: evaluate(d) for s, d in frames.items()}
    st.dataframe(pd.DataFrame([{"股票": s, "評分": r["total"], "狀態": r["label"], "RSI14": r["rsi"],
                                "ATR 佔股價": r["atr_pct"], "2×ATR 參考風險線": r["stop"]}
                               for s, r in res.items()]), use_container_width=True, hide_index=True)
    pick = st.selectbox("查看評分明細", list(res), key="sig_pick")
    r = res[pick]
    st.metric(f"{pick} 評分", f"{r['total']} / 125", r["label"])
    st.write(r["parts"])
    for n in r["notes"]:
        st.write("- " + n)
    st.subheader("我的持倉（不會儲存）")
    c1, c2 = st.columns(2)
    shares = c1.number_input("持有股數", min_value=0.0, value=0.0, step=1.0)
    cost = c2.number_input("平均成本（USD）", min_value=0.0, value=0.0, step=0.01)
    if shares > 0 and cost > 0:
        price = float(frames[pick]["Close"].iloc[-1])
        st.metric("未實現盈虧（未計費用、稅項）", f"{(price - cost) * shares:,.2f} USD", f"{price / cost - 1:.2%}")
        if pd.notna(r["stop"]):
            st.write(f"以 2×ATR 為風險線約 {r['stop']:,.2f} USD；跌至該價位的損益約 {(r['stop'] - cost) * shares:,.2f} USD。")
    st.info("偏強不代表應買入，偏弱也不代表應沽出；請結合基本面、倉位和自己的風險承受能力。")

with exports:
    combined = pd.concat([d.reset_index().assign(Ticker=s) for s, d in frames.items()], ignore_index=True)
    st.dataframe(combined, use_container_width=True, hide_index=True)
    st.download_button("下載全部日線 CSV", csv_bytes(combined, False), "prices_all.csv", "text/csv")
    st.download_button("下載概覽 CSV", csv_bytes(summary, False), "summary.csv", "text/csv")
    st.caption("數據來源：Yahoo Finance（透過 yfinance）。")
