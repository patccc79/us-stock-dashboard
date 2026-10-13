from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from fetch_us_stocks import fetch, enrich
from signals import evaluate
from strategy import analyze, build_excel, RULES

st.set_page_config(page_title="美股研究儀表板", page_icon="📈", layout="wide")
st.title("美股研究儀表板")
st.caption("日線研究原型｜規則化參考，不構成投資建議，也不會自動落單")


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
    start = st.date_input("開始日期", today - timedelta(days=1095))
    end_inclusive = st.date_input("結束日期（包含當日）", today)
    chart_mode = st.radio("價格圖", ["未調整 K 線", "調整收市價及均線"])
    if st.button("重新抓取資料"):
        load_prices.clear()
    st.caption("資料快取 15 分鐘。策略需要約 200 個交易日暖機，建議至少選 2 年。")

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

single, comparison, signals, strat, exports = st.tabs(["個股圖表", "同區間比較", "買賣參考訊號", "策略與績效", "資料與下載"])
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
    st.caption("簡易研究評分（滿分 125）：趨勢、動能、風險。不是預測或買賣建議。")
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

with strat:
    st.caption("規則化策略：技術指標 → 入場／出場條件 → 風險回報比 → 歷史回測。結果是研究參考，不是買賣指示。")
    c1, c2, c3, c4, c5 = st.columns(5)
    params = {"stop_mult": c1.number_input("止損倍數（×ATR）", 0.5, 6.0, 2.0, 0.5),
              "target_mult": c2.number_input("目標倍數（×ATR）", 0.5, 10.0, 3.0, 0.5),
              "cost_bps": c3.number_input("單邊成本（bps）", 0.0, 100.0, 10.0, 1.0),
              "account": c4.number_input("帳戶資金（USD）", 1000.0, 100000000.0, 100000.0, 1000.0),
              "risk_pct": c5.number_input("單筆風險（%）", 0.1, 5.0, 1.0, 0.1)}
    results, skipped = {}, []
    for s, d in frames.items():
        if len(d) < 220:
            skipped.append(s)
            continue
        try:
            results[s] = analyze(d, params)
        except Exception as exc:
            skipped.append(f"{s}（{exc}）")
    if skipped:
        st.warning("以下股票資料不足 220 個交易日或計算失敗，已略過策略分析：" + "、".join(skipped))
    if not results:
        st.info("請把開始日期提早，至少涵蓋約 2 年。")
    else:
        st.dataframe(pd.DataFrame([{"股票": s, "規則狀態": r["rec"]["狀態"], "收市價": r["rec"]["收市價"],
                                    "止損價": r["rec"]["止損價"], "目標價": r["rec"]["目標價（ATR）"],
                                    "風險回報比": r["rec"]["風險回報比（ATR目標）"],
                                    "建議最大股數": r["rec"]["建議最大股數"],
                                    "策略總回報": r["metrics"].set_index("指標").loc["總回報", "策略"],
                                    "買入持有總回報": r["metrics"].set_index("指標").loc["總回報", "買入持有"]}
                                   for s, r in results.items()]), use_container_width=True, hide_index=True)
        p2 = st.selectbox("策略明細", list(results), key="strat_pick")
        R = results[p2]; rec = R["rec"]; D = R["data"]
        k = st.columns(4)
        k[0].metric("規則狀態", rec["狀態"])
        k[1].metric("止損價 / 目標價", f"{rec['止損價']:,.2f} / {rec['目標價（ATR）']:,.2f}")
        k[2].metric("風險回報比（ATR）", fmt(rec["風險回報比（ATR目標）"]))
        k[3].metric("風險回報比（60日高位）", fmt(rec["風險回報比（60日高位）"]))
        st.write(f"以帳戶 {params['account']:,.0f} USD、單筆風險 {params['risk_pct']}% 計：最大約 {rec['建議最大股數']:,} 股，"
                 f"倉位市值約 {rec['倉位市值']:,.0f} USD，若觸及止損損失約 {rec['單筆風險金額']:,.0f} USD。")
        st.dataframe(pd.DataFrame(rec["條件"], columns=["入場條件", "目前是否成立"]), hide_index=True)
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.04)
        f.add_trace(go.Scatter(x=D.index, y=D["Adj Close"], name="調整收市價"), row=1, col=1)
        for col, nm in (("SMA_50_Adj", "50 日"), ("SMA_200_Adj", "200 日"), ("BB_Upper", "布林上軌"), ("BB_Lower", "布林下軌")):
            f.add_trace(go.Scatter(x=D.index, y=D[col], name=nm, line=dict(width=1)), row=1, col=1)
        ent, ext = D[D["Event"] == "入場"], D[D["Event"].str.startswith("出場")]
        f.add_trace(go.Scatter(x=ent.index, y=ent["Adj Close"], mode="markers", name="入場訊號",
                               marker=dict(symbol="triangle-up", size=10, color="green")), row=1, col=1)
        f.add_trace(go.Scatter(x=ext.index, y=ext["Adj Close"], mode="markers", name="出場訊號",
                               marker=dict(symbol="triangle-down", size=10, color="red")), row=1, col=1)
        f.add_trace(go.Bar(x=D.index, y=D["MACD_Hist"], name="MACD 柱"), row=2, col=1)
        f.add_trace(go.Scatter(x=D.index, y=D["RSI14"], name="RSI14"), row=3, col=1)
        f.add_hline(y=70, line_dash="dot", row=3, col=1)
        f.add_hline(y=40, line_dash="dot", row=3, col=1)
        f.update_layout(height=760, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(f, use_container_width=True)
        e = go.Figure([go.Scatter(x=D.index, y=D["Strategy_Equity"], name="策略（扣成本）"),
                       go.Scatter(x=D.index, y=D["BuyHold_Equity"], name="買入持有")])
        e.update_layout(title="權益曲線（起點 = 1）", height=380)
        st.plotly_chart(e, use_container_width=True)
        st.subheader("歷史績效評比")
        st.dataframe(R["metrics"], use_container_width=True, hide_index=True)
        st.subheader("交易紀錄")
        st.dataframe(R["trades"], use_container_width=True, hide_index=True)
        with st.expander("策略規則與假設"):
            for a, b in RULES:
                st.write(f"**{a}**：{b}")
        st.download_button("下載 Excel 策略報告", build_excel(results, params), "us_stock_strategy_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with exports:
    combined = pd.concat([d.reset_index().assign(Ticker=s) for s, d in frames.items()], ignore_index=True)
    st.dataframe(combined, use_container_width=True, hide_index=True)
    st.download_button("下載全部日線 CSV", csv_bytes(combined, False), "prices_all.csv", "text/csv")
    st.download_button("下載概覽 CSV", csv_bytes(summary, False), "summary.csv", "text/csv")
    st.caption("數據來源：Yahoo Finance（透過 yfinance）。")
