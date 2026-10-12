import io
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from fetch_us_stocks import fetch, enrich

st.set_page_config(page_title="美股研究儀表板", page_icon="📈", layout="wide")
st.title("美股研究儀表板")
st.caption("VOO · NBIS · AAOI｜日線研究原型，不提供自動落單或投資建議")

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol, start, end):
    df = enrich(fetch(symbol, None, start, end, 3))
    return df, datetime.now(timezone.utc).isoformat()


def fmt(value, pct=False):
    if pd.isna(value):
        return "資料不足"
    return f"{value:.2%}" if pct else f"{value:,.2f}"


def csv_bytes(df, index=True):
    return df.to_csv(index=index).encode("utf-8-sig")


with st.sidebar:
    st.header("設定")
    symbols = st.multiselect("股票", ["VOO", "NBIS", "AAOI"], default=["VOO", "NBIS", "AAOI"])
    today = datetime.now(timezone.utc).date()
    start = st.date_input("開始日期", today - timedelta(days=730))
    end_inclusive = st.date_input("結束日期（包含當日）", today)
    chart_mode = st.radio("價格圖", ["未調整 K 線", "調整收市價及均線"])
    if st.button("重新抓取資料"):
        load_prices.clear()
    st.caption("資料快取 15 分鐘；切換圖表不需重複下載。")

if not symbols:
    st.info("請先選擇至少一隻股票。")
    st.stop()
if start > end_inclusive:
    st.error("開始日期不能晚於結束日期。")
    st.stop()
end = (end_inclusive + timedelta(days=1)).isoformat()
frames, stamps = {}, {}
with st.spinner("正在抓取日線資料…"):
    for symbol in symbols:
        try:
            frames[symbol], stamps[symbol] = load_prices(symbol, start.isoformat(), end)
        except Exception as exc:
            st.warning(f"{symbol} 下載失敗：{exc}")
if not frames:
    st.error("所有股票均下載失敗。請檢查網絡、日期或稍後重試。")
    st.stop()

st.warning("最新日線可能尚未收市，並非保證即時報價。歷史代號／公司業務變動前後的可比性尚未核實。")
rows=[]
for symbol, df in frames.items():
    last=df.iloc[-1]
    rows.append({"股票":symbol, "首筆日期":str(df.index[0].date()),
                 "最新日期":str(df.index[-1].date()), "交易日數":len(df),
                 "未調整收市價":last["Close"], "調整收市價":last["Adj Close"],
                 "每日回報":last["Return_1D"], "區間最大回撤":df["Drawdown_From_Observed_Peak"].min()})
summary=pd.DataFrame(rows)
st.subheader("市場概覽")
cols=st.columns(len(frames))
for col, (symbol, df) in zip(cols, frames.items()):
    last=df.iloc[-1]
    col.metric(f"{symbol} 收市價（USD）", fmt(last["Close"]),
               delta=f"調整價日回報 {fmt(last['Return_1D'], True)}" if pd.notna(last["Return_1D"]) else None)
    col.caption(f"資料日期：{df.index[-1].date()}")
st.dataframe(summary, use_container_width=True, hide_index=True)

single, comparison, exports = st.tabs(["個股圖表", "同區間比較", "資料與下載"])
with single:
    symbol=st.selectbox("分析股票", list(frames))
    df=frames[symbol]
    fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                      row_heights=[0.75,0.25])
    if chart_mode == "未調整 K 線":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                      low=df["Low"], close=df["Close"], name="未調整價格"), row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index,y=df["Adj Close"],name="調整收市價"),row=1,col=1)
        for n in (20,50,200):
            fig.add_trace(go.Scatter(x=df.index,y=df[f"SMA_{n}_Adj"],name=f"{n} 日均線"),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="成交量"),row=2,col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    fig.update_yaxes(title_text="USD",row=1,col=1)
    fig.update_yaxes(title_text="股數",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("均線只疊加在調整價圖上，避免將未調整 K 線與調整均線混用。均線僅用選定區間資料計算，未滿所需交易日數時留空。")
    dd=go.Figure(go.Scatter(x=df.index,y=df["Drawdown_From_Observed_Peak"] * 100,
                            fill="tozeroy",name="回撤"))
    dd.update_layout(title="相對選定區間內歷史高點的回撤",yaxis_title="%",height=280)
    st.plotly_chart(dd,use_container_width=True)

with comparison:
    common=pd.concat({s:d["Adj Close"] for s,d in frames.items()},axis=1).dropna()
    st.caption("僅使用所有成功下載股票共同有資料的日期；不前向填補。回報基於調整收市價，不扣交易成本或稅項。")
    if len(common)<2:
        st.info("共同日期不足兩個交易日，無法比較。")
    else:
        rebased=common.div(common.iloc[0]).mul(100)
        fig=go.Figure()
        for s in common.columns:
            fig.add_trace(go.Scatter(x=rebased.index,y=rebased[s],name=s))
        fig.update_layout(title="共同起點 = 100",height=430,yaxis_title="指數")
        st.plotly_chart(fig,use_container_width=True)
        daily=common.pct_change(fill_method=None).dropna()
        comparison_rows=[]
        for s in common.columns:
            comparison_rows.append({"股票":s, "共同區間回報":common[s].iloc[-1]/common[s].iloc[0]-1,
              "年化波動率（252交易日）":daily[s].std(ddof=1)*np.sqrt(252) if len(daily)>1 else np.nan,
              "共同區間最大回撤":(common[s]/common[s].cummax()-1).min()})
        st.write(f"共同區間：{common.index[0].date()} 至 {common.index[-1].date()}，{len(common)} 個觀測日")
        st.dataframe(pd.DataFrame(comparison_rows),use_container_width=True,hide_index=True)
        st.caption("表中 0.01 = 1%。波動率以共同日期相鄰觀測回報及 252 日假設估算；若供應商缺漏交易日，估算可能受影響。")
        if len(daily)>1 and len(common.columns)>1:
            corr=daily.corr()
            heat=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
                                     zmin=-1,zmax=1,colorscale="RdBu",reversescale=True))
            heat.update_layout(title="共同區間回報相關性",height=350)
            st.plotly_chart(heat,use_container_width=True)
        st.download_button("下載共同區間指數 CSV",csv_bytes(rebased),"rebased_comparison.csv","text/csv")

with exports:
    combined=pd.concat([d.reset_index().assign(Ticker=s) for s,d in frames.items()],ignore_index=True)
    st.dataframe(combined,use_container_width=True,hide_index=True)
    st.download_button("下載全部日線 CSV",csv_bytes(combined,False),"prices_all.csv","text/csv")
    st.download_button("下載概覽 CSV",csv_bytes(summary,False),"summary.csv","text/csv")
    for s,d in frames.items():
        st.download_button(f"下載 {s} CSV",csv_bytes(d),f"{s}.csv","text/csv",key=f"export_{s}")
    st.write("成功抓取時間（UTC）",stamps)
    st.caption("數據來源：Yahoo Finance（透過 yfinance）。這個版本不包含財報、新聞、回測或買賣訊號。")
