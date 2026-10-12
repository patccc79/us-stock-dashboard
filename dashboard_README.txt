美股 Streamlit 儀表板 — Python 3.10+

1. 解壓 ZIP，於該資料夾開啟終端機。
2. 建議建立虛擬環境：python -m venv .venv
   Windows：.venv\Scripts\activate
   macOS/Linux：source .venv/bin/activate
3. python -m pip install -r requirements.txt
4. python -m streamlit run app.py
5. 開啟終端機顯示的本機網址，通常為 http://localhost:8501。

app.py 必須與 fetch_us_stocks.py 放在同一資料夾。
需要網絡連線。无需 API key，不連接券商、GitHub 或 Supabase。
本機應用，不代表已部署至公網。

功能：日期篩選、股票選擇、15 分鐘快取、手動刷新、原始 K 線、
調整價均線、成交量、回撤、共同日期比較、相關性及 CSV 下載。
結束日期在畫面上包含當日；底層 API 使用翌日為不包含截止日。
價格指標均線使用調整收市價；K 線使用未調整 OHLC。
均線不含選定開始日前暖機資料，故首 19/49/199 筆均線留空。
波動率是樣本標準差乘 sqrt(252)，非預測；最大回撤僅在選定區間。
比較不補缺失值，僅採用共同日期；缺漏交易日可能影響波動率估計。
下載失敗會顯示錯誤並保留其他成功股票，不加入假數據。
最新日線可能未完成；數據可比性、延遲及完整性須另外驗證。

驗證狀態：語法檢查完成；目前環境沒有實際啟動 Streamlit 或連線抓取測試。
來源：
https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
https://docs.streamlit.io/develop/api-reference/charts
https://docs.streamlit.io/1.30.0/develop/api-reference/caching-and-state/st.cache_data
