# -*- coding: utf-8 -*-
# Streamlit App：Goodinfo 年增率 + 財務比率 + 雷達圖 + K 線（讀 GitHub Raw CSV）
# ---------------------------------------------------------------

import re, time, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="Goodinfo 財務儀表板", layout="wide")

# === GitHub Raw CSV 路徑 ===
CSV_YOY = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E5%B9%B4%E5%A2%9E%E7%8E%87_%E6%AD%B7%E5%B9%B4%E6%AF%94%E8%BC%83_%E5%90%AB%E6%96%B0%E7%94%A2%E6%A5%AD%E5%88%86%E9%A1%9Etest1.csv"
CSV_GM  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%87%9F%E6%A5%AD%E6%AF%9B%E5%88%A9%E7%8E%87test2.csv"
CSV_OM  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%87%9F%E6%A5%AD%E5%88%A9%E7%9B%8A%E7%8E%87test2.csv"
CSV_CF  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%8F%BE%E9%87%91%E6%B5%81%E9%87%8F%E2%80%93%E7%87%9F%E6%A5%AD%E6%B4%BB%E5%8B%95%E7%8F%BE%E9%87%91%E6%B5%81%E9%87%8Ftest2.csv"

# === 1) 年增率資料載入 ===
@st.cache_data
def load_yoy_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.map(lambda x: str(x).replace("\xa0"," ").replace("\u3000"," ").strip())
    if "新產業分類" in df.columns:
        df.rename(columns={"新產業分類": "產業分類"}, inplace=True)
    yoy_cols = [c for c in df.columns if "年增率" in c and not c.startswith("平均")]
    df[yoy_cols] = df[yoy_cols].apply(pd.to_numeric, errors="coerce")
    df_m = df.melt(id_vars=["代號","名稱","產業分類"], value_vars=yoy_cols,
                   var_name="期間", value_name="年增率")
    def parse_month_to_date(month_str):
        m = re.search(r"(\d{2})M(\d{2})", str(month_str))
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=2000 + y, month=mo, day=1)
        return pd.NaT
    df_m["日期"] = df_m["期間"].apply(parse_month_to_date)
    return df_m

# === 2) 財務比率資料載入 ===
@st.cache_data
def load_financial_ratios(csv_gm, csv_om, csv_cf):
    gm = pd.read_csv(csv_gm, encoding="utf-8-sig")
    om = pd.read_csv(csv_om, encoding="utf-8-sig")
    cf = pd.read_csv(csv_cf, encoding="utf-8-sig")

    def melt_df(df, value_name):
        qcols = [c for c in df.columns if "Q" in c]
        if not qcols:
            return pd.DataFrame()
        d = df.melt(id_vars=["代號","名稱"], value_vars=qcols, var_name="期間", value_name=value_name)
        d["季度"] = d["期間"].str.extract(r"(\d{2}Q\d)")[0]
        d["日期"] = pd.PeriodIndex(d["季度"], freq="Q").to_timestamp("Q")
        return d

    gm_m = melt_df(gm, "毛利率")
    om_m = melt_df(om, "營益率")
    cf_m = melt_df(cf, "營業金流")

    df_fin = gm_m.merge(om_m, on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.merge(cf_m, on=["代號","名稱","日期"], how="outer")
    return df_fin.sort_values(["代號","日期"]).reset_index(drop=True)

# === 3) Yahoo Finance 日K線 ===
@st.cache_data
def fetch_history(symbol: str):
    start = int(datetime(2019,1,1).timestamp())
    end = int(time.time())
    for suffix in [".TW", ".TWO"]:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}"
        params = {"period1": start, "period2": end, "interval": "1d"}
        try:
            data = requests.get(url, params=params, timeout=10).json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame({
                "Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]
            }, index=pd.to_datetime(ts, unit="s"))
            df["MA5"] = df["Close"].rolling(5).mean()
            df["MA20"] = df["Close"].rolling(20).mean()
            return df
        except:
            continue
    return pd.DataFrame()

# === 載入資料 ===
df_yoy = load_yoy_data(CSV_YOY)
df_fin = load_financial_ratios(CSV_GM, CSV_OM, CSV_CF)

# === 側邊選單 ===
st.sidebar.title("📊 Goodinfo 財務儀表板")
inds = sorted(df_yoy["產業分類"].dropna().unique())
sel_inds = st.sidebar.multiselect("選擇產業分類", inds)
manual_codes = st.sidebar.text_input("或輸入股票代號（逗號分隔）", "2330,1101").split(",")

filtered = df_yoy.copy()
if sel_inds: filtered = filtered[filtered["產業分類"].isin(sel_inds)]
if manual_codes: filtered = pd.concat([filtered, df_yoy[df_yoy["代號"].isin([c.strip() for c in manual_codes])]])

stocks = filtered[["代號","名稱"]].drop_duplicates()
opts = {f"{r['代號']} {r['名稱']}": r["代號"] for _,r in stocks.iterrows()}
selected = st.sidebar.selectbox("選擇股票", list(opts.keys()))

# === 顯示圖表 ===
if selected:
    code = opts[selected]
    yoy_s = df_yoy[df_yoy["代號"] == code].sort_values("日期")
    fin_s = df_fin[df_fin["代號"] == code].sort_values("日期")
    df_yf = fetch_history(code)

    st.markdown(f"## 📈 {code} 走勢與財務分析")

    # --- 年增率 + K 線 ---
    if not df_yf.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8,0.2])
        fig.add_trace(go.Candlestick(x=df_yf.index, open=df_yf["Open"], high=df_yf["High"],
                                     low=df_yf["Low"], close=df_yf["Close"], name="K 線",
                                     increasing_line_color="red", decreasing_line_color="green"), row=1, col=1)
        for w in [5,20]:
            fig.add_trace(go.Scatter(x=df_yf.index, y=df_yf[f"MA{w}"], name=f"MA{w}"), row=1,col=1)
        if not yoy_s.empty:
            fig.add_trace(go.Scatter(x=yoy_s["日期"], y=yoy_s["年增率"], name="年增率", yaxis="y2", line=dict(dash="dot")), row=1,col=1)
        fig.add_trace(go.Bar(x=df_yf.index, y=df_yf["Volume"], name="成交量"), row=2,col=1)
        fig.update_layout(height=700, hovermode="x unified", title=f"{code} K 線 + 月營收年增率")
        st.plotly_chart(fig, use_container_width=True)

    # --- 財務比率趨勢 ---
    if not fin_s.empty:
        st.subheader("📊 財務比率趨勢")
        fig2 = px.line(fin_s, x="日期", y=["毛利率","營益率"], markers=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("💰 營業活動現金流量")
        fig3 = px.bar(fin_s, x="日期", y="營業金流")
        st.plotly_chart(fig3, use_container_width=True)

st.caption("資料來源：Goodinfo.tw | 製作：rogerting17")
