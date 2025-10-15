# -*- coding: utf-8 -*-
# Streamlit App：Goodinfo 年增率 + 財務比率 + 雷達圖 + K 線 (Render 版 / 線上CSV版)
# ---------------------------------------------------------------

import re, time, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

st.set_page_config(page_title="Goodinfo 財務儀表板", layout="wide")

# === 線上 CSV 檔路徑 (請改成你自己的 GitHub Raw URL) ===
CSV_YOY = "https://raw.githubusercontent.com/<你的帳號>/<repo名>/main/Goodinfo_年增率_歷年比較_含新產業分類test1.csv"
CSV_GM  = "https://raw.githubusercontent.com/<你的帳號>/<repo名>/main/Goodinfo_營業毛利率test2.csv"
CSV_OM  = "https://raw.githubusercontent.com/<你的帳號>/<repo名>/main/Goodinfo_營業利益率test2.csv"
CSV_CF  = "https://raw.githubusercontent.com/<你的帳號>/<repo名>/main/Goodinfo_現金流量–營業活動現金流量test2.csv"

@st.cache_data
def load_csv(url):
    df = pd.read_csv(url, encoding="utf-8-sig")
    return df

def load_yoy_data(csv_path: str) -> pd.DataFrame:
    df = load_csv(csv_path)
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

def load_financial_ratios(csv_gm, csv_om, csv_cf):
    gm, om, cf = load_csv(csv_gm), load_csv(csv_om), load_csv(csv_cf)
    def melt_df(df, value_name):
        qcols = [c for c in df.columns if "Q" in c]
        d = df.melt(id_vars=["代號","名稱"], value_vars=qcols, var_name="期間", value_name=value_name)
        d["季度"] = d["期間"].str.extract(r"(\d{2}Q\d)")[0]
        d["日期"] = pd.PeriodIndex(d["季度"], freq="Q").to_timestamp("Q")
        return d
    gm, om, cf = melt_df(gm, "毛利率"), melt_df(om, "營益率"), melt_df(cf, "營業金流")
    df_fin = gm.merge(om, on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.merge(cf, on=["代號","名稱","日期"], how="outer")
    return df_fin.sort_values(["代號","日期"]).reset_index(drop=True)

def fetch_history(symbol):
    start_dt = int(datetime(2019,1,1).timestamp())
    end_dt = int(time.time())
    headers = {"User-Agent":"Mozilla/5.0"}
    for suffix in (".TW",".TWO"):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}"
            params = {"period1":start_dt,"period2":end_dt,"interval":"1d"}
            data = requests.get(url, params=params, headers=headers, timeout=20).json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            df = pd.DataFrame({
                "Open": q["open"], "High": q["high"], "Low": q["low"],
                "Close": q["close"], "Volume": q["volume"]
            }, index=pd.to_datetime(ts, unit="s"))
            for w in (5,10,20,60):
                df[f"MA{w}"] = df["Close"].rolling(w).mean()
            return df.dropna()
        except:
            continue
    return pd.DataFrame()

st.sidebar.title("📂 查詢條件")

df_yoy = load_yoy_data(CSV_YOY)
df_fin = load_financial_ratios(CSV_GM, CSV_OM, CSV_CF)

inds = sorted(df_yoy['產業分類'].dropna().unique())
sel_inds = st.sidebar.multiselect("選擇產業分類", inds)
manual_input = st.sidebar.text_input("或輸入股票代號（逗號分隔）", "2330,1101")
manual_codes = [c.strip() for c in manual_input.split(',') if c.strip()]

filtered = df_yoy.copy()
if sel_inds:
    filtered = filtered[filtered['產業分類'].isin(sel_inds)]
if manual_codes:
    filtered = pd.concat([filtered, df_yoy[df_yoy['代號'].isin(manual_codes)]])

stocks = filtered[['代號','名稱']].drop_duplicates()
opts = {f"{r['代號']} {r['名稱']}": r['代號'] for _,r in stocks.iterrows()}
selected = st.sidebar.multiselect("選擇股票", list(opts.keys()), default=list(opts.keys())[:1])

st.markdown("## 📈 年增率 + 財務比率儀表板")

def normalize(values, cols):
    out=[]
    for v,c in zip(values,cols):
        c=c.dropna()
        if c.empty: out.append(0); continue
        if c.max()==c.min(): out.append(50); continue
        out.append((v-c.min())/(c.max()-c.min())*100)
    return out

if len(selected)==1:
    code = opts[selected[0]]
    yoy_s = df_yoy[df_yoy["代號"]==code].sort_values("日期")
    fin_s = df_fin[df_fin["代號"]==code].sort_values("日期")

    if not yoy_s.empty:
        st.plotly_chart(px.line(yoy_s,x="日期",y="年增率",title=f"{code} 月營收年增率",markers=True),use_container_width=True)
    if not fin_s.empty:
        ratio_cols=[c for c in ["毛利率","營益率","營業金流"] if c in fin_s.columns]
        dfp=fin_s.melt(id_vars="日期",value_vars=ratio_cols,var_name="指標",value_name="數值")
        st.plotly_chart(px.line(dfp,x="日期",y="數值",color="指標",title=f"{code} 財務比率趨勢"),use_container_width=True)
        latest=fin_s.dropna().tail(1)
        categories=["毛利率","營益率","營業金流"]
        vals=[latest.get(c,pd.Series([0])).values[0] for c in categories]
        scaled=normalize(vals,[fin_s[c] for c in categories])
        fig3=go.Figure()
        fig3.add_trace(go.Scatterpolar(r=scaled,theta=categories,fill="toself",name="最新"))
        fig3.update_layout(title=f"{code} 財務雷達圖",polar=dict(radialaxis=dict(range=[0,100])))
        st.plotly_chart(fig3,use_container_width=True)

st.caption("☁️ 本版本直接從線上 CSV 讀取資料（請將網址改成你的 GitHub Raw 連結）")
