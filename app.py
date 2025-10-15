# -*- coding: utf-8 -*-
# Streamlit App：Goodinfo 年增率 + 財務比率 + 雷達圖（Render / Google Drive 版）
# ---------------------------------------------------------------
# 特色：
# - 以 Google Drive 下載連結讀取四個 CSV（穩定、可公開存取）
# - 自動偵測編碼 (utf-8-sig / utf-8 / big5)
# - 保留原本儀表板所有視覺化功能（K 線、年增率、財務比率、雷達圖、排行、連續成長）
# - 新增「折線圖樣式」切換（markers only 或 lines+markers）
# - 已移除所有 Selenium 與「資料更新控制區」
# ---------------------------------------------------------------

import os, re, time, copy, io, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots
from requests.exceptions import HTTPError, RequestException

# =========================
# 0) 全域設定
# =========================
st.set_page_config(page_title="年增率 + 財務比率分析儀表板 (Render/Drive)", layout="wide")

# === 你的 Google Drive 檔案 ID（任何知道連結者可檢視）===
# 若之後更新，只需替換這四個 ID 或在側邊欄貼入新的分享連結即可
GD_ID_YOY = "1sds9YcZi55eG3moooeueVHMsDVBx7JwB"
GD_ID_GM  = "1s8A_tFh4e8a1VxtYPJg0kocxoRjXlBIm"
GD_ID_OM  = "18r5PwDngcyzGf1wfGHWbOLmLeKqMdEyg"
GD_ID_CF  = "1gVgb0FpgRHPK1RW9_ym4HqsQeYZCUm1f"

# =========================
# 1) 下載/讀檔工具
# =========================
def gdrive_id_from_any(url_or_id: str) -> str:
    """接受 Google Drive 分享連結或純 ID，回傳 ID。"""
    if "/file/d/" in url_or_id:
        m = re.search(r"/file/d/([^/]+)/", url_or_id)
        if m:
            return m.group(1)
    return url_or_id.strip()

def gdrive_csv_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

@st.cache_data(show_spinner="Downloading CSV from Google Drive…", ttl=3600)
def read_csv_from_gdrive(file_id: str, timeout=15, max_retries=3) -> pd.DataFrame:
    """
    從 Google Drive 下載 CSV，嘗試多種編碼。
    會快取 1 小時以減少外部請求。
    """
    url = gdrive_csv_url(file_id)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            raw = r.content
            # 嘗試多種常見編碼
            for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    return df
                except Exception:
                    continue
            # 如果都不行，再用 pandas 自動推測（無 encoding）
            try:
                df = pd.read_csv(io.BytesIO(raw))
                return df
            except Exception as e:
                last_err = e
        except RequestException as e:
            last_err = e
            time.sleep(1.2)  # 簡單退避
    raise RuntimeError(f"Failed to read CSV from Google Drive after {max_retries} tries: {last_err}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).replace("\xa0"," ").replace("\u3000"," ").strip())
    return df

# =========================
# 2) Yahoo Finance：日 K 線
# =========================
@st.cache_data(show_spinner="Fetching price history from Yahoo…", ttl=3600)
def fetch_history_from_2019(symbol: str) -> pd.DataFrame:
    start_dt = int(datetime(2019,1,1).timestamp())
    end_dt = int(time.time())
    headers = {"User-Agent":"Mozilla/5.0"}
    def fetch_with_suffix(suffix):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}"
        params = {"period1":start_dt,"period2":end_dt,"interval":"1d",
                  "includePrePost":"false","events":"div,splits"}
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    try:
        data = fetch_with_suffix(".TW")
    except HTTPError:
        try:
            data = fetch_with_suffix(".TWO")
        except HTTPError:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    try:
        result = data["chart"]["result"][0]
        ts = result["timestamp"]; q = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose",[{}])[0].get("adjclose",[None]*len(ts))
        df = pd.DataFrame({
            "Open": q["open"], "High": q["high"], "Low": q["low"],
            "Close": q["close"], "Volume": q["volume"], "AdjClose": adjclose
        }, index=pd.to_datetime(ts, unit="s"))
        df = df.dropna(subset=["Open","Close"])
        for w in (5,10,20,60,120,240):
            df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
        return df
    except Exception:
        return pd.DataFrame()

# =========================
# 3) 資料載入（年增率 + 財務比率）
# =========================
@st.cache_data(show_spinner="Loading YOY data…", ttl=3600)
def load_yoy_data_from_drive(file_id: str) -> pd.DataFrame:
    df = read_csv_from_gdrive(file_id)
    df = normalize_columns(df)

    # 固定處理「平均 年增率」
    if "平均 年增率" in df.columns and "平均年增率" not in df.columns:
        df["平均年增率"] = df["平均 年增率"]
        df.drop(columns=["平均 年增率"], inplace=True, errors="ignore")

    # 固定使用「新產業分類」欄位，改名為「產業分類」
    if "新產業分類" in df.columns and "產業分類" not in df.columns:
        df.rename(columns={"新產業分類": "產業分類"}, inplace=True)
    if "產業分類" not in df.columns:
        raise KeyError("❌ 找不到『新產業分類』或『產業分類』欄位，請確認 CSV 格式。")

    # 抓出所有「年增率」欄位（排除平均）
    yoy_cols = [c for c in df.columns if ("年增率" in c) and (not str(c).strip().startswith("平均"))]
    df[yoy_cols] = df[yoy_cols].apply(pd.to_numeric, errors="coerce")

    # 寬轉長
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

@st.cache_data(show_spinner="Loading financial ratios…", ttl=3600)
def load_financial_ratios_from_drive(id_gm: str, id_om: str, id_cf: str) -> pd.DataFrame:
    gm = normalize_columns(read_csv_from_gdrive(id_gm))
    om = normalize_columns(read_csv_from_gdrive(id_om))
    cf = normalize_columns(read_csv_from_gdrive(id_cf))

    # 毛利率
    gm_cols = [c for c in gm.columns if ("毛利" in c and "%" in c) or re.search(r"\d{2}Q\d", str(c))]
    if "代號" not in gm.columns or "名稱" not in gm.columns:
        # 嘗試從「代號名稱」合欄拆
        maybe = gm.columns[1] if len(gm.columns) > 1 else None
        if maybe:
            gm[["代號","名稱"]] = gm[maybe].astype(str).str.extract(r"(\d{4})(.+)")
    gm_m = gm.melt(id_vars=[c for c in ["代號","名稱"] if c in gm.columns],
                   value_vars=[c for c in gm_cols if c not in ["代號","名稱"]],
                   var_name="期間", value_name="毛利率")
    gm_m["季度"] = gm_m["期間"].astype(str).str.extract(r"(\d{2}Q\d)")[0]
    gm_m = gm_m.dropna(subset=["季度"])
    gm_m["日期"] = pd.PeriodIndex(gm_m["季度"], freq="Q").to_timestamp("Q")

    # 營益率
    om_cols = [c for c in om.columns if ("營益" in c and "%" in c) or re.search(r"\d{2}Q\d", str(c))]
    if "代號" not in om.columns or "名稱" not in om.columns:
        maybe = om.columns[1] if len(om.columns) > 1 else None
        if maybe:
            om[["代號","名稱"]] = om[maybe].astype(str).str.extract(r"(\d{4})(.+)")
    om_m = om.melt(id_vars=[c for c in ["代號","名稱"] if c in om.columns],
                   value_vars=[c for c in om_cols if c not in ["代號","名稱"]],
                   var_name="期間", value_name="營益率")
    om_m["季度"] = om_m["期間"].astype(str).str.extract(r"(\d{2}Q\d)")[0]
    om_m = om_m.dropna(subset=["季度"])
    om_m["日期"] = pd.PeriodIndex(om_m["季度"], freq="Q").to_timestamp("Q")

    # 營業金流
    cf_cols = [c for c in cf.columns if re.match(r"\d{2}Q\d.*營業活動", str(c))]
    if "代號" not in cf.columns or "名稱" not in cf.columns:
        maybe = cf.columns[1] if len(cf.columns) > 1 else None
        if maybe:
            cf[["代號","名稱"]] = cf[maybe].astype(str).str.extract(r"(\d{4})(.+)")
    cf_m = cf.melt(id_vars=[c for c in ["代號","名稱"] if c in cf.columns],
                   value_vars=[c for c in cf_cols if c not in ["代號","名稱"]],
                   var_name="期間", value_name="營業金流")
    cf_m["季度"] = cf_m["期間"].astype(str).str.extract(r"(\d{2}Q\d)")[0]
    cf_m["日期"] = pd.PeriodIndex(cf_m["季度"], freq="Q").to_timestamp("Q")
    cf_m = cf_m[["代號","名稱","日期","營業金流"]]

    # 合併
    df_fin = gm_m.merge(om_m[["代號","名稱","日期","營益率"]], on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.merge(cf_m, on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.sort_values(["代號","日期"]).reset_index(drop=True)
    # 數值轉型
    for col in ["毛利率","營益率","營業金流"]:
        if col in df_fin.columns:
            df_fin[col] = pd.to_numeric(df_fin[col], errors="coerce")
    return df_fin

# =========================
# 4) 側邊欄：資料來源/選項
# =========================
st.sidebar.title("📂 查詢條件 / 控制面板")

with st.sidebar.expander("資料來源（可貼分享連結覆蓋）", True):
    in_yoy = st.text_input("年增率（Drive 連結或 ID）", GD_ID_YOY)
    in_gm  = st.text_input("毛利率（Drive 連結或 ID）", GD_ID_GM)
    in_om  = st.text_input("營益率（Drive 連結或 ID）", GD_ID_OM)
    in_cf  = st.text_input("營業金流（Drive 連結或 ID）", GD_ID_CF)

    # 正規化成 ID
    id_yoy = gdrive_id_from_any(in_yoy)
    id_gm  = gdrive_id_from_any(in_gm)
    id_om  = gdrive_id_from_any(in_om)
    id_cf  = gdrive_id_from_any(in_cf)

# 折線圖顯示樣式選擇
line_style = st.sidebar.radio("折線圖樣式（財務比率）", ["線條＋圓點", "只有圓點"], index=0)
markers_flag = True
mode_line = "lines+markers" if line_style == "線條＋圓點" else "markers"

show_yoy = st.sidebar.checkbox("📈 顯示月營收年增率（含產業平均）", True)
show_kline = st.sidebar.checkbox("🕯️ 顯示 K 線 + 均線（Yahoo）", True)
show_fin = st.sidebar.checkbox("📊 顯示財務比率（毛利/營益/金流）", True)
show_radar = st.sidebar.checkbox("🧭 年度雷達圖（毛利/營益/金流）", True)
show_radar_mix = st.sidebar.checkbox("🧭 綜合雷達圖（毛利/營益/金流/營收年增率）", True)
normalize_radar = st.sidebar.checkbox("⚖️ 雷達圖正規化 (0-100)", True)

# =========================
# 5) 讀取資料（Drive）
# =========================
with st.spinner("讀取資料中…"):
    df_yoy = load_yoy_data_from_drive(id_yoy)
    df_fin = load_financial_ratios_from_drive(id_gm, id_om, id_cf)

# 產業/股票選單
inds = sorted(df_yoy['產業分類'].dropna().unique())
sel_inds = st.sidebar.multiselect("選擇產業分類（可多選）", inds)
manual_input = st.sidebar.text_input("或輸入股票代號（逗號分隔）", "2330,1101")
manual_codes = [c.strip() for c in manual_input.split(',') if c.strip()]

filtered = df_yoy.copy()
if sel_inds:
    filtered = filtered[filtered['產業分類'].isin(sel_inds)]
if manual_codes:
    filtered = pd.concat([filtered, df_yoy[df_yoy['代號'].isin(manual_codes)]], ignore_index=True)

stocks = filtered[['代號', '名稱']].drop_duplicates()
opts = {f"{r['代號']} {r['名稱']}": r['代號'] for _, r in stocks.iterrows()}
default_keys = list(opts.keys())[:1] if len(opts) else []
selected = st.sidebar.multiselect("選擇股票", list(opts.keys()), default=default_keys)

st.markdown("## 年增率 + K 線 + 財務比率 儀表板（Drive 版）")

# =========================
# 6) 單股圖表區
# =========================
def normalize_values(values, all_data):
    if not normalize_radar: return values
    scaled = []
    for i, v in enumerate(values):
        col = all_data[i].dropna()
        if len(col) == 0:
            scaled.append(0); continue
        v = 0 if v is None or (isinstance(v, float) and np.isnan(v)) else v
        vmin, vmax = col.min(), col.max()
        if pd.isna(v) or pd.isna(vmin) or pd.isna(vmax):
            scaled.append(0); continue
        if vmax == vmin:
            scaled.append(50); continue
        scaled.append((v - vmin) / (vmax - vmin) * 100)
    return scaled

if len(selected) == 1:
    code = opts[selected[0]]

    # 年增率（該股）
    yoy_s = df_yoy[df_yoy["代號"] == code].sort_values("日期")

    # 產業平均
    if not yoy_s.empty:
        industry = yoy_s["產業分類"].iloc[0]
        ind_avg = df_yoy[df_yoy["產業分類"] == industry].groupby("日期")["年增率"].mean().reset_index()
    else:
        industry, ind_avg = "未知", pd.DataFrame(columns=["日期", "年增率"])

    # --- K 線 + 均線 + 月營收年增率 ---
    if show_kline:
        df_yf = fetch_history_from_2019(code)
        if df_yf.empty:
            st.warning(f"{code}.TW 無法從 Yahoo Finance 取得日線資料")
        else:
            vol_colors = np.where(df_yf["Close"] >= df_yf["Open"], "red", "green")
            fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                  row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{}]])
            fig_k.add_trace(
                go.Candlestick(x=df_yf.index, open=df_yf["Open"], high=df_yf["High"],
                               low=df_yf["Low"], close=df_yf["Close"], name="K 線",
                               increasing_line_color='red', decreasing_line_color='green'),
                row=1, col=1, secondary_y=False
            )
            for w in (5, 10, 20, 60, 120, 240):
                fig_k.add_trace(go.Scatter(x=df_yf.index, y=df_yf[f"MA{w}"], mode="lines", name=f"MA{w}"),
                                row=1, col=1, secondary_y=False)
            if not yoy_s.empty and show_yoy:
                fig_k.add_trace(go.Scatter(x=yoy_s["日期"], y=yoy_s["年增率"], mode="lines+markers",
                                           name=f"{code} 年增率", line=dict(dash="dot")),
                                row=1, col=1, secondary_y=True)
            if not ind_avg.empty and show_yoy:
                fig_k.add_trace(go.Scatter(x=ind_avg["日期"], y=ind_avg["年增率"], mode="lines+markers",
                                           name=f"{industry} 平均年增率", line=dict(dash="dash")),
                                row=1, col=1, secondary_y=True)
            fig_k.add_trace(go.Bar(x=df_yf.index, y=df_yf["Volume"], marker_color=vol_colors,
                                   name="成交量", showlegend=False), row=2, col=1)
            fig_k.update_layout(
                title=f"🕯️ {code}.TW K 線 + 均線 + 成交量 + 月營收年增率 (含產業平均)",
                hovermode="x unified", height=760, dragmode="pan",
                xaxis=dict(rangeslider=dict(visible=True), type="date"),
                yaxis=dict(title="股價"),
                yaxis2=dict(title="月營收年增率 (%)", overlaying="y", side="right", showgrid=False),
                yaxis3=dict(title="成交量")
            )
            st.plotly_chart(fig_k, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

# =========================
# 7) 財務比率 + 雷達圖
# =========================
if (show_fin or show_radar or show_radar_mix) and len(selected) == 1:
    code = opts[selected[0]]
    fin_s = df_fin[df_fin["代號"] == code].sort_values("日期")
    yoy_s = df_yoy[df_yoy["代號"] == code].sort_values("日期")

    if show_fin:
        st.markdown("### 📊 財務比率趨勢")
        # 折線圖：毛利率 / 營益率
        ratio_cols = [c for c in ["毛利率", "營益率"] if c in fin_s.columns]
        if ratio_cols and not fin_s.empty:
            df_ratio = fin_s[["日期"] + ratio_cols].dropna(subset=["日期"])
            df_plot = df_ratio.melt(id_vars="日期", var_name="指標", value_name="數值")
            df_plot["數值"] = pd.to_numeric(df_plot["數值"], errors="coerce")
            df_plot = df_plot.sort_values("日期")

            fig = px.line(df_plot, x="日期", y="數值", color="指標", title=f"{code} 毛利率 / 營益率",
                          markers=(mode_line != "lines"))
            # 使用者選項：只顯示圓點 or 線+點
            fig.update_traces(mode=mode_line, connectgaps=True)

            fig.update_layout(template="plotly_dark", title_x=0.05,
                              hovermode="x unified", height=500, legend_title_text="指標")
            st.plotly_chart(fig, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

        # 現金流量柱狀圖
        if "營業金流" in fin_s.columns and not fin_s["營業金流"].dropna().empty:
            df_cf = fin_s[["日期", "營業金流"]].dropna().sort_values("日期")
            fig_cf = px.bar(df_cf, x="日期", y="營業金流", title=f"{code} 營業活動現金流量（億）")
            fig_cf.update_layout(template="plotly_dark", hovermode="x unified", height=400)
            st.plotly_chart(fig_cf, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

    # 年份清單（雷達圖）
    years_fin = sorted(fin_s["日期"].dt.year.dropna().unique().tolist()) if not fin_s.empty else []
    years_yoy = sorted(yoy_s["日期"].dt.year.dropna().unique().tolist()) if not yoy_s.empty else []
    all_years = sorted(set(years_fin) | set(years_yoy))
    default_years = all_years[-1:] if all_years else []

    # 年度雷達圖
    if show_radar:
        st.markdown("### 🧭 年度雷達圖（毛利率 / 營益率 / 營業金流）")
        st.caption("💡 雙擊圖中心可回復原始大小")
        chosen_years = st.multiselect("選擇年份（財務比率雷達圖）", all_years, default=default_years, key="radar_fin_years")
        if chosen_years:
            categories = ["毛利率","營益率","營業金流"]
            all_data = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"]]
            fig_radar = go.Figure(); colors = px.colors.qualitative.Bold
            for i, yr in enumerate(chosen_years):
                color = colors[i % len(colors)]
                yr_df = fin_s[fin_s["日期"].dt.year == yr].sort_values("日期").tail(1)
                values = [
                    float(yr_df["毛利率"].values[0]) if not yr_df.empty and pd.notna(yr_df["毛利率"].values[0]) else 0,
                    float(yr_df["營益率"].values[0]) if not yr_df.empty and pd.notna(yr_df["營益率"].values[0]) else 0,
                    float(yr_df["營業金流"].values[0]) if not yr_df.empty and pd.notna(yr_df["營業金流"].values[0]) else 0,
                ]
                scaled = normalize_values(values, all_data)
                fig_radar.add_trace(go.Scatterpolar(
                    r=scaled, theta=categories, fill="toself", name=str(yr),
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("rgb","rgba").replace(")",",0.3)"),
                    mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in values], textfont=dict(color="black"),
                    textposition="top center"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if normalize_radar else None)),
                title=f"{code} 財務比率雷達圖（年度比較）", showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displaylogo": False})

    # 綜合雷達圖
    if show_radar_mix:
        st.markdown("### 🧭 綜合雷達圖（毛利率 / 營益率 / 營業金流 / 營收年增率）")
        st.caption("💡 雙擊圖中心可回復原始大小")
        chosen_years2 = st.multiselect("選擇年份（綜合雷達圖）", all_years, default=default_years, key="radar_mix_years")
        if chosen_years2:
            categories_all = ["毛利率","營益率","營業金流","月營收年增率"]
            all_data = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"], yoy_s["年增率"]]
            fig_radar_all = go.Figure(); colors = px.colors.qualitative.Dark24
            for i, yr in enumerate(chosen_years2):
                color = colors[i % len(colors)]
                latest_fin = fin_s[fin_s["日期"].dt.year == yr].sort_values("日期").tail(1)
                latest_yoy = yoy_s[yoy_s["日期"].dt.year == yr].sort_values("日期").tail(1)
                values = [
                    float(latest_fin["毛利率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["毛利率"].values[0]) else 0,
                    float(latest_fin["營益率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營益率"].values[0]) else 0,
                    float(latest_fin["營業金流"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營業金流"].values[0]) else 0,
                    float(latest_yoy["年增率"].values[0]) if not latest_yoy.empty and pd.notna(latest_yoy["年增率"].values[0]) else 0,
                ]
                scaled = normalize_values(values, all_data)
                fig_radar_all.add_trace(go.Scatterpolar(
                    r=scaled, theta=categories_all, fill="toself", name=str(yr),
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("rgb","rgba").replace(")",",0.3)"),
                    mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in values], textfont=dict(color="black"),
                    textposition="top center"
                ))
            fig_radar_all.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if normalize_radar else None)),
                title=f"{code} 綜合財務 + 營收雷達圖（年度比較）", showlegend=True
            )
            st.plotly_chart(fig_radar_all, use_container_width=True, config={"displaylogo": False})

# =========================
# 8) 平均年增率排行榜
# =========================
with st.expander("🏆 平均年增率排行榜 Top 10", True):
    try:
        df_raw = normalize_columns(read_csv_from_gdrive(id_yoy))
        avg_col = next((c for c in df_raw.columns if c.strip() in ("平均年增率","平均 年增率")), None)
        if avg_col:
            df_avg = df_raw[["代號","名稱", avg_col]].rename(columns={avg_col:"平均年增率"}).dropna()
            df_avg["平均年增率"] = pd.to_numeric(df_avg["平均年增率"], errors="coerce")
            rank_df = df_avg.sort_values("平均年增率", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(rank_df.style.format({"平均年增率": "{:.2f}%"}), use_container_width=True)
        else:
            st.info("原始檔未包含『平均年增率』欄位")
    except Exception as e:
        st.warning(f"排行榜生成失敗：{e}")

# =========================
# 9) 近三個月年增率連續成長
# =========================
with st.expander("📈 近三個月年增率連續成長", True):
    df_temp = df_yoy.dropna(subset=["日期"])
    unique_months = sorted(df_temp["日期"].unique())
    last3 = unique_months[-3:] if len(unique_months) >= 3 else []
    result = []
    if last3:
        df_l3 = df_temp[df_temp["日期"].isin(last3)]
        for sid in df_l3["代號"].unique():
            d = df_l3[df_l3["代號"] == sid].sort_values("日期")
            if len(d) == 3:
                y1,y2,y3 = d["年增率"].values
                if pd.notna(y1) and pd.notna(y2) and pd.notna(y3) and y1 < y2 < y3:
                    result.append({
                        "代號":sid, "名稱":d.iloc[0]["名稱"], "產業分類":d.iloc[0]["產業分類"],
                        "月份1":pd.Timestamp(d.iloc[0]["日期"]).strftime("%Y-%m"), "年增率1":round(float(y1),2),
                        "月份2":pd.Timestamp(d.iloc[1]["日期"]).strftime("%Y-%m"), "年增率2":round(float(y2),2),
                        "月份3":pd.Timestamp(d.iloc[2]["日期"]).strftime("%Y-%m"), "年增率3":round(float(y3),2)
                    })
    if result:
        df_res = pd.DataFrame(result)
        industries = ["全部顯示"] + sorted(df_res["產業分類"].dropna().unique())
        sel_ind = st.selectbox("選擇產業分類（篩選）", industries, index=0)
        if sel_ind != "全部顯示":
            df_res = df_res[df_res["產業分類"] == sel_ind]
        st.dataframe(df_res, use_container_width=True)
    else:
        st.info("目前沒有股票符合『近三個月連續成長』條件。")

# =========================
# 尾註
# =========================
st.caption("小提醒：年增率為『月資料』，財務比率為『季資料』。資料來源：你提供的 Google Drive CSV。")
