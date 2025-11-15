# -*- coding: utf-8 -*-
# Streamlit App：Goodinfo 年增率 + 財務比率 + 雷達圖（只讀 / Google Drive 版）
# ---------------------------------------------------------------
import os, re, time, copy, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots
from requests.exceptions import HTTPError

# =========================
# 0) 全域設定 & 路徑集中管理（Google Drive 來源）
# =========================
st.set_page_config(page_title="年增率 + 財務比率分析儀表板（只讀雲端版）", layout="wide")

# 你的 Google Drive 分享連結（維持你提供的四個）
URL_YOY = "https://drive.google.com/file/d/1sds9YcZi55eG3moooeueVHMsDVBx7JwB/view?usp=sharing"
URL_GM  = "https://drive.google.com/file/d/1s8A_tFh4e8a1VxtYPJg0kocxoRjXlBIm/view?usp=sharing"
URL_OM  = "https://drive.google.com/file/d/18r5PwDngcyzGf1wfGHWbOLmLeKqMdEyg/view?usp=sharing"
URL_CF  = "https://drive.google.com/file/d/1gVgb0FpgRHPK1RW9_ym4HqsQeYZCUm1f/view?usp=sharing"

YF_HEADERS = {"User-Agent":"Mozilla/5.0"}

# =========================
# 工具：Drive 直連、讀檔、防掛
# =========================
def gdrive_to_direct(url: str) -> str:
    """把 'file/d/<id>/view?...' 轉為可下載直連 uc?export=download&id=<id>"""
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        return url
    fid = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={fid}"

def robust_read_csv(src: str, **kwargs) -> pd.DataFrame:
    """
    穩健讀取 CSV：
    - 自動將 GDrive 分享連結轉直連
    - 多編碼嘗試（utf-8-sig/utf-8/big5/cp950）
    - 3 次重試
    """
    path = gdrive_to_direct(src)
    encodings = [kwargs.pop("encoding", None), "utf-8-sig", "utf-8", "big5", "cp950"]
    tries = 3
    last_err = None
    for _ in range(tries):
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except Exception as e:
                last_err = e
        time.sleep(0.8)
    raise last_err

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).replace("\xa0"," ").replace("\u3000"," ").strip())
    return df

# =========================
# 1) 年增率：載入（寬轉長、年月->日期）
# =========================
@st.cache_data(show_spinner=True)
def load_yoy_data(src: str) -> pd.DataFrame:
    df = robust_read_csv(src)
    df = clean_columns(df)

    # 固定處理「平均 年增率」欄位
    if "平均 年增率" in df.columns and "平均年增率" not in df.columns:
        df["平均年增率"] = df["平均 年增率"]
        df.drop(columns=["平均 年增率"], inplace=True)

    # ✅ 固定使用「新產業分類」欄位，改名為「產業分類」
    if "新產業分類" in df.columns:
        df.rename(columns={"新產業分類": "產業分類"}, inplace=True)
    elif "產業分類" not in df.columns:
        raise KeyError("❌ 找不到『新產業分類』或『產業分類』欄位，請確認 CSV 格式。")

    # 抓出所有「年增率」欄位（排除平均）
    yoy_cols = [c for c in df.columns if ("年增率" in c) and (not str(c).strip().startswith("平均"))]
    df[yoy_cols] = df[yoy_cols].apply(pd.to_numeric, errors="coerce")

    # 寬轉長格式（melt）
    df_m = df.melt(id_vars=["代號","名稱","產業分類"], value_vars=yoy_cols,
                   var_name="期間", value_name="年增率")

    # 解析「25M06」→ Timestamp(2025, 6, 1)
    def parse_month_to_date(month_str):
        m = re.search(r"(\d{2})M(\d{2})", str(month_str))
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=2000 + y, month=mo, day=1)
        return pd.NaT

    df_m["日期"] = df_m["期間"].apply(parse_month_to_date)
    return df_m

# =========================
# =========================
# 2) 財務比率（毛利/營益/金流）載入與整理
# =========================
@st.cache_data(show_spinner=True)
def load_financial_ratios(url_gm: str, url_om: str, url_cf: str) -> pd.DataFrame:
    """
    從 Google Drive 載入毛利率、營益率、營業金流三份 CSV，
    自動容錯欄位名稱、移除空白、若最新季度資料缺值過多則沿用上一季。
    """
    import re as _re

    # === 強化欄位偵測與清理 ===
    def _load_and_detect(url, keyword_pattern):
        df = robust_read_csv(url)
        df = clean_columns(df)
        cols = []
        for c in df.columns:
            c_clean = re.sub(r"[\s\u3000\xa0]+", "", str(c))  # 移除全形與半形空白
            if re.search(r"\d{2}Q\d", c_clean) and re.search(keyword_pattern, c_clean):
                cols.append(c)
        return df, cols

    # --- 分別載入三份資料 ---
    gm, gm_cols = _load_and_detect(url_gm, r"毛利")
    om, om_cols = _load_and_detect(url_om, r"(營業利益|營益)")
    cf, cf_cols = _load_and_detect(url_cf, r"(營業活動|現金流)")

    # 若仍抓不到欄位就提示
    if not gm_cols:
        st.warning("⚠️ 無法偵測到『毛利率』欄位，請確認檔案是否包含『毛利率(%)』等欄名。")
    if not om_cols:
        st.warning("⚠️ 無法偵測到『營益率』欄位，請確認檔案是否包含『營業利益率(%)』或『營益率(%)』等欄名。")
    if not cf_cols:
        st.warning("⚠️ 無法偵測到『營業活動現金流量』欄位，請確認檔案格式。")

    # === 通用季度補值邏輯（三者共用） ===
    def apply_latest_quarter_fix(df, cols, label):
        if not cols:
            return df, f"⚠️ 無 {label} 資料"
        last_col = cols[-1]
        prev_col = cols[-2] if len(cols) > 1 else None
        valid_ratio = df[last_col].notna().sum() / max(len(df), 1)
        if valid_ratio < 0.3 and prev_col:
            df[last_col] = df[prev_col]
            return df, f"⚠️ 最新季度「{last_col}」({label}) 未完整，暫以「{prev_col}」沿用"
        return df, f"✅ 使用最新季度「{last_col}」({label})"

    gm, gm_status = apply_latest_quarter_fix(gm, gm_cols, "毛利率")
    om, om_status = apply_latest_quarter_fix(om, om_cols, "營益率")
    cf, cf_status = apply_latest_quarter_fix(cf, cf_cols, "營業金流")

    # === 寬轉長 ===
    def melt_and_parse(df, cols, val_name):
        df_m = df.melt(id_vars=["代號", "名稱"], value_vars=cols,
                       var_name="期間", value_name=val_name)
        df_m["季度"] = df_m["期間"].str.extract(r"(\d{2}Q\d)")[0]
        df_m = df_m.dropna(subset=["季度"])
        df_m["日期"] = pd.PeriodIndex(df_m["季度"], freq="Q").to_timestamp("Q")
        return df_m

    gm_m = melt_and_parse(gm, gm_cols, "毛利率")
    om_m = melt_and_parse(om, om_cols, "營益率")
    cf_m = melt_and_parse(cf, cf_cols, "營業金流")

    # === 合併 & 狀態說明 ===
    status_flag = f"{gm_status}；{om_status}；{cf_status}"
    gm_m["更新狀態"] = status_flag
    om_m["更新狀態"] = status_flag
    cf_m["更新狀態"] = status_flag

    df_fin = gm_m.merge(
        om_m[["代號", "名稱", "日期", "營益率"]], on=["代號", "名稱", "日期"], how="outer"
    ).merge(
        cf_m[["代號", "名稱", "日期", "營業金流"]], on=["代號", "名稱", "日期"], how="outer"
    )

    df_fin = df_fin.sort_values(["代號", "日期"]).reset_index(drop=True)
    df_fin["更新狀態"] = status_flag
    return df_fin



# =========================
# 3) Yahoo Finance：日 K 線
# =========================
@st.cache_data(show_spinner=True)
def fetch_history_from_2019(symbol: str) -> pd.DataFrame:
    start_dt = int(datetime(2019,1,1).timestamp())
    end_dt = int(time.time())

    def fetch_with_suffix(suffix):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}"
        params = {"period1":start_dt,"period2":end_dt,"interval":"1d",
                  "includePrePost":"false","events":"div,splits"}
        resp = requests.get(url, params=params, headers=YF_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    try:
        data = fetch_with_suffix(".TW")
    except HTTPError:
        try:
            data = fetch_with_suffix(".TWO")
        except HTTPError:
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
# 4) 主題與 UI 控制
# =========================
st.sidebar.title("📂 查詢條件 / 控制面板")

# 主題
theme_choice = st.sidebar.radio("主題 Theme", ["🌞 淺色", "🌙 深色"], index=0)
is_dark = (theme_choice == "🌙 深色")
plotly_template = "plotly_dark" if is_dark else "plotly"

# 全域字體大小
font_size = st.sidebar.slider("🔠 全域字體大小", min_value=12, max_value=24, value=16, step=1)

# 財務比率折線樣式
line_mode_choice = st.sidebar.radio("財務比率折線樣式", ["線", "線 + 點"], index=1)
line_mode = "lines+markers" if line_mode_choice == "線 + 點" else "lines"

# 連續成長月數
grow_n = st.sidebar.slider("📈 連續成長月數（年增率）", min_value=1, max_value=12, value=3)

# 雷達圖正規化
norm_choice = st.sidebar.selectbox(
    "雷達圖正規化方法",
    ["不正規化", "MinMax 0-100（全體）", "MinMax 0-100（僅該股）", "Z-score（全體）"],
    index=1
)
show_real_labels = st.sidebar.checkbox("雷達圖顯示『實際值』標籤", True)

# 重新載入資料
if st.sidebar.button("🔄 重新載入資料"):
    load_yoy_data.clear()
    load_financial_ratios.clear()
    fetch_history_from_2019.clear()
    st.rerun()  # ✅ 新版 Streamlit 用這個

# 顏色樣式
BG_MAIN = "#0E1117" if is_dark else "#FFFFFF"
TEXT_COLOR = "#FFFFFF" if is_dark else "#111111"
PLOT_BG = "#111418" if is_dark else "#FFFFFF"
PAPER_BG = "#0E1117" if is_dark else "#FFFFFF"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG_MAIN};
        color: {TEXT_COLOR};
    }}
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: {TEXT_COLOR} !important;
        font-size: {font_size}px !important;
    }}
    .stButton>button {{
        font-size: {max(12, font_size-2)}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 5) 載入資料
# =========================
with st.sidebar.expander("資料來源（Drive 直讀）", True):
    st.caption(f"年增率：{URL_YOY}")
    st.caption(f"毛利率：{URL_GM}")
    st.caption(f"營益率：{URL_OM}")
    st.caption(f"現金流量：{URL_CF}")

with st.spinner("Loading YoY & Financial ratios ..."):
    df_yoy = load_yoy_data(URL_YOY)
    df_fin = load_financial_ratios(URL_GM, URL_OM, URL_CF)

# =========================
# 6) 篩選控制
# =========================
inds = sorted(df_yoy['產業分類'].dropna().unique())
sel_inds = st.sidebar.multiselect("選擇產業分類（可多選）", inds)
manual_input = st.sidebar.text_input("或輸入股票代號（逗號分隔）", "2330,1101")
manual_codes = [c.strip() for c in manual_input.split(',') if c.strip()]

filtered = df_yoy.copy()
if sel_inds:
    filtered = filtered[filtered['產業分類'].isin(sel_inds)]
if manual_codes:
    filtered = pd.concat([filtered, df_yoy[df_yoy['代號'].isin(manual_codes)]])

stocks = filtered[['代號', '名稱']].drop_duplicates()
opts = {f"{r['代號']} {r['名稱']}": r['代號'] for _, r in stocks.iterrows()}

default_keys = list(opts.keys())[:1] if len(opts) else []
selected = st.sidebar.multiselect("選擇股票", list(opts.keys()), default=default_keys)

show_yoy = st.sidebar.checkbox("📈 顯示月營收年增率（含產業平均）", True)
show_kline = st.sidebar.checkbox("🕯️ 顯示 K 線 + 均線（Yahoo）", True)
show_fin = st.sidebar.checkbox("📊 顯示財務比率（毛利/營益/金流）", True)
show_radar = st.sidebar.checkbox("🧭 年度雷達圖（毛利/營益/金流）", True)
show_radar_mix = st.sidebar.checkbox("🧭 綜合雷達圖（毛利/營益/金流/營收年增率）", True)

st.markdown(f"<h2 style='margin-top:0'>年增率 + K 線 + 財務比率 儀表板</h2>", unsafe_allow_html=True)

# =========================
# 7) 正規化工具（雷達圖用）
# =========================
def normalize_series(value, series, method="MinMaxAll", fallback=50.0):
    if method == "None":
        return float(value) if pd.notna(value) else 0.0
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value):
        return fallback
    v = float(value)
    if method in ("MinMaxAll", "MinMaxSelf"):
        lo, hi = s.min(), s.max()
        if np.isclose(hi, lo):
            return fallback
        return (v - lo) / (hi - lo) * 100.0
    if method == "ZAll":
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0:
            return fallback
        z = (v - mu) / sigma
        z = max(min(z, 3.0), -3.0)
        return (z + 3.0) / 6.0 * 100.0
    return float(value)

def get_norm_method(choice: str):
    return {
        "不正規化": "None",
        "MinMax 0-100（全體）": "MinMaxAll",
        "MinMax 0-100（僅該股）": "MinMaxSelf",
        "Z-score（全體）": "ZAll",
    }.get(choice, "MinMaxAll")

# =========================
# 8) 單一股票：年增率 + 產業平均 + K 線
# =========================
if len(selected) == 1:
    code = opts[selected[0]]
    yoy_s = df_yoy[df_yoy["代號"] == code].sort_values("日期")

    # 產業平均
    if not yoy_s.empty:
        industry = yoy_s["產業分類"].iloc[0]
        ind_avg = df_yoy[df_yoy["產業分類"] == industry].groupby("日期")["年增率"].mean().reset_index()
    else:
        industry, ind_avg = "未知", pd.DataFrame(columns=["日期", "年增率"])

    # --- K 線 + 均線 + 年增率 ---
    if show_kline:
        df_yf = fetch_history_from_2019(code)
        if df_yf.empty:
            st.warning(f"{code}.TW 無法從 Yahoo Finance 取得日線資料")
        else:
            vol_colors = np.where(df_yf["Close"] >= df_yf["Open"], "#E13D3D", "#2DB77E")
            fig_k = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                row_heights=[0.78, 0.22], specs=[[{"secondary_y": True}], [{}]]
            )
            fig_k.add_trace(
                go.Candlestick(x=df_yf.index, open=df_yf["Open"], high=df_yf["High"],
                               low=df_yf["Low"], close=df_yf["Close"], name="K 線",
                               increasing_line_color='#E13D3D', decreasing_line_color='#2DB77E'),
                row=1, col=1, secondary_y=False
            )
            for w in (5, 10, 20, 60, 120, 240):
                fig_k.add_trace(go.Scatter(x=df_yf.index, y=df_yf[f"MA{w}"], mode="lines", name=f"MA{w}"),
                                row=1, col=1, secondary_y=False)

            if show_yoy and not yoy_s.empty:
                fig_k.add_trace(go.Scatter(x=yoy_s["日期"], y=yoy_s["年增率"], mode="lines+markers",
                                           name=f"{code} 年增率", line=dict(dash="dot")),
                                row=1, col=1, secondary_y=True)
            if show_yoy and not ind_avg.empty:
                fig_k.add_trace(go.Scatter(x=ind_avg["日期"], y=ind_avg["年增率"], mode="lines+markers",
                                           name=f"{industry} 平均年增率", line=dict(dash="dash")),
                                row=1, col=1, secondary_y=True)
            fig_k.add_trace(go.Bar(x=df_yf.index, y=df_yf["Volume"], marker_color=vol_colors,
                                   name="成交量", showlegend=False), row=2, col=1)

            fig_k.update_layout(
                template=plotly_template,
                title=f"{code}.TW K 線 + 均線 + 成交量 + 月營收年增率 (含產業平均)",
                hovermode="x unified", height=760, dragmode="pan",
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
                xaxis=dict(rangeslider=dict(visible=True), type="date"),
                yaxis=dict(title="股價"),
                yaxis2=dict(title="月營收年增率 (%)", overlaying="y", side="right", showgrid=False),
                yaxis3=dict(title="成交量")
            )
            st.plotly_chart(fig_k, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

# =========================
# 9) 財務比率（單股） + 年度雷達圖 + 綜合雷達圖
# =========================
def radar_text_color():
    return "#E5E7EB" if is_dark else "#111827"

# === 個股沿用上一季邏輯 ===
def get_latest_with_fallback(df, year):
    """取指定年度的最後一季；若該年無有效值則沿用上一季"""
    df_y = df[df["日期"].dt.year == year].sort_values("日期")
    if not df_y.empty:
        return df_y.tail(1)
    # 若該年無資料 → 向前找上一季
    df_prev = df[df["日期"] < pd.Timestamp(year=year, month=1, day=1)].sort_values("日期")
    if not df_prev.empty:
        last_prev = df_prev.tail(1)
        st.warning(f"⚠️ {year} 年資料未公布，已沿用上一季（{last_prev['日期'].iloc[0].strftime('%Y-%m-%d')}）")
        return last_prev
    return pd.DataFrame()  # 若無上一季也沒資料則回傳空

if (show_fin or show_radar or show_radar_mix) and len(selected) == 1:
    code = opts[selected[0]]
    fin_s = df_fin[df_fin["代號"] == code].sort_values("日期")
    yoy_s_single = df_yoy[df_yoy["代號"] == code].sort_values("日期")

    # 顯示沿用狀態提示
    if show_fin and "更新狀態" in fin_s.columns:
        last_status = [s for s in fin_s["更新狀態"].unique().tolist() if isinstance(s, str)]
        if last_status:
            st.info("；".join(last_status))

    # === 將季度轉換為連續日期（例如 24Q2 → 2024-06-30） ===
    def convert_quarter_to_date(df):
        if "日期" in df.columns:
            df = df.copy()
            if df["日期"].dtype == "object" or (len(df) and isinstance(df["日期"].iloc[0], str)):
                df["日期"] = df["日期"].astype(str).str.extract(r"(\d{2}Q\d)")
                df["日期"] = pd.to_datetime(
                    df["日期"]
                    .str.replace("Q1", "-03-31")
                    .str.replace("Q2", "-06-30")
                    .str.replace("Q3", "-09-30")
                    .str.replace("Q4", "-12-31"),
                    errors="coerce"
                )
        return df

    fin_s = convert_quarter_to_date(fin_s)

    # 折線圖：毛利率 / 營益率
    if show_fin:
        st.markdown("### 📊 財務比率趨勢")
        ratio_cols = [c for c in ["毛利率", "營益率"] if c in fin_s.columns]
        if ratio_cols and not fin_s.empty:
            df_ratio = fin_s[["日期"] + ratio_cols].dropna(subset=["日期"])
            df_plot = df_ratio.melt(id_vars="日期", var_name="指標", value_name="數值")
            df_plot["數值"] = pd.to_numeric(df_plot["數值"], errors="coerce")
            df_plot = df_plot.sort_values("日期")
            fig = px.line(df_plot, x="日期", y="數值", color="指標", title=f"{code} 毛利率 / 營益率")
            for tr in fig.data:
                tr.mode = line_mode
                tr.connectgaps = True
            fig.update_layout(
                template=plotly_template, title_x=0.05, hovermode="x unified",
                height=500, legend_title_text="指標",
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

        # 現金流量柱狀圖
        if "營業金流" in fin_s.columns and not fin_s["營業金流"].dropna().empty:
            df_cf = fin_s[["日期", "營業金流"]].dropna().sort_values("日期")
            fig_cf = px.bar(df_cf, x="日期", y="營業金流",
                            title=f"{code} 營業活動現金流量（億）")
            fig_cf.update_layout(
                template=plotly_template, hovermode="x unified", height=400,
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig_cf, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

    # 年份篩選
    years_fin = sorted(fin_s["日期"].dt.year.dropna().unique().tolist()) if not fin_s.empty else []
    years_yoy = sorted(yoy_s_single["日期"].dt.year.dropna().unique().tolist()) if not yoy_s_single.empty else []
    all_years = sorted(set(years_fin) | set(years_yoy))
    default_years = all_years[-1:] if all_years else []

    # 正規化方法
    norm_method = get_norm_method(norm_choice)
    def norm_value(v, all_series, self_series):
        if norm_method == "None":
            return v
        if norm_method == "MinMaxSelf":
            return normalize_series(v, self_series, method="MinMaxSelf", fallback=50.0)
        if norm_method == "MinMaxAll":
            return normalize_series(v, all_series, method="MinMaxAll", fallback=50.0)
        if norm_method == "ZAll":
            return normalize_series(v, all_series, method="ZAll", fallback=50.0)
        return v

    # 年度雷達圖
    if show_radar:
        st.markdown("### 🧭 年度雷達圖（毛利率 / 營益率 / 營業金流）")
        st.caption("💡 提示：若該年資料未公布，將自動沿用上一季。")
        chosen_years = st.multiselect("選擇年份（財務比率雷達圖）", all_years, default=default_years, key="radar_fin_years")
        if chosen_years:
            categories = ["毛利率","營益率","營業金流"]
            all_data = [df_fin["毛利率"], df_fin["營益率"], df_fin["營業金流"]]
            self_data = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"]]
            fig_radar = go.Figure()
            for yr in chosen_years:
                latest_fin = get_latest_with_fallback(fin_s, yr)
                real_vals = [
                    float(latest_fin["毛利率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["毛利率"].values[0]) else 0.0,
                    float(latest_fin["營益率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營益率"].values[0]) else 0.0,
                    float(latest_fin["營業金流"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營業金流"].values[0]) else 0.0,
                ]
                norm_vals = [
                    norm_value(real_vals[0], all_data[0], self_data[0]),
                    norm_value(real_vals[1], all_data[1], self_data[1]),
                    norm_value(real_vals[2], all_data[2], self_data[2]),
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm_vals, theta=categories, fill="toself", name=str(yr),
                    mode="lines+markers+text" if show_real_labels else "lines+markers",
                    text=[f"{v:.1f}" for v in real_vals] if show_real_labels else None,
                    textfont=dict(color=radar_text_color(), size=font_size),
                ))
            fig_radar.update_layout(
                template=plotly_template,
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if norm_method != "None" else None)),
                title=f"{code} 財務比率雷達圖（年度比較）",
                showlegend=True,
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displaylogo": False})

    # 綜合雷達圖
    if show_radar_mix:
        st.markdown("### 🧭 綜合雷達圖（毛利率 / 營益率 / 營業金流 / 月營收年增率）")
        st.caption("💡 若該年資料未公布，將自動沿用上一季。")
        chosen_years2 = st.multiselect("選擇年份（綜合雷達圖）", all_years, default=default_years, key="radar_mix_years")
        if chosen_years2:
            categories_all = ["毛利率","營益率","營業金流","月營收年增率"]
            all_data_mix = [df_fin["毛利率"], df_fin["營益率"], df_fin["營業金流"], df_yoy["年增率"]]
            self_data_mix = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"], yoy_s_single["年增率"]]
            fig_radar_all = go.Figure()
            for yr in chosen_years2:
                latest_fin = get_latest_with_fallback(fin_s, yr)
                latest_yoy = get_latest_with_fallback(yoy_s_single, yr)
                real_vals = [
                    float(latest_fin["毛利率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["毛利率"].values[0]) else 0.0,
                    float(latest_fin["營益率"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營益率"].values[0]) else 0.0,
                    float(latest_fin["營業金流"].values[0]) if not latest_fin.empty and pd.notna(latest_fin["營業金流"].values[0]) else 0.0,
                    float(latest_yoy["年增率"].values[0]) if not latest_yoy.empty and pd.notna(latest_yoy["年增率"].values[0]) else 0.0,
                ]
                norm_vals = [
                    norm_value(real_vals[0], all_data_mix[0], self_data_mix[0]),
                    norm_value(real_vals[1], all_data_mix[1], self_data_mix[1]),
                    norm_value(real_vals[2], all_data_mix[2], self_data_mix[2]),
                    norm_value(real_vals[3], all_data_mix[3], self_data_mix[3]),
                ]
                fig_radar_all.add_trace(go.Scatterpolar(
                    r=norm_vals, theta=categories_all, fill="toself", name=str(yr),
                    mode="lines+markers+text" if show_real_labels else "lines+markers",
                    text=[f"{real_vals[0]:.1f}%", f"{real_vals[1]:.1f}%", f"{real_vals[2]:.1f}", f"{real_vals[3]:.1f}%"] if show_real_labels else None,
                    textfont=dict(color=radar_text_color(), size=font_size)
                ))
            fig_radar_all.update_layout(
                template=plotly_template,
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if norm_method != "None" else None)),
                title=f"{code} 綜合財務 + 營收雷達圖（年度比較）",
                showlegend=True,
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig_radar_all, use_container_width=True, config={"displaylogo": False})


# =========================
# 10) 平均年增率排行榜
# =========================
with st.expander("🏆 平均年增率排行榜 Top 10", True):
    try:
        df_raw = robust_read_csv(URL_YOY)
        df_raw = clean_columns(df_raw)
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
# 11) 連續 N 個月 年增率連續成長（滑桿控制 N）
# =========================
with st.expander(f"📈 近 {grow_n} 個月年增率連續成長", True):
    df_temp = df_yoy.dropna(subset=["日期"]).copy()
    result = []
    unique_months = sorted(df_temp["日期"].dropna().unique())
    if len(unique_months) >= grow_n:
        target_months = unique_months[-grow_n:]
        df_lN = df_temp[df_temp["日期"].isin(target_months)]
        for sid in df_lN["代號"].unique():
            d = df_lN[df_lN["代號"] == sid].sort_values("日期")
            if len(d) == grow_n:
                vals = pd.to_numeric(d["年增率"], errors="coerce").values
                if np.all(np.diff(vals) > 0):  # 嚴格遞增
                    row = {"代號": sid, "名稱": d.iloc[0]["名稱"], "產業分類": d.iloc[0]["產業分類"]}
                    for i, (dt, v) in enumerate(zip(d["日期"], vals), start=1):
                        row[f"月份{i}"] = pd.to_datetime(dt).strftime("%Y-%m")
                        row[f"年增率{i}"] = round(float(v), 2)
                    result.append(row)
    if result:
        df_res = pd.DataFrame(result)
        industries = ["全部顯示"] + sorted(df_res["產業分類"].dropna().unique())
        sel_ind = st.selectbox("選擇產業分類（篩選）", industries, index=0)
        if sel_ind != "全部顯示":
            df_res = df_res[df_res["產業分類"] == sel_ind]
        st.dataframe(df_res, use_container_width=True)
    else:
        st.info(f"目前沒有股票符合『近 {grow_n} 個月連續成長』條件。")

# =========================
# 12) 多檔股票 vs 產業平均 年增率趨勢
# =========================
with st.expander("🧯 多檔股票與產業平均的年增率趨勢", False):
    all_years_yoy = sorted(df_yoy["日期"].dt.year.dropna().unique())
    if len(all_years_yoy) >= 1:
        start_y = st.selectbox("聚焦起始年", all_years_yoy, index=0, key="focus_start")
        end_y = st.selectbox("聚焦結束年", all_years_yoy, index=len(all_years_yoy)-1, key="focus_end")
    else:
        start_y, end_y = None, None

    sel_multi = st.multiselect("選擇多檔股票", list(opts.keys()), default=list(opts.keys())[:2])
    if sel_multi:
        fig_full = go.Figure()
        fig_focus = go.Figure()
        for sk in sel_multi:
            sid = opts[sk]
            s = df_yoy[df_yoy["代號"] == sid].sort_values("日期")
            if s.empty: continue
            ind = s["產業分類"].iloc[0]
            ind_avg2 = df_yoy[df_yoy["產業分類"] == ind].groupby("日期")["年增率"].mean().reset_index()
            fig_full.add_trace(go.Scatter(x=s["日期"], y=s["年增率"], mode="lines+markers", name=f"{sid}"))
            fig_full.add_trace(go.Scatter(x=ind_avg2["日期"], y=ind_avg2["年增率"], mode="lines+markers",
                                          name=f"{ind} 平均", line=dict(dash="dot")))
            if start_y and end_y:
                focus = s[(s["日期"].dt.year >= start_y) & (s["日期"].dt.year <= end_y)]
                ind_focus = ind_avg2[(ind_avg2["日期"].dt.year >= start_y) & (ind_avg2["日期"].dt.year <= end_y)]
                fig_focus.add_trace(go.Scatter(x=focus["日期"], y=focus["年增率"], mode="lines+markers", name=f"{sid}"))
                fig_focus.add_trace(go.Scatter(x=ind_focus["日期"], y=ind_focus["年增率"], mode="lines+markers",
                                               name=f"{ind} 平均", line=dict(dash="dot")))
        for fig_ in (fig_full, fig_focus):
            fig_.update_layout(
                template=plotly_template,
                hovermode="x unified", height=520,
                font=dict(size=font_size, color=TEXT_COLOR),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
            )
        fig_full.update_layout(title="📊 全期年增率趨勢")
        st.plotly_chart(fig_full, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})
        if start_y and end_y:
            fig_focus.update_layout(title=f"🔍 {start_y} ~ {end_y} 年 年增率趨勢")
            st.plotly_chart(fig_focus, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

# =========================
# 尾註
# =========================
st.caption("小提醒：年增率為『月資料』，財務比率為『季資料』；若最新季度尚未完整，圖表會暫以上一季沿用顯示。")
