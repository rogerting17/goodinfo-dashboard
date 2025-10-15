# -*- coding: utf-8 -*-
# Streamlit App：Goodinfo 年增率 + 財務比率 + 雷達圖（Render 版：移除更新，讀取線上 CSV）
# ---------------------------------------------------------------

import re, time, io, requests, copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

# =========================
# 0) 全域設定
# =========================
st.set_page_config(page_title="年增率 + 財務比率分析儀表板", layout="wide")

# === 使用者主題切換 ===
theme = st.sidebar.radio("🌗 主題模式", ["淺色", "深色"], index=0, help="切換圖表配色（不影響 Streamlit 本身佈景）")
PLOTLY_TEMPLATE = "plotly_dark" if theme == "深色" else "plotly"
PAPER_BG = "#111111" if theme == "深色" else "white"
PLOT_BG = "#111111" if theme == "深色" else "white"
FONT_COLOR = "white" if theme == "深色" else "black"

# === GitHub Raw CSV（請確認是 /main/ 而不是 /refs/heads/） ===
CSV_YOY = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E5%B9%B4%E5%A2%9E%E7%8E%87_%E6%AD%B7%E5%B9%B4%E6%AF%94%E8%BC%83_%E5%90%AB%E6%96%B0%E7%94%A2%E6%A5%AD%E5%88%86%E9%A1%9Etest1.csv"
CSV_GM  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%87%9F%E6%A5%AD%E6%AF%9B%E5%88%A9%E7%8E%87test2.csv"
CSV_OM  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%87%9F%E6%A5%AD%E5%88%A9%E7%9B%8A%E7%8E%87test2.csv"
CSV_CF  = "https://raw.githubusercontent.com/rogerting17/goodinfo-dashboard/main/Goodinfo_%E7%8F%BE%E9%87%91%E6%B5%81%E9%87%8F%E2%80%93%E7%87%9F%E6%A5%AD%E6%B4%BB%E5%8B%95%E7%8F%BE%E9%87%91%E6%B5%81%E9%87%8Ftest2.csv"

# =========================
# 小工具：穩定抓遠端 CSV（有 timeout + 錯誤處理）
# =========================
def safe_read_csv(url: str, encoding: str = "utf-8-sig", timeout: int = 12) -> pd.DataFrame:
    try:
        st.write(f"🔹 正在載入：{url}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text), encoding=encoding)
    except Exception as e:
        st.error(f"⚠️ 無法載入 {url}：{e}")
        return pd.DataFrame()

# ==================================================
# 1) 年增率：載入（寬轉長、年月->日期）
# ==================================================
@st.cache_data
def load_yoy_data(url: str) -> pd.DataFrame:
    df = safe_read_csv(url)
    if df.empty:
        return df

    df.columns = df.columns.map(lambda x: str(x).replace("\xa0"," ").replace("\u3000"," ").strip())

    # 固定處理「平均 年增率」欄位
    if "平均 年增率" in df.columns:
        df["平均年增率"] = df["平均 年增率"]
        df.drop(columns=["平均 年增率"], inplace=True)

    # ✅ 固定使用「新產業分類」欄位，改名為「產業分類」
    if "新產業分類" in df.columns:
        df.rename(columns={"新產業分類": "產業分類"}, inplace=True)
    elif "產業分類" not in df.columns:
        st.warning("❌ 找不到『新產業分類』或『產業分類』欄位，請確認 CSV 格式。")
        df["產業分類"] = np.nan  # 防呆補欄

    # 抓出所有「年增率」欄位（排除平均）
    yoy_cols = [c for c in df.columns if ("年增率" in c) and (not str(c).strip().startswith("平均"))]
    if not yoy_cols:
        st.warning("⚠️ 年增率欄位未偵測到，請確認 CSV。")
        return pd.DataFrame(columns=["代號","名稱","產業分類","期間","年增率","日期"])

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

# ==================================================
# 2) 財務比率（毛利/營益/營業金流）：穩定載入 + 連續時間序列
# ==================================================
@st.cache_data
def load_financial_ratios(csv_gm: str, csv_om: str, csv_cf: str) -> pd.DataFrame:
    gm = safe_read_csv(csv_gm)
    om = safe_read_csv(csv_om)
    cf = safe_read_csv(csv_cf)

    def melt_df_quarter(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if df.empty or "代號" not in df.columns or "名稱" not in df.columns:
            return pd.DataFrame(columns=["代號","名稱","日期",value_name])
        # 允許欄名中包含 Q 或 季（有些檔可能用中文）
        qcols = [c for c in df.columns if ("Q" in c) or ("季" in c)]
        if not qcols:
            return pd.DataFrame(columns=["代號","名稱","日期",value_name])

        d = df.melt(id_vars=["代號","名稱"], value_vars=qcols, var_name="期間", value_name=value_name)
        # 擷取 24Q3 這種格式
        d["季度"] = d["期間"].astype(str).str.extract(r"(\d{2}Q\d)")[0]
        # 轉為季末日期（Q1→03/31, Q2→06/30, Q3→09/30, Q4→12/31）
        d["日期"] = (
            d["季度"]
            .str.replace("Q1", "-03-31", regex=False)
            .str.replace("Q2", "-06-30", regex=False)
            .str.replace("Q3", "-09-30", regex=False)
            .str.replace("Q4", "-12-31", regex=False)
        )
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        return d[["代號","名稱","日期",value_name]]

    gm_m = melt_df_quarter(gm, "毛利率")
    om_m = melt_df_quarter(om, "營益率")

    # CF：欄名可能像 "24Q2 營業活動現金流量(億)"，我們抓有「營業活動」字樣者
    def melt_df_cf(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "代號" not in df.columns or "名稱" not in df.columns:
            return pd.DataFrame(columns=["代號","名稱","日期","營業金流"])
        cf_cols = [c for c in df.columns if re.search(r"\d{2}Q\d.*營業活動", str(c))]
        if not cf_cols:
            # 若抓不到，用所有 Q 欄位兜，但欄名不含營業活動時當作營業金流
            cf_cols = [c for c in df.columns if "Q" in str(c)]
        if not cf_cols:
            return pd.DataFrame(columns=["代號","名稱","日期","營業金流"])

        d = df.melt(id_vars=["代號","名稱"], value_vars=cf_cols, var_name="期間", value_name="營業金流")
        d["季度"] = d["期間"].astype(str).str.extract(r"(\d{2}Q\d)")[0]
        d["日期"] = (
            d["季度"]
            .str.replace("Q1", "-03-31", regex=False)
            .str.replace("Q2", "-06-30", regex=False)
            .str.replace("Q3", "-09-30", regex=False)
            .str.replace("Q4", "-12-31", regex=False)
        )
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        return d[["代號","名稱","日期","營業金流"]]

    cf_m = melt_df_cf(cf)

    df_fin = gm_m.merge(om_m, on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.merge(cf_m, on=["代號","名稱","日期"], how="outer")
    df_fin = df_fin.sort_values(["代號","日期"]).reset_index(drop=True)
    return df_fin

# ==================================================
# 3) Yahoo Finance：日 K 線
# ==================================================
@st.cache_data
def fetch_history_from_2019(symbol: str) -> pd.DataFrame:
    start_dt = int(datetime(2019,1,1).timestamp())
    end_dt = int(time.time())
    headers = {"User-Agent":"Mozilla/5.0"}
    def fetch_with_suffix(suffix):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{suffix}"
        params = {"period1":start_dt,"period2":end_dt,"interval":"1d",
                  "includePrePost":"false","events":"div,splits"}
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
        return resp.json()

    data = None
    for suf in (".TW",".TWO"):
        try:
            data = fetch_with_suffix(suf)
            break
        except Exception:
            continue
    if not data:
        return pd.DataFrame()

    try:
        result = data["chart"]["result"][0]
        ts = result["timestamp"]; q = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": q["open"], "High": q["high"], "Low": q["low"],
            "Close": q["close"], "Volume": q["volume"]
        }, index=pd.to_datetime(ts, unit="s"))
        df = df.dropna(subset=["Open","Close"])
        for w in (5,10,20,60,120,240):
            df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
        return df
    except Exception:
        return pd.DataFrame()

# ==================================================
# 4) UI：側邊欄（資料來源提示 + 查詢條件）
# ==================================================
st.sidebar.title("📂 查詢條件 / 控制面板")

with st.sidebar.expander("資料來源（線上 CSV）", True):
    st.caption(f"年增率：{CSV_YOY}")
    st.caption(f"毛利率：{CSV_GM}")
    st.caption(f"營益率：{CSV_OM}")
    st.caption(f"現金流量：{CSV_CF}")

# 📦 載入資料
st.info("📦 載入年增率資料中…")
df_yoy = load_yoy_data(CSV_YOY)
st.info("📦 載入財務比率資料中…")
df_fin = load_financial_ratios(CSV_GM, CSV_OM, CSV_CF)

# ================================================
# 📊 主查詢區域
# ================================================
inds = sorted(df_yoy['產業分類'].dropna().unique()) if not df_yoy.empty else []
sel_inds = st.sidebar.multiselect("選擇產業分類（可多選）", inds)
manual_input = st.sidebar.text_input("或輸入股票代號（逗號分隔）", "2330,1101")
manual_codes = [c.strip() for c in manual_input.split(',') if c.strip()]

filtered = df_yoy.copy()
if not df_yoy.empty and sel_inds:
    filtered = filtered[filtered['產業分類'].isin(sel_inds)]
if not df_yoy.empty and manual_codes:
    filtered = pd.concat([filtered, df_yoy[df_yoy['代號'].isin(manual_codes)]], ignore_index=True)

stocks = filtered[['代號', '名稱']].drop_duplicates() if not filtered.empty else pd.DataFrame(columns=["代號","名稱"])
opts = {f"{r['代號']} {r['名稱']}": r['代號'] for _, r in stocks.iterrows()}

default_keys = list(opts.keys())[:1] if len(opts) else []
selected = st.sidebar.multiselect("選擇股票", list(opts.keys()), default=default_keys)

show_yoy = st.sidebar.checkbox("📈 顯示月營收年增率（含產業平均）", True)
show_kline = st.sidebar.checkbox("🕯️ 顯示 K 線 + 均線（Yahoo）", True)
show_fin = st.sidebar.checkbox("📊 顯示財務比率（毛利/營益/金流）", True)
show_radar = st.sidebar.checkbox("🧭 年度雷達圖（毛利/營益/金流）", True)
show_radar_mix = st.sidebar.checkbox("🧭 綜合雷達圖（毛利/營益/金流/營收年增率）", True)
normalize_radar = st.sidebar.checkbox("⚖️ 雷達圖正規化 (0-100)", True)

st.markdown("## 年增率 + K 線 + 財務比率 儀表板")

# ==================================================
# 5) 單一股票：年增率 + 產業平均 + K 線
# ==================================================
def normalize_values(values, all_data):
    if not normalize_radar: return values
    scaled = []
    for i, v in enumerate(values):
        col = all_data[i].dropna()
        if len(col) == 0: scaled.append(0); continue
        if col.max() == col.min(): scaled.append(50); continue
        scaled.append((v - col.min()) / (col.max() - col.min()) * 100)
    return scaled

if len(selected) == 1 and df_yoy is not None and not df_yoy.empty:
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
            fig_k = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{}]]
            )
            fig_k.add_trace(
                go.Candlestick(
                    x=df_yf.index, open=df_yf["Open"], high=df_yf["High"],
                    low=df_yf["Low"], close=df_yf["Close"], name="K 線",
                    increasing_line_color='red', decreasing_line_color='green'
                ),
                row=1, col=1, secondary_y=False
            )
            for w in (5, 10, 20, 60, 120, 240):
                fig_k.add_trace(
                    go.Scatter(x=df_yf.index, y=df_yf[f"MA{w}"], mode="lines", name=f"MA{w}"),
                    row=1, col=1, secondary_y=False
                )
            if show_yoy and not yoy_s.empty:
                fig_k.add_trace(
                    go.Scatter(
                        x=yoy_s["日期"], y=yoy_s["年增率"], mode="lines+markers",
                        name=f"{code} 年增率", line=dict(dash="dot")
                    ),
                    row=1, col=1, secondary_y=True
                )
            if show_yoy and not ind_avg.empty:
                fig_k.add_trace(
                    go.Scatter(
                        x=ind_avg["日期"], y=ind_avg["年增率"], mode="lines+markers",
                        name=f"{industry} 平均年增率", line=dict(dash="dash")
                    ),
                    row=1, col=1, secondary_y=True
                )
            fig_k.add_trace(
                go.Bar(x=df_yf.index, y=df_yf["Volume"], marker_color=vol_colors,
                       name="成交量", showlegend=False),
                row=2, col=1
            )
            fig_k.update_layout(
                title=f"🕯️ {code}.TW K 線 + 均線 + 成交量 + 月營收年增率 (含產業平均)",
                hovermode="x unified", height=760, dragmode="pan",
                xaxis=dict(rangeslider=dict(visible=True), type="date"),
                yaxis=dict(title="股價"),
                yaxis2=dict(title="月營收年增率 (%)", overlaying="y", side="right", showgrid=False),
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR)
            )
            st.plotly_chart(fig_k, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

# ==================================================
# 6) 財務比率（單股） + 年度雷達圖 + 綜合雷達圖
# ==================================================
if (show_fin or show_radar or show_radar_mix) and len(selected) == 1 and not df_fin.empty:
    code = opts[selected[0]]
    fin_s = df_fin[df_fin["代號"] == code].sort_values("日期")
    yoy_s = df_yoy[df_yoy["代號"] == code].sort_values("日期") if not df_yoy.empty else pd.DataFrame()

    if show_fin and not fin_s.empty:
        st.markdown("### 📊 財務比率趨勢")

        # === 折線圖：毛利率 / 營益率（connectgaps=True） ===
        ratio_cols = [c for c in ["毛利率", "營益率"] if c in fin_s.columns]
        if ratio_cols:
            df_ratio = fin_s[["日期"] + ratio_cols].copy()
            df_plot = df_ratio.melt(id_vars="日期", var_name="指標", value_name="數值")
            df_plot["數值"] = pd.to_numeric(df_plot["數值"], errors="coerce")
            df_plot = df_plot.sort_values("日期")

            fig = px.line(
                df_plot, x="日期", y="數值", color="指標",
                title=f"{code} 毛利率 / 營益率", markers=True, template=PLOTLY_TEMPLATE
            )
            fig.update_traces(connectgaps=True)
            fig.update_layout(height=500, legend_title_text="指標",
                              paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

        # === 現金流量柱狀圖 ===
        if "營業金流" in fin_s.columns and not fin_s["營業金流"].dropna().empty:
            df_cf = fin_s[["日期", "營業金流"]].dropna().sort_values("日期")
            fig_cf = px.bar(df_cf, x="日期", y="營業金流", title=f"{code} 營業活動現金流量（億）", template=PLOTLY_TEMPLATE)
            fig_cf.update_layout(height=400, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR))
            st.plotly_chart(fig_cf, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

    # === 年份清單 ===
    years_fin = sorted(fin_s["日期"].dt.year.dropna().unique().tolist()) if not fin_s.empty else []
    years_yoy = sorted(yoy_s["日期"].dt.year.dropna().unique().tolist()) if not yoy_s.empty else []
    all_years = sorted(set(years_fin) | set(years_yoy))
    default_years = all_years[-1:] if all_years else []

    # ---- 年度雷達圖 ----
    def normalize_values(values, all_data_series):
        if not normalize_radar:
            return values
        scaled = []
        for i, v in enumerate(values):
            s = pd.to_numeric(all_data_series[i], errors="coerce").dropna()
            if s.empty:
                scaled.append(0); continue
            lo, hi = s.min(), s.max()
            if hi == lo:
                scaled.append(50); continue
            scaled.append((v - lo) / (hi - lo) * 100)
        return scaled

    if show_radar:
        st.markdown("### 🧭 年度雷達圖（毛利率 / 營益率 / 營業金流）")
        st.caption("💡 快速提示：雙擊圖中心可快速恢復原始大小")
        chosen_years = st.multiselect("選擇年份（財務比率雷達圖）", all_years, default=default_years, key="radar_fin_years")
        if chosen_years:
            categories = ["毛利率","營益率","營業金流"]
            all_data = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"]]
            fig_radar = go.Figure(); colors = px.colors.qualitative.Bold
            for i, yr in enumerate(chosen_years):
                color = colors[i % len(colors)]
                yr_df = fin_s[fin_s["日期"].dt.year == yr].sort_values("日期").tail(1)
                values = [
                    float(pd.to_numeric(yr_df["毛利率"]).iloc[0]) if not yr_df.empty and "毛利率" in yr_df else 0,
                    float(pd.to_numeric(yr_df["營益率"]).iloc[0]) if not yr_df.empty and "營益率" in yr_df else 0,
                    float(pd.to_numeric(yr_df["營業金流"]).iloc[0]) if not yr_df.empty and "營業金流" in yr_df else 0,
                ]
                scaled = normalize_values(values, all_data)
                fig_radar.add_trace(go.Scatterpolar(
                    r=scaled, theta=categories, fill="toself", name=str(yr),
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("rgb","rgba").replace(")",",0.3)"),
                    mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in values], textfont=dict(color=FONT_COLOR),
                    textposition="top center"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if normalize_radar else None)),
                title=f"{code} 財務比率雷達圖（年度比較）", showlegend=True,
                template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR)
            )
            initial_state = copy.deepcopy(fig_radar.layout)
            fig_radar.update_layout(
                updatemenus=[dict(type="buttons", showactive=False, x=1.05, y=1.15,
                                  buttons=[dict(label="Reset View", method="relayout", args=[initial_state])])]
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displaylogo": False})

    # ---- 綜合雷達圖 ----
    if show_radar_mix:
        st.markdown("### 🧭 綜合雷達圖（毛利率 / 營益率 / 營業金流 / 營收年增率）")
        st.caption("💡 快速提示：雙擊圖中心可快速恢復原始大小")
        chosen_years2 = st.multiselect("選擇年份（綜合雷達圖）", all_years, default=default_years, key="radar_mix_years")
        if chosen_years2:
            categories_all = ["毛利率","營益率","營業金流","月營收年增率"]
            all_data_all = [fin_s["毛利率"], fin_s["營益率"], fin_s["營業金流"], yoy_s["年增率"] if not yoy_s.empty else pd.Series(dtype=float)]
            fig_radar_all = go.Figure(); colors = px.colors.qualitative.Dark24
            for i, yr in enumerate(chosen_years2):
                color = colors[i % len(colors)]
                latest_fin = fin_s[fin_s["日期"].dt.year == yr].sort_values("日期").tail(1)
                latest_yoy = yoy_s[yoy_s["日期"].dt.year == yr].sort_values("日期").tail(1) if not yoy_s.empty else pd.DataFrame()
                values = [
                    float(pd.to_numeric(latest_fin["毛利率"]).iloc[0]) if not latest_fin.empty and "毛利率" in latest_fin else 0,
                    float(pd.to_numeric(latest_fin["營益率"]).iloc[0]) if not latest_fin.empty and "營益率" in latest_fin else 0,
                    float(pd.to_numeric(latest_fin["營業金流"]).iloc[0]) if not latest_fin.empty and "營業金流" in latest_fin else 0,
                    float(pd.to_numeric(latest_yoy["年增率"]).iloc[0]) if not latest_yoy.empty and "年增率" in latest_yoy else 0,
                ]
                scaled = normalize_values(values, all_data_all)
                fig_radar_all.add_trace(go.Scatterpolar(
                    r=scaled, theta=categories_all, fill="toself", name=str(yr),
                    line=dict(color=color, width=2),
                    fillcolor=color.replace("rgb","rgba").replace(")",",0.3)"),
                    mode="lines+markers+text",
                    text=[f"{v:.1f}" for v in values], textfont=dict(color=FONT_COLOR),
                    textposition="top center"
                ))
            fig_radar_all.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100] if normalize_radar else None)),
                title=f"{code} 綜合財務 + 營收雷達圖（年度比較）", showlegend=True,
                template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR)
            )
            initial_state = copy.deepcopy(fig_radar_all.layout)
            fig_radar_all.update_layout(
                updatemenus=[dict(type="buttons", showactive=False, x=1.05, y=1.15,
                                  buttons=[dict(label="Reset View", method="relayout", args=[initial_state])])]
            )
            st.plotly_chart(fig_radar_all, use_container_width=True, config={"displaylogo": False})

# ==================================================
# 7) 平均年增率排行榜
# ==================================================
with st.expander("🏆 平均年增率排行榜 Top 10", True):
    try:
        df_raw = safe_read_csv(CSV_YOY)
        if not df_raw.empty:
            df_raw.columns = df_raw.columns.map(lambda x: str(x).replace("\xa0"," ").replace("\u3000"," ").strip())
            avg_col = next((c for c in df_raw.columns if c.strip() in ("平均年增率","平均 年增率")), None)
            if avg_col:
                df_avg = df_raw[["代號","名稱", avg_col]].rename(columns={avg_col:"平均年增率"}).dropna()
                df_avg["平均年增率"] = pd.to_numeric(df_avg["平均年增率"], errors="coerce")
                rank_df = df_avg.sort_values("平均年增率", ascending=False).head(10).reset_index(drop=True)
                st.dataframe(rank_df.style.format({"平均年增率": "{:.2f}%"}), use_container_width=True)
            else:
                st.info("原始檔未包含『平均年增率』欄位")
        else:
            st.info("年增率檔案載入失敗，無法產生排行榜")
    except Exception as e:
        st.warning(f"排行榜生成失敗：{e}")

# ==================================================
# 8) 近三個月年增率連續成長
# ==================================================
with st.expander("📈 近三個月年增率連續成長", True):
    if df_yoy.empty:
        st.info("尚未載入年增率資料")
    else:
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
                            "代號":sid,"名稱":d.iloc[0]["名稱"],"產業分類":d.iloc[0]["產業分類"],
                            "月份1":pd.to_datetime(d.iloc[0]["日期"]).strftime("%Y-%m"),"年增率1":round(float(y1),2),
                            "月份2":pd.to_datetime(d.iloc[1]["日期"]).strftime("%Y-%m"),"年增率2":round(float(y2),2),
                            "月份3":pd.to_datetime(d.iloc[2]["日期"]).strftime("%Y-%m"),"年增率3":round(float(y3),2)
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

# ==================================================
# 9) 多檔股票 vs 產業平均 年增率趨勢
# ==================================================
with st.expander("🧯 多檔股票與產業平均的年增率趨勢", False):
    if df_yoy.empty:
        st.info("尚未載入年增率資料")
    else:
        all_years_yoy = sorted(df_yoy["日期"].dt.year.dropna().unique())
        if len(all_years_yoy) >= 1:
            start_y = st.selectbox("聚焦起始年", all_years_yoy, index=0, key="focus_start")
            end_y = st.selectbox("聚焦結束年", all_years_yoy, index=len(all_years_yoy)-1, key="focus_end")
        else:
            start_y, end_y = None, None

        sel_multi = st.multiselect("選擇多檔股票", list(opts.keys()), default=list(opts.keys())[:2] if len(opts)>=2 else list(opts.keys()))
        if sel_multi:
            fig_full = go.Figure()
            fig_focus = go.Figure()
            for sk in sel_multi:
                sid = opts[sk]
                s = df_yoy[df_yoy["代號"] == sid].sort_values("日期")
                if s.empty: 
                    continue
                ind = s["產業分類"].iloc[0] if "產業分類" in s.columns and not s["產業分類"].isna().all() else "未知產業"
                ind_avg2 = df_yoy[df_yoy["產業分類"] == ind].groupby("日期")["年增率"].mean().reset_index() if ind != "未知產業" else pd.DataFrame(columns=["日期","年增率"])

                fig_full.add_trace(go.Scatter(x=s["日期"], y=s["年增率"], mode="lines+markers", name=f"{sid}"))
                if not ind_avg2.empty:
                    fig_full.add_trace(go.Scatter(x=ind_avg2["日期"], y=ind_avg2["年增率"], mode="lines+markers",
                                                  name=f"{ind} 平均", line=dict(dash="dot")))
                if start_y and end_y:
                    focus = s[(s["日期"].dt.year >= start_y) & (s["日期"].dt.year <= end_y)]
                    ind_focus = ind_avg2[(ind_avg2["日期"].dt.year >= start_y) & (ind_avg2["日期"].dt.year <= end_y)] if not ind_avg2.empty else pd.DataFrame(columns=["日期","年增率"])
                    fig_focus.add_trace(go.Scatter(x=focus["日期"], y=focus["年增率"], mode="lines+markers", name=f"{sid}"))
                    if not ind_focus.empty:
                        fig_focus.add_trace(go.Scatter(x=ind_focus["日期"], y=ind_focus["年增率"], mode="lines+markers",
                                                       name=f"{ind} 平均", line=dict(dash="dot")))
            fig_full.update_layout(title="📊 全期年增率趨勢", hovermode="x unified", height=520,
                                   template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR))
            st.plotly_chart(fig_full, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})
            if start_y and end_y:
                fig_focus.update_layout(title=f"🔍 {start_y} ~ {end_y} 年 年增率趨勢", hovermode="x unified", height=520,
                                        template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_COLOR))
                st.plotly_chart(fig_focus, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]})

# =========================
# 尾註
# =========================
st.caption("小提醒：年增率為『月資料』，財務比率為『季資料』。")
