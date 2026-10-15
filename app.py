"""
股票系統｜手機決策台（Streamlit）
只要輸入股票代碼 → 自動抓歷史日K → 套用股票系統規則 → 輸出可進場／等待／不可進場與回測勝率。
"""
import re

import pandas as pd
import streamlit as st

import importlib

import engine as E
import market_chips as MC
import report_view as RV
import sectors as SEC

# Streamlit Cloud 更新時可能混用新舊檔案；每次重新載入本專案模組，確保同一版本。
for _m in (E, MC, SEC, RV):
    importlib.reload(_m)

st.set_page_config(page_title="股票系統", page_icon="📈", layout="centered")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:760px}
.verdict{border-radius:16px;padding:18px 18px 14px;margin:6px 0 12px;color:#fff}
.verdict .big{font-size:34px;font-weight:800;line-height:1.1}
.verdict .sub{font-size:14px;opacity:.92;margin-top:6px;line-height:1.5}
.v-green{background:#1f8a5b}.v-orange{background:#b7791f}.v-red{background:#c0392b}
.kpi{border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:10px 12px;margin-bottom:8px}
.kpi .lab{font-size:12px;opacity:.75}.kpi .val{font-size:24px;font-weight:700}
.kpi .note{font-size:12px;opacity:.75}
.small{font-size:12px;opacity:.75;line-height:1.6}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(code):
    return E.load_prices(code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_market():
    """回傳 (加權指數 5 年收盤序列, 大盤環境 dict)。"""
    idx = E.load_index(5)
    return idx, E.market_env(idx)


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector(code):
    """族群強弱用：同業收盤價 → 族群指數（測試中）。回傳 (族群指數或 None, 族群名稱, 同業檔數)。"""
    name, peers = SEC.peers_of(code)
    if not peers:
        return None, "", 0
    closes = E.fetch_closes(peers)
    return E.sector_index(list(closes.values())), name, len(closes)


@st.cache_data(ttl=1800, show_spinner=False)
def get_spot_chips():
    return MC.spot_institutions(10)


@st.cache_data(ttl=1800, show_spinner=False)
def get_fut_chips():
    df, src, errs = MC.futures_institutions(10)
    px, perr = MC.futures_close(10)
    return df, src, errs + ([perr] if perr else []), px


@st.cache_data(ttl=1800, show_spinner=False)
def get_taiex():
    return MC.taiex_close()


def tw_color(v):
    """台股慣例：正數紅、負數綠。"""
    try:
        return "color:#e53935" if v > 0 else "color:#2e9d5b" if v < 0 else ""
    except TypeError:
        return ""


def chip_chart(df, cols, unit, line_s, line_name):
    try:
        _chip_chart(df, cols, unit, line_s, line_name)
    except Exception:  # noqa: BLE001
        st.bar_chart(df.set_index("日期")[cols])


def _chip_chart(df, cols, unit, line_s, line_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = df["日期"].dt.strftime("%m/%d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=df["合計"], name="合計", marker_color="rgba(56,140,230,.75)"))
    for c, clr in zip(cols, ("#e53935", "#2e9d5b", "#f39c12")):
        fig.add_trace(go.Scatter(x=x, y=df[c], name=c, mode="lines+markers", line=dict(color=clr, width=2)))
    if line_s is not None and len(line_s):
        ys = [line_s.get(d) for d in df["日期"]]
        fig.add_trace(go.Scatter(x=x, y=ys, name=line_name, mode="lines+markers",
                                 line=dict(color="#f1c40f", width=2, dash="dot")), secondary_y=True)
    fig.update_layout(height=330, margin=dict(l=4, r=4, t=8, b=4), dragmode=False,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)))
    fig.update_yaxes(title_text=unit, secondary_y=False, zeroline=True)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chip_table(df, cols, line_s, line_name, fmt):
    t = df[["日期"] + cols + ["合計"]].copy()
    if line_s is not None and len(line_s):
        t[line_name] = [line_s.get(d) for d in t["日期"]]
    t = t.iloc[::-1]
    t["日期"] = t["日期"].dt.strftime("%m/%d")
    sty = (t.style.format({c: fmt for c in cols + ["合計"]} | ({line_name: "{:,.0f}"} if line_name in t else {}),
                          na_rep="—")
           .map(tw_color, subset=cols + ["合計"]))
    st.dataframe(sty, hide_index=True, use_container_width=True)


def render_chips():
    st.markdown("#### 大盤籌碼（官方盤後資料）")
    st.caption("證交所、期交所每天約 15:00 後公布；30 分鐘自動重新抓取。是盤後資料，不是盤中即時。")
    with st.spinner("抓取三大法人資料…"):
        spot, latest, s_err = get_spot_chips()
        fut, f_src, f_err, fut_px = get_fut_chips()
        taiex = get_taiex()

    # ---- 外資今昨比較 ----
    m1, m2 = st.columns(2)
    if spot is not None and len(spot) >= 2:
        a, b = spot.iloc[-1], spot.iloc[-2]
        m1.metric(f"外資現貨買賣超（{a['日期']:%m/%d}）", f"{a['外資']:,.1f} 億",
                  f"{a['外資'] - b['外資']:+,.1f} 億 vs 前日", delta_color="inverse")
    if fut is not None and len(fut) >= 2:
        a, b = fut.iloc[-1], fut.iloc[-2]
        m2.metric(f"外資台指期淨未平倉（{a['日期']:%m/%d}）", f"{a['外資']:,.0f} 口",
                  f"{a['外資'] - b['外資']:+,.0f} 口 vs 前日", delta_color="inverse")
    st.caption("數字為負＝賣超／淨空單；台股慣例紅色為正、綠色為負。")

    t1, t2, t3 = st.tabs(["現貨三大法人", "期貨三大法人", "當日買賣金額"])
    with t1:
        if spot is None:
            st.warning("無法取得證交所三大法人資料。")
            st.caption("；".join(s_err[:3]))
        else:
            chip_chart(spot, ["外資", "投信", "自營"], "億元", taiex, "加權指數")
            chip_table(spot, ["外資", "投信", "自營"], taiex, "加權指數", "{:,.2f}")
            st.caption("來源：臺灣證券交易所 BFI82U（上市）。外資含外資自營商；自營含自行買賣與避險。單位：億元。")
    with t2:
        if fut is None:
            st.warning("無法取得期交所臺股期貨三大法人資料。")
            st.caption("；".join(f_err[:3]))
        else:
            use_px, nm = (fut_px, "期貨收盤") if fut_px is not None and len(fut_px) else (taiex, "加權指數")
            chip_chart(fut, ["外資", "投信", "自營"], "口", use_px, nm)
            chip_table(fut, ["外資", "投信", "自營"], use_px, nm, "{:,.0f}")
            st.caption(f"來源：{f_src}。臺股期貨（TX）未平倉多空淨額，單位：口。"
                       + ("期貨收盤價取得失敗，改顯示加權指數。" if nm == "加權指數" else "期貨收盤為近月一般盤。"))
    with t3:
        if latest is None:
            st.warning("無法取得當日三大法人買賣金額。")
        else:
            d, tb = latest
            st.markdown(f"**{d:%Y/%m/%d} 三大法人買賣金額統計（上市）**")
            sty = (tb.style.format({"買進金額": "{:,.0f}", "賣出金額": "{:,.0f}", "買賣差額": "{:,.0f}"})
                   .map(tw_color, subset=["買賣差額"]))
            st.dataframe(sty, hide_index=True, use_container_width=True)
            st.caption("單位：元。來源：臺灣證券交易所。")


def money(x):
    return f"{x:,.0f}"


def price(x):
    return f"{x:,.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------
st.markdown("### 📈 股票系統｜短線決策")
st.caption("5～15 交易日・淨利 2～3%・風控優先")

with st.form("q"):
    code = st.text_input("股票代碼", placeholder="例如 2059、2330、00981A", max_chars=7)
    with st.expander("進階設定（通常不用改）"):
        c1, c2 = st.columns(2)
        capital = c1.number_input("本次投入上限（元）", 1000, 10_000_000, 50_000, 1000)
        max_loss = c2.number_input("單筆最大損失（元）", 100, 1_000_000, 1_500, 100)
        target = c1.selectbox("淨獲利目標", [3.0, 2.0], format_func=lambda x: f"{x:g}%")
        hold = c2.selectbox("最長持有（交易日）", [15, 10, 5])
        discount = c1.number_input("手續費折扣（6 折填 0.6）", 0.1, 1.0, 1.0, 0.05)
        min_rr = c2.number_input("最低風險報酬比", 0.5, 3.0, 1.0, 0.1)
        st.markdown("**三項過濾（測試中）**")
        f1 = st.checkbox("停損距離至少 1 倍 ATR（太近不進場）", value=E.DEFAULT_PARAMS["f_stop_atr"])
        f2 = st.checkbox("大盤空頭時股數減半", value=E.DEFAULT_PARAMS["f_market"])
        f3 = st.checkbox("族群明顯弱於大盤時等待", value=E.DEFAULT_PARAMS["f_sector"])
        csv = st.file_uploader("自動抓不到資料時，可上傳 CSV（Date,Open,High,Low,Close,Volume；量單位：張）",
                               type=["csv"])
    go = st.form_submit_button("分析", type="primary", use_container_width=True)

if not go:
    st.markdown('<div class="small">輸入代碼後按「分析」。資料為收盤後日K，盤中為延遲且成交量不完整，'
                '建議收盤後判斷。本工具是研究用途，不是下單指令。</div>', unsafe_allow_html=True)
    st.stop()

code = (code or "").strip().upper()
if not re.fullmatch(r"\d{4,6}[A-Z]?", code):
    st.error("請輸入正確的台股代碼，例如 2059 或 00981A。")
    st.stop()

params = {"capital": capital, "max_loss": max_loss, "target_net": target, "hold_days": hold,
          "fee_discount": discount, "min_rr": min_rr, "f_stop_atr": f1, "f_market": f2, "f_sector": f3}

with st.spinner("抓取歷史股價與計算中…"):
    if csv is not None:
        try:
            df, src, errs = E.parse_csv(csv), "使用者上傳 CSV", []
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV 讀取失敗：{e}")
            st.stop()
    else:
        df, src, errs = get_prices(code)
    if df is None:
        st.error("抓不到這檔股票的歷史資料。")
        st.caption("；".join(errs))
        st.info("請確認代碼，或打開「進階設定」上傳 CSV。")
        st.stop()
    mkt, market = get_market()
    sec, sec_name, _ = get_sector(code)
    try:
        an = E.Analyzer(df, code, params, mkt=mkt, sec=sec, sec_name=sec_name)
        r = an.report(market)
    except Exception as e:  # noqa: BLE001
        st.error(f"分析失敗：{e}")
        st.stop()

# ---------------- 單檔報告（與產業掃描共用 report_view.py）----------------
RV.render_report(r, an, df, src, target, hold, title=code)

# ---------------- 大盤籌碼 ----------------
try:
    render_chips()
except Exception as e:  # noqa: BLE001
    st.warning(f"大盤籌碼區塊載入失敗：{str(e)[:120]}")

st.markdown(f'<div class="small">引擎 {E.ENGINE_VERSION}・族群強弱以 sectors.py 自訂分類計算（測試中）；未納入進場判斷：個股法人籌碼、財報事件、週K；下方大盤籌碼僅供對照。'
            '研究用途，不是投資建議或下單指令。</div>', unsafe_allow_html=True)
