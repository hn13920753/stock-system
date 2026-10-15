"""
股票系統｜分析引擎
------------------------------------------------------------
小朱老師 AI 工作系統：短線 5～15 交易日、淨利 2～3%、風控優先。

本檔只做計算，不含畫面。所有門檻集中在 TH，屬「測試中 V0.1」，
之後依回測與實際交易結果修正。
"""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np
import pandas as pd

ENGINE_VERSION = "V1.3（量化門檻 V0.3・測試中）"

# ------------------------------------------------------------------
# 量化門檻（測試中 V0.1）
# ------------------------------------------------------------------
TH = {
    "vol_burst": 2.0,     # 爆大量：成交量 ≥ 前 5 日均量 × 2
    "vol_big": 1.5,       # 大量：成交量 ≥ 前 5 日均量 × 1.5
    "long_red": 3.0,      # 長紅 K：漲幅 ≥ 3%
    "body": 0.5,          # 長紅 K：實體 ≥ 當日振幅 50%
    "pivot": 3,           # 波段高低點：左右各 3 根確認（確認後才使用，避免偷看未來）
    "low_near": 0.05,     # 低檔：近 10 日最低價距 60 日最低 5% 以內
    "low_drop": 0.15,     # 低檔：60 日內高低差 ≥ 15%（確實跌過）
    "squeeze": 0.03,      # 均線糾結：MA5/10/20 差距 ≤ 3%
    "small_k": 2.0,       # 碎步：單日漲跌幅 ≤ 2%
    "retest_days": 10,    # 突破後 10 日內的回測買點
    "retest_hold": 0.97,  # 回測不可跌破突破價 3% 以上
    "limit": 9.5,         # 漲跌停判定（%）
    # ---- V1.2 新增 ----
    "atr_min": 1.2,       # 股性：ATR14 占股價 % 下限（太小 → 15 日內難達 3%）
    "atr_max": 4.5,       # 股性：ATR14 占股價 % 上限（太大 → 停損距離過大）
    "limit_buffer": 0.005,  # 拉回限價＝突破價 × 1.005
    "limit_valid": 3,     # 限價單有效交易日數
    "stop_atr": 0.5,      # 拉回限價的停損＝突破價 − 0.5 × ATR（跌回突破價下＝假突破）
    "no_chase": 0.01,     # 收盤型買點：隔日開盤高於收盤 1% 以上不追
    "b_pull_min": 0.03,   # 策略B：自近 10 日高點回檔至少 3%
    "b_pull_max": 0.12,   # 策略B：回檔不超過 12%
    "b_touch": 0.015,     # 策略B：回檔低點需觸及 MA10/MA20 的 1.5% 以內
    "b_vr": 1.2,          # 策略B：觸發紅 K 量比 ≥ 1.2
    # ---- V1.3 新增（三項過濾，測試中）----
    "stop_atr_min": 1.0,  # 停損距離（進場價－停損）至少 1 × ATR14，太近容易被正常波動洗出
    "rs_days": 20,        # 族群強弱：比較近 20 個交易日報酬
    "rs_weak": -3.0,      # 族群 20 日報酬比大盤落後超過 3 個百分點 → 弱勢族群，降為等待
    "rs_min_peers": 3,    # 族群指數至少要有 3 檔同業資料
}

# 三項過濾的狀態（規則管理：正式採用／測試中／封存）
# 2026-10-15 回測（research/filters.txt，全產業 1,387 筆）：
#   ① 先到停利 32.9% → 40.1%、平均 −0.16% → −0.05%，前後兩段方向一致 → 預設開
#   ② 只調整部位大小、不改勝率；空頭時訊號平均較差但後段樣本少 → 當風控用，預設開
#   ③ 落後大盤的族群訊號反而沒有比較差（兩段皆然）→ 證據不支持，預設關（保留可開）
FILTERS = {
    "f_stop_atr": ("停損 ≥ 1 ATR", "測試中・回測支持"),
    "f_market": ("大盤濾網（空頭時股數減半）", "測試中・風控用"),
    "f_sector": ("族群強弱（明顯弱於大盤時等待）", "測試中・回測不支持，預設關"),
}

PATTERNS = {
    1: "低檔大量長紅K", 2: "破切反彈過高", 3: "大量雙腳反轉", 4: "缺口之上續漲",
    5: "碎步上漲攻擊", 6: "底部洗盤上攻", 7: "空轉多過空高", 8: "紅黑紅",
    9: "突破大量黑K", 10: "低檔連2日大量突破", 11: "低檔大量長下影",
    12: "空頭反彈月線上盤整", 13: "雙盤底大量紅K突破", 14: "雙弧底突破",
    15: "均線糾結紅K突破", 16: "突破ABC", 17: "島型反轉", 18: "突破上升軌道",
}
MANUAL_ONLY = {13, 14, 16}  # 尚未自動化，需人工看圖確認

DEFAULT_PARAMS = {
    "target_net": 3.0,      # 淨獲利目標 %
    "hold_days": 15,        # 最長持有交易日
    "fee_rate": 0.001425,   # 券商手續費率
    "fee_discount": 1.0,    # 手續費折扣（1.0 = 無折扣，0.6 = 6 折）
    "capital": 50000,       # 本次投入上限（元）
    "max_loss": 1500,       # 單筆最大可接受損失（元）
    "min_rr": 1.0,          # 最低風險報酬比
    "f_stop_atr": True,     # 過濾①：停損距離 < 1 ATR 不進場（測試中）
    "f_market": True,       # 過濾②：加權指數空頭時股數減半（測試中）
    "f_sector": False,      # 過濾③：族群明顯弱於大盤時降為等待（測試中；回測不支持，預設關）
}


# ------------------------------------------------------------------
# 交易成本與價格跳動單位
# ------------------------------------------------------------------
def is_etf(code: str) -> bool:
    return str(code).startswith("00")


def tick_size(p: float, etf: bool = False) -> float:
    if etf:
        return 0.01 if p < 50 else 0.05
    for lim, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if p < lim:
            return t
    return 5.0


def round_up(p: float, etf: bool = False) -> float:
    t = tick_size(p, etf)
    return round(math.ceil(p / t - 1e-9) * t, 2)


def round_down(p: float, etf: bool = False) -> float:
    t = tick_size(p, etf)
    return round(math.floor(p / t + 1e-9) * t, 2)


def cost_rates(code: str, params: dict) -> tuple[float, float]:
    """回傳 (買進成本率, 賣出成本率)。股票證交稅 0.3%，ETF 0.1%。"""
    fee = params["fee_rate"] * params["fee_discount"]
    tax = 0.001 if is_etf(code) else 0.003
    return fee, fee + tax


def target_price(entry: float, net_pct: float, buy: float, sell: float, etf: bool) -> float:
    raw = entry * (1 + buy) * (1 + net_pct / 100) / (1 - sell)
    return round_up(raw, etf)


def net_return(entry: float, exit_: float, buy: float, sell: float) -> float:
    return (exit_ * (1 - sell)) / (entry * (1 + buy)) * 100 - 100


# ------------------------------------------------------------------
# 資料來源：Yahoo → 證交所 → CSV
# ------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.dropna()
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"])]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_yahoo(code: str, years: int = 5):
    import yfinance as yf

    for suf in (".TW", ".TWO"):
        try:
            h = yf.Ticker(code + suf).history(period=f"{years}y", interval="1d",
                                              auto_adjust=False, actions=False)
        except Exception:
            continue
        if h is None or len(h) < 80:
            continue
        h = h.rename(columns=str.lower)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        h["volume"] = h["volume"] / 1000.0  # 股 → 張
        h = h[h["volume"] > 0]               # 去掉未成交日
        return _clean(h), f"Yahoo Finance（{code}{suf}）"
    return None, None


def fetch_twse(code: str, months: int = 36):
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    rows, misses = [], 0
    today = dt.date.today()
    for k in range(months):
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        url = ("https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json"
               f"&date={y}{m:02d}01&stockNo={code}")
        try:
            j = s.get(url, timeout=10).json()
        except Exception:
            j = {}
        if j.get("stat") != "OK" or not j.get("data"):
            misses += 1
            if k < 3 and misses >= 3:
                break
            continue
        for a in j["data"]:
            try:
                yy, mm, dd = a[0].split("/")
                num = lambda x: float(str(x).replace(",", ""))
                rows.append({
                    "date": pd.Timestamp(int(yy) + 1911, int(mm), int(dd)),
                    "open": num(a[3]), "high": num(a[4]), "low": num(a[5]),
                    "close": num(a[6]), "volume": num(a[1]) / 1000.0,
                })
            except Exception:
                continue
        time.sleep(0.4)
    if len(rows) < 80:
        return None, None
    df = pd.DataFrame(rows).set_index("date")
    return _clean(df), f"臺灣證交所 STOCK_DAY（{code}）"


def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["date", "open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError("CSV 缺少欄位：" + "、".join(miss))
    for c in need[1:]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("/", "-"), errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    return _clean(df)


def load_prices(code: str):
    """依序嘗試資料源，回傳 (df, 來源說明, 錯誤訊息列表)。"""
    errors = []
    try:
        df, src = fetch_yahoo(code)
        if df is not None:
            return df, src, errors
        errors.append("Yahoo Finance 無資料")
    except Exception as e:  # noqa: BLE001
        errors.append(f"Yahoo Finance 失敗：{e}")
    try:
        df, src = fetch_twse(code)
        if df is not None:
            return df, src, errors
        errors.append("證交所無資料（上櫃股票不在證交所）")
    except Exception as e:  # noqa: BLE001
        errors.append(f"證交所失敗：{e}")
    return None, None, errors


def load_index(years: int = 5):
    """加權指數日收盤（Series，索引為日期）。失敗回傳 None。"""
    try:
        import yfinance as yf
        h = yf.Ticker("^TWII").history(period=f"{years}y", interval="1d", auto_adjust=False)
        if h is None or len(h) < 60:
            return None
        c = h["Close"].astype(float)
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c[~c.index.duplicated(keep="last")].sort_index()
    except Exception:
        return None


def market_states(c: pd.Series) -> pd.Series:
    """每日大盤狀態分數：1 多頭（收盤 > MA20 > MA60 且 MA20 上彎）、0 空頭（收盤 < MA20 < MA60）、0.5 盤整。"""
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    slope = ma20 - ma20.shift(5)
    s = pd.Series(0.5, index=c.index)
    s[(c > ma20) & (ma20 > ma60) & (slope > 0)] = 1.0
    s[(c < ma20) & (ma20 < ma60)] = 0.0
    s[ma60.isna()] = np.nan
    return s


def market_env(c: pd.Series | None = None):
    """台股加權指數環境（模組 01）。失敗回傳 None。可傳入 load_index() 的結果避免重抓。"""
    try:
        if c is None:
            c = load_index(1)
        if c is None or len(c) < 60:
            return None
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        score = float(market_states(c).iloc[-1])
        state = {1.0: "多頭", 0.0: "空頭"}.get(score, "盤整")
        return {"state": state, "score": score, "close": float(c.iloc[-1]),
                "ma20": float(ma20.iloc[-1]), "ma60": float(ma60.iloc[-1]),
                "date": pd.Timestamp(c.index[-1]).strftime("%Y-%m-%d")}
    except Exception:
        return None


def fetch_closes(stocks: list[dict], years: int = 5) -> dict:
    """批次抓多檔收盤價（族群指數用）。stocks：[{"code","market"}]。回傳 {code: Series}，抓不到的略過。"""
    out = {}
    if not stocks:
        return out
    try:
        import yfinance as yf
        tick = {s["code"]: f"{s['code']}.{s.get('market', 'TW')}" for s in stocks}
        data = yf.download(list(tick.values()), period=f"{years}y", interval="1d", group_by="ticker",
                           auto_adjust=False, actions=False, threads=True, progress=False)
        for code, tk in tick.items():
            try:
                sub = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                c = sub["Close"].dropna().astype(float)
                if len(c) >= 60:
                    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
                    out[code] = c
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


def sector_index(closes: list) -> pd.Series | None:
    """族群指數：同業每日報酬等權平均後累乘（起點 100）。每天至少 rs_min_peers 檔有資料才計算。"""
    cs = [x for x in closes if x is not None and len(x) >= 60]
    if len(cs) < TH["rs_min_peers"]:
        return None
    cs = [x[~x.index.duplicated(keep="last")].sort_index() for x in cs]
    ret = pd.concat([x.pct_change() for x in cs], axis=1)
    cnt = ret.notna().sum(axis=1)
    avg = ret.mean(axis=1).where(cnt >= TH["rs_min_peers"])
    avg = avg.dropna()
    if len(avg) < 60:
        return None
    return (1 + avg).cumprod() * 100


# ------------------------------------------------------------------
# 分析器
# ------------------------------------------------------------------
class Analyzer:
    def __init__(self, df: pd.DataFrame, code: str, params: dict | None = None,
                 mkt: pd.Series | None = None, sec: pd.Series | None = None, sec_name: str = ""):
        """mkt：加權指數收盤序列；sec：族群指數序列（sector_index）。缺少時對應過濾顯示「無法判定」、不套用。"""
        if len(df) < 80:
            raise ValueError(f"歷史資料只有 {len(df)} 根，至少需要 80 根日 K。")
        self.df = df.copy()
        self.code = code
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.etf = is_etf(code)
        self.buy_c, self.sell_c = cost_rates(code, self.p)
        self._indicators()
        self._context(mkt, sec)
        self.sec_name = sec_name
        self._pivots()
        self.hits = [[] for _ in range(self.n)]
        for i in range(70, self.n):
            self.hits[i] = self._patterns(i)

    # ---------------- 指標 ----------------
    def _indicators(self):
        d = self.df
        c = d["close"]
        for k in (5, 10, 20, 60):
            d[f"ma{k}"] = c.rolling(k).mean()
        d["vma5p"] = d["volume"].shift(1).rolling(5).mean()
        d["vma5"] = d["volume"].rolling(5).mean()
        d["vma20"] = d["volume"].rolling(20).mean()
        d["vr"] = d["volume"] / d["vma5p"]
        d["chg"] = c.pct_change() * 100
        rng = (d["high"] - d["low"]).replace(0, np.nan)
        top = d[["open", "close"]].max(axis=1)
        bot = d[["open", "close"]].min(axis=1)
        d["upper"] = ((d["high"] - top) / rng).fillna(0)
        d["lower"] = ((bot - d["low"]) / rng).fillna(0)
        d["bodyr"] = ((d["close"] - d["open"]).abs() / rng).fillna(0)
        # KD (9,3,3)
        lo9 = d["low"].rolling(9, min_periods=1).min()
        hi9 = d["high"].rolling(9, min_periods=1).max()
        rsv = ((c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100).fillna(50).to_numpy()
        k = np.empty(len(d)); dd = np.empty(len(d))
        kp = dp = 50.0
        for i, r in enumerate(rsv):
            kp = kp * 2 / 3 + r / 3
            dp = dp * 2 / 3 + kp / 3
            k[i], dd[i] = kp, dp
        d["k"], d["d"] = k, dd
        # MACD (12,26,9)
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        d["dif"] = e12 - e26
        d["dea"] = d["dif"].ewm(span=9, adjust=False).mean()
        d["hist"] = d["dif"] - d["dea"]
        # RSI (Wilder)
        delta = c.diff()
        for n in (6, 14):
            up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            d[f"rsi{n}"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)
        d["hi60"] = d["high"].rolling(60).max()
        d["lo60"] = d["low"].rolling(60).min()
        d["pos60"] = (c - d["lo60"]) / (d["hi60"] - d["lo60"]).replace(0, np.nan)
        prev_c = c.shift(1)
        tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_c).abs(), (d["low"] - prev_c).abs()], axis=1).max(axis=1)
        d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        d["atrp"] = d["atr"] / c * 100
        d["hh20p"] = d["high"].shift(1).rolling(20).max()
        d["ll20p"] = d["low"].shift(1).rolling(20).min()

        self.n = len(d)
        g = lambda col: d[col].to_numpy(dtype=float)
        self.O, self.H, self.L, self.C, self.V = g("open"), g("high"), g("low"), g("close"), g("volume")
        self.VR, self.CHG = np.nan_to_num(g("vr")), np.nan_to_num(g("chg"))
        self.UP, self.LOW_SH, self.BR = g("upper"), g("lower"), g("bodyr")
        self.MA5, self.MA10, self.MA20, self.MA60 = g("ma5"), g("ma10"), g("ma20"), g("ma60")
        self.K, self.D, self.HIST = g("k"), g("d"), g("hist")
        self.HI60, self.LO60, self.POS60 = g("hi60"), g("lo60"), g("pos60")
        self.HH20P, self.LL20P = g("hh20p"), g("ll20p")
        self.ATR, self.ATRP = g("atr"), g("atrp")
        self.dates = [x.strftime("%Y-%m-%d") for x in d.index]

    def _context(self, mkt, sec):
        """把大盤與族群對齊到個股的交易日（只用當天以前的資料）。"""
        idx = self.df.index
        nd = TH["rs_days"]
        self.MKT = np.full(self.n, np.nan)
        self.MKT_RET = np.full(self.n, np.nan)
        self.SEC_RET = np.full(self.n, np.nan)
        if mkt is not None and len(mkt) >= 60:
            u = mkt.index.union(idx)
            self.MKT = market_states(mkt).reindex(u).ffill().reindex(idx).to_numpy(dtype=float)
            mm = mkt.reindex(u).ffill().reindex(idx)  # 以個股交易日對齊後再算 20 日報酬
            self.MKT_RET = ((mm / mm.shift(nd) - 1) * 100).to_numpy(dtype=float)
        if sec is not None and len(sec) >= 60:
            sv = sec.reindex(sec.index.union(idx)).ffill().reindex(idx)
            self.SEC_RET = ((sv / sv.shift(nd) - 1) * 100).to_numpy(dtype=float)
        self.SEC_RS = self.SEC_RET - self.MKT_RET  # 族群相對大盤（百分點）

    def stop_atr(self, entry, stop, i):
        a = self.ATR[i]
        return (entry - stop) / a if np.isfinite(a) and a > 0 else float("nan")

    def filter_notes(self, i, sig=None):
        """三項過濾（測試中）的判斷結果。回傳 dict：block（擋下進場的原因）、halve（股數減半原因）、info（說明）。"""
        out = {"block": [], "halve": [], "info": []}
        if sig is not None:
            k = self.stop_atr(sig["entry"], round_down(sig["stop"], self.etf), i)
            if np.isfinite(k):
                txt = f"停損距離 {k:.2f} 倍 ATR"
                out["info"].append(txt)
                if self.p.get("f_stop_atr") and k < TH["stop_atr_min"]:
                    out["block"].append(f"{txt} < {TH['stop_atr_min']:g}（小於一天正常波動，容易被洗出）【測試中】")
        ms = self.MKT[i]
        if np.isfinite(ms):
            if ms == 0.0:
                out["info"].append("加權指數空頭")
                if self.p.get("f_market"):
                    out["halve"].append("加權指數空頭（收盤 < MA20 < MA60），股數減半【測試中】")
        rs = self.SEC_RS[i]
        if self.etf:
            out["info"].append("族群強弱：ETF 不適用")
        elif np.isfinite(rs):
            name = self.sec_name or "族群"
            out["info"].append(f"{name} 近 {TH['rs_days']} 日比大盤 {rs:+.1f} 個百分點")
            if self.p.get("f_sector") and rs < TH["rs_weak"]:
                out["block"].append(f"{name}近 {TH['rs_days']} 日落後大盤 {rs:.1f} 個百分點（弱勢族群）【測試中】")
        else:
            out["info"].append("族群強弱：無同業資料，無法判定")
        return out

    def _pivots(self):
        n, H, L = TH["pivot"], self.H, self.L
        ph, pl = [], []
        for k in range(n, self.n - n):
            if H[k] == H[k - n:k + n + 1].max() and (not ph or k - ph[-1] > n or H[k] != H[ph[-1]]):
                ph.append(k)
            if L[k] == L[k - n:k + n + 1].min() and (not pl or k - pl[-1] > n or L[k] != L[pl[-1]]):
                pl.append(k)
        self.ph, self.pl = np.array(ph, dtype=int), np.array(pl, dtype=int)

    def conf_highs(self, i):  # 到第 i 天為止已確認的波段高點
        return self.ph[self.ph <= i - TH["pivot"]]

    def conf_lows(self, i):
        return self.pl[self.pl <= i - TH["pivot"]]

    # ---------------- 基本判斷 ----------------
    def red(self, i):
        return self.C[i] > self.O[i] and self.CHG[i] > 0

    def long_red(self, i):
        return self.C[i] > self.O[i] and self.CHG[i] >= TH["long_red"] and self.BR[i] >= TH["body"]

    def big(self, i):
        return self.VR[i] >= TH["vol_big"]

    def burst(self, i):
        return self.VR[i] >= TH["vol_burst"]

    def cross_up(self, i, lvl):
        return self.C[i] > lvl >= self.C[i - 1]

    def low_zone(self, i):
        lo10 = self.L[max(0, i - 9):i + 1].min()
        lo60, hi60 = self.LO60[i], self.HI60[i]
        if not (np.isfinite(lo60) and np.isfinite(hi60)):
            return False
        return lo10 <= lo60 * (1 + TH["low_near"]) and hi60 >= lo60 * (1 + TH["low_drop"])

    # ---------------- 18 型態 ----------------
    def _patterns(self, i):
        out = []
        H, L, C, O, VR = self.H, self.L, self.C, self.O, self.VR

        def add(pid, ref, stop, note=""):
            out.append({"id": pid, "name": PATTERNS[pid], "ref": float(ref),
                        "stop": float(stop), "note": note})

        # 1 低檔大量長紅K
        if self.low_zone(i) and self.long_red(i) and self.burst(i):
            add(1, (O[i] + C[i]) / 2, L[i], "低檔爆量長紅（突破價取長紅 1/2）")

        # 2 破切反彈過高 / 7 空轉多過空高
        hs = self.conf_highs(i - 1)
        if len(hs) >= 2:
            k1, k2 = hs[-2], hs[-1]
            if H[k1] > H[k2]:
                slope = (H[k2] - H[k1]) / (k2 - k1)
                line = lambda x: H[k2] + slope * (x - k2)
                crossed = any(C[j] > line(j) and C[j - 1] <= line(j - 1)
                              for j in range(max(k2 + 1, i - 9), i + 1))
                if crossed and self.cross_up(i, H[k2]) and self.red(i) and self.big(i):
                    add(2, H[k2], L[i], f"破下降切線並過反彈高 {H[k2]:g}")
        m = i - 60 + int(np.argmin(L[i - 60:i]))
        prior = hs[hs < m] if len(hs) else hs
        if len(prior) and i - m >= 5:
            lvl = H[prior[-1]]
            if self.cross_up(i, lvl) and self.red(i) and self.big(i):
                add(7, lvl, L[i], f"突破空頭高點 {lvl:g}")

        # 3 大量雙腳反轉
        ls = self.conf_lows(i - 1)
        ls = ls[ls >= i - 60]
        if len(ls) >= 2:
            l1, l2 = ls[-2], ls[-1]
            if l2 - l1 >= 5 and L[l1] * 0.97 <= L[l2] <= L[l1] * 1.12 and i - l2 <= 30:
                vf = lambda k: VR[max(0, k - 1):k + 2].max()
                if vf(l1) >= TH["vol_big"] and vf(l2) >= TH["vol_big"]:
                    neck = H[l1:l2 + 1].max()
                    if self.cross_up(i, neck) and self.red(i) and self.big(i):
                        add(3, neck, L[i], f"雙腳 {L[l1]:g}/{L[l2]:g}，突破頸線 {neck:g}")

        # 4 缺口之上續漲
        for g in range(i - 3, i - 21, -1):
            if L[g] > H[g - 1] and VR[g] >= TH["vol_big"]:
                gap = H[g - 1]
                if L[g:i].min() > gap:
                    top = H[g:i].max()
                    if C[i] > top and self.red(i) and self.big(i):
                        add(4, top, L[i], f"缺口 {gap:g} 未回補，突破整理 {top:g}")
                break

        # 5 碎步上漲攻擊
        w = range(i - 8, i)
        if (all(abs(self.CHG[k]) <= TH["small_k"] for k in w)
                and sum(L[k] > L[k - 1] for k in w) >= 5
                and all(C[k] >= self.MA20[k] * 0.98 for k in w)
                and self.MA20[i - 1] >= self.MA20[i - 9]):
            top = H[i - 8:i].max()
            if self.long_red(i) and self.big(i) and C[i] > top:
                add(5, top, L[i], "碎步墊高後長紅攻擊")

        # 6 底部洗盤上攻
        bl, bh = L[i - 50:i - 10].min(), H[i - 50:i - 10].max()
        if bh / bl - 1 <= 0.20 and L[i - 10:i].min() < bl and C[i - 1] >= bl:
            if self.cross_up(i, bh) and self.red(i) and self.big(i):
                add(6, bh, L[i], f"破底洗盤後突破整理上緣 {bh:g}")

        # 8 紅黑紅
        if (C[i - 2] > O[i - 2] and C[i - 1] < O[i - 1] and C[i - 1] > O[i - 2]
                and self.red(i) and C[i] > max(H[i - 2], H[i - 1]) and C[i] > self.MA20[i]):
            add(8, max(H[i - 2], H[i - 1]), L[i - 2:i + 1].min(), "多方夾擊後突破")

        # 9 突破大量黑K
        for b in range(i - 1, i - 16, -1):
            if C[b] < O[b] and VR[b] >= TH["vol_burst"]:
                if self.cross_up(i, H[b]) and self.red(i) and self.big(i):
                    add(9, H[b], L[i], f"突破 {self.dates[b]} 爆量黑K高點 {H[b]:g}")
                break

        # 10 低檔連2日大量突破
        for a in range(i - 3, i - 7, -1):
            if VR[a] >= TH["vol_big"] and VR[a + 1] >= TH["vol_big"] and self.low_zone(a + 1):
                lvl = max(H[a], H[a + 1])
                if self.cross_up(i, lvl) and self.red(i):
                    add(10, lvl, min(L[a], L[a + 1]), f"突破 2 日大量高點 {lvl:g}")
                break

        # 11 低檔大量長下影
        for s in range(i - 1, i - 6, -1):
            if self.LOW_SH[s] >= 0.5 and VR[s] >= TH["vol_burst"] and self.low_zone(s):
                if self.cross_up(i, H[s]) and self.red(i):
                    add(11, H[s], L[s], f"{self.dates[s]} 長下影被收復")
                break

        # 12 空頭反彈月線上盤整
        if self.MA60[i - 10] < self.MA60[i - 30]:
            w = range(i - 8, i)
            top, bot = H[i - 8:i].max(), L[i - 8:i].min()
            if all(C[k] >= self.MA20[k] for k in w) and top / bot - 1 <= 0.08:
                if self.cross_up(i, top) and self.red(i) and self.big(i):
                    add(12, top, L[i], f"月線上整理後突破 {top:g}")

        # 15 均線糾結紅K突破
        mas = [self.MA5[i - 1], self.MA10[i - 1], self.MA20[i - 1]]
        if (max(mas) - min(mas)) / C[i - 1] <= TH["squeeze"] and abs(self.MA20[i - 1] - self.MA60[i - 1]) / C[i - 1] <= 0.06:
            top_ma = max(self.MA5[i], self.MA10[i], self.MA20[i], self.MA60[i])
            if self.long_red(i) and self.big(i) and C[i] > top_ma and C[i] > self.HH20P[i]:
                add(15, self.HH20P[i], L[i], "均線糾結後長紅突破")

        # 17 島型反轉
        for k in range(2, 11):
            g = i - k
            if H[g] < L[g - 1]:
                isl = H[g:i].max()
                if isl < L[g - 1] and L[i] > isl and VR[i] >= 1.2:
                    add(17, isl, isl, f"島型區 {self.dates[g]} 起，向上跳空")
                break

        # 18 突破上升軌道
        x = np.arange(i - 30, i)
        y = H[i - 30:i]
        b, a = np.polyfit(x, y, 1)
        if b > 0:
            fit_c = np.polyval(np.polyfit(x, C[i - 30:i], 1), x)
            ss = ((C[i - 30:i] - C[i - 30:i].mean()) ** 2).sum()
            r2 = 1 - ((C[i - 30:i] - fit_c) ** 2).sum() / ss if ss > 0 else 0
            gain = (a + b * (i - 1)) / (a + b * (i - 30)) - 1
            res = (y - (a + b * x)).max()
            up_i, up_p = a + b * i + res, a + b * (i - 1) + res
            if r2 >= 0.6 and 0.03 <= gain <= 0.25 and C[i] > up_i and C[i - 1] <= up_p \
                    and self.red(i) and self.big(i):
                add(18, up_i, L[i], f"突破上升軌道線 {up_i:.2f}")
        return out

    # ---------------- 結構與生命週期 ----------------
    def structure(self, i):
        hs, ls = self.conf_highs(i), self.conf_lows(i)
        if len(hs) < 2 or len(ls) < 2:
            return {"state": "無法判定", "score": None, "last_high": None, "last_low": None}
        h1, h2, l1, l2 = self.H[hs[-2]], self.H[hs[-1]], self.L[ls[-2]], self.L[ls[-1]]
        c = self.C[i]
        if c < l2:
            state = "空頭（跌破前波低點）"
        elif h2 > h1 and l2 > l1:
            state = "多頭（頭頭高、底底高）"
        elif h2 < h1 and l2 < l1:
            state = "空頭（頭頭低、底底低）"
        else:
            state = "盤整"
        if c > h2 and not state.startswith("多頭"):
            state += "，已突破前波高點"
        score = 1.0 if state.startswith("多頭") else 0.0 if state.startswith("空頭") else 0.5
        return {"state": state, "score": score, "last_high": float(h2), "last_low": float(l2),
                "high_date": self.dates[hs[-1]], "low_date": self.dates[ls[-1]]}

    def lifecycle(self, i, st):
        if st["state"].startswith("多頭"):
            return 5, "⑤ 多頭結構、趨勢延伸"
        recent = [h for j in range(max(70, i - TH["retest_days"]), i + 1) for h in self.hits[j]]
        if recent and self.C[i] >= max(h["ref"] for h in recent) * TH["retest_hold"]:
            return 4, "④ 突破頸線、回測確認中"
        if st["last_high"] and self.C[i] >= st["last_high"] * 0.97:
            return 3, "③ 測試頸線（前波高點）"
        if any(self.low_zone(j) and self.long_red(j) and self.big(j) for j in range(i - 9, i + 1)):
            return 2, "② 底部出現帶量長紅"
        hs = self.conf_highs(i)
        if len(hs) >= 2 and self.H[hs[-2]] > self.H[hs[-1]]:
            k1, k2 = hs[-2], hs[-1]
            line = self.H[k2] + (self.H[k2] - self.H[k1]) / (k2 - k1) * (i - k2)
            if self.C[i] > line:
                return 1, "① 突破下降趨勢線"
        return 0, "尚未進入轉多階段"

    # ---------------- 高檔衰退警報 ----------------
    def decline(self, i):
        sig = []
        if not (np.isfinite(self.POS60[i]) and self.POS60[i] >= 0.6):
            return sig
        d = self.df
        if self.C[i] > self.C[i - 10] and d["vma5"].iloc[i] < d["vma20"].iloc[i] * 0.8:
            sig.append("價漲量縮")
        if self.K[i] >= 80:
            sig.append(f"KD 高檔（K={self.K[i]:.0f}）")
        if any(self.C[j] < self.O[j] and self.CHG[j] <= -3 and self.VR[j] >= 2 for j in range(i - 4, i + 1)):
            sig.append("爆量長黑")
        if any(self.UP[j] >= 0.5 and self.VR[j] >= 1.2 for j in range(i - 2, i + 1)):
            sig.append("長上影線")
        if any(self.K[j] < self.D[j] and self.K[j - 1] >= self.D[j - 1] and self.K[j - 1] >= 70
               for j in range(i - 2, i + 1)):
            sig.append("KD 高檔死叉")
        if self.HIST[i] > 0 and self.HIST[i] < self.HIST[i - 1] < self.HIST[i - 2] < self.HIST[i - 3]:
            sig.append("MACD 柱狀體連續縮短")
        return sig

    # ---------------- 壓力 ----------------
    def resistance_above(self, i, price):
        hs = self.conf_highs(i)
        hs = hs[hs >= i - 120]
        lv = sorted(self.H[k] for k in hs if self.H[k] > price * 1.001)
        return float(lv[0]) if lv else None

    # ---------------- 進場訊號（V1.2：四種買點） ----------------
    def strategy_b(self, i):
        """策略 B：多頭回檔 → 觸及 MA10/MA20 支撐不破月線 → 第一根帶量紅 K。"""
        C, H, L = self.C, self.H, self.L
        if not (self.MA20[i] > self.MA60[i] and self.MA20[i] > self.MA20[i - 5] and C[i] > self.MA60[i]):
            return None
        if self.structure(i)["state"].startswith("空頭"):
            return None
        pk = i - 10 + int(np.argmax(H[i - 10:i]))
        if pk >= i - 1:
            return None
        seg = L[pk + 1:i + 1]
        kl = pk + 1 + int(np.argmin(seg))
        pb = float(seg.min())
        depth = 1 - pb / H[pk]
        if not (TH["b_pull_min"] <= depth <= TH["b_pull_max"]):
            return None
        sup = max(self.MA10[kl], self.MA20[kl])
        if pb > sup * (1 + TH["b_touch"]):
            return None
        if any(C[k] < self.MA20[k] * 0.98 for k in range(pk + 1, i + 1)):
            return None
        trig = lambda k: self.red(k) and C[k] > H[k - 1] and self.VR[k] >= TH["b_vr"]
        if not trig(i) or any(trig(k) for k in range(kl + 1, i)):
            return None
        stop = float(min(L[i], L[i - 1]))  # 跌破止跌 K（前一日與觸發紅 K）最低點＝失敗
        return {"type": "多頭回檔買上漲（策略B）", "mode": "close", "entry": float(C[i]), "stop": stop,
                "ref": float(H[i - 1]), "ids": [], "src": self.dates[i],
                "note": f"自 {H[pk]:g} 回檔 {depth * 100:.1f}% 至 MA10/MA20 支撐後首根帶量紅K"}

    def candidates(self, i, live=True):
        """第 i 天收盤後所有可能的買點。live=True 時，含 1～2 天前突破、仍在有效期的限價單。"""
        C, L = self.C, self.L
        out = []
        ids = lambda j: [h["id"] for h in self.hits[j]]
        # A. 突破收盤買（積極）
        if self.hits[i]:
            out.append({"type": "突破收盤買（積極）", "mode": "close", "entry": float(C[i]),
                        "stop": max(h["stop"] for h in self.hits[i]),
                        "ref": max(h["ref"] for h in self.hits[i]), "ids": ids(i), "src": self.dates[i]})
        # B. 拉回限價買（突破後掛單，3 日有效）
        lo = i - (TH["limit_valid"] - 1) if live else i
        for j in range(i, max(70, lo) - 1, -1):
            if not self.hits[j]:
                continue
            ref = max(h["ref"] for h in self.hits[j])
            limit = min(round_up(ref * (1 + TH["limit_buffer"]), self.etf), float(C[j]))
            stop = ref - TH["stop_atr"] * self.ATR[j]
            if j < i and (L[j + 1:i + 1].min() <= limit or C[j + 1:i + 1].min() < stop):
                break  # 已成交或已跌破 → 單子失效
            out.append({"type": "拉回限價買（突破後掛單）", "mode": "limit", "entry": float(limit),
                        "stop": float(stop), "ref": ref, "ids": ids(j), "src": self.dates[j],
                        "days_left": TH["limit_valid"] - (i - j)})
            break
        # C. 回測確認買（保守）
        for j in range(i - 2, max(70, i - TH["retest_days"]) - 1, -1):
            if not self.hits[j]:
                continue
            ref = max(h["ref"] for h in self.hits[j])
            pb = L[j + 1:i + 1].min()
            if (pb >= ref * TH["retest_hold"] and pb < C[j]
                    and C[i] > self.O[i] and C[i] > self.H[i - 1] and C[i] >= ref):
                out.append({"type": "回測確認買（保守）", "mode": "close", "entry": float(C[i]),
                            "stop": float(pb), "ref": ref, "ids": ids(j), "src": self.dates[j]})
            break
        # D. 策略 B（收盤買）
        b = self.strategy_b(i)
        if b:
            out.append(b)
        # E. 策略 B 掛單：觸發紅 K 後拉回實體 1/2 買，停損同止跌 K 低點
        for j in range(i, max(70, lo) - 1, -1):
            bj = b if j == i else self.strategy_b(j)
            if not bj:
                continue
            limit = min(round_up((self.O[j] + C[j]) / 2, self.etf), float(C[j]))
            if j < i and (L[j + 1:i + 1].min() <= limit or C[j + 1:i + 1].min() < bj["stop"]):
                break
            out.append({**bj, "type": "多頭回檔掛單（策略B）", "mode": "limit", "entry": float(limit),
                        "src": self.dates[j], "days_left": TH["limit_valid"] - (i - j)})
            break
        return [c for c in out if c["stop"] < c["entry"]]

    def signal(self, i):
        """相容舊介面：回傳風報比最好的候選。"""
        cs = self.candidates(i)
        return max(cs, key=lambda c: self.plan(i, c["entry"], c["stop"])["rr"]) if cs else None

    def suitability(self, i):
        ap = self.ATRP[i]
        if not np.isfinite(ap):
            return None
        if ap > TH["atr_max"]:
            return f"股性波動過大（日均波動 ATR {ap:.1f}% > {TH['atr_max']}%），停損距離難以配合 {self.p['target_net']:g}% 目標"
        if ap < TH["atr_min"]:
            return f"股性波動過小（ATR {ap:.1f}% < {TH['atr_min']}%），{self.p['hold_days']} 日內不易達 {self.p['target_net']:g}%"
        return None

    def vetoes(self, i, st=None):
        v = []
        if self.CHG[i] <= -TH["limit"] or self.CHG[i - 1] <= -TH["limit"]:
            v.append("近 2 日出現跌停，尚未止跌")
        if np.isfinite(self.LL20P[i]) and self.C[i] < self.LL20P[i]:
            v.append("收盤跌破近 20 日最低點，仍在破底")
        dec = self.decline(i)
        if len(dec) >= 3:
            v.append("高檔衰退警報：" + "、".join(dec))
        return v

    def plan(self, i, entry, stop):
        tp3 = target_price(entry, self.p["target_net"], self.buy_c, self.sell_c, self.etf)
        tp2 = target_price(entry, 2.0, self.buy_c, self.sell_c, self.etf)
        stop = round_down(stop, self.etf)
        loss_pct = net_return(entry, stop, self.buy_c, self.sell_c)
        rr = self.p["target_net"] / abs(loss_pct) if loss_pct < 0 else float("inf")
        per_share_cost = entry * (1 + self.buy_c)
        per_share_loss = per_share_cost - stop * (1 - self.sell_c)
        shares = int(min(self.p["capital"] // per_share_cost,
                         self.p["max_loss"] // per_share_loss if per_share_loss > 0 else 1e12))
        res = self.resistance_above(i, entry)
        return {"entry": entry, "tp3": tp3, "tp2": tp2, "stop": stop, "loss_pct": loss_pct,
                "rr": rr, "shares": max(shares, 0), "cost": shares * per_share_cost,
                "max_loss": shares * per_share_loss,
                "profit": shares * (tp3 * (1 - self.sell_c) - per_share_cost),
                "resistance": res}

    def tradable(self, i, sig, entry=None):
        """風控檢查：否決條件、股性、風報比、上方壓力。回傳 (可否, 原因列表, 計畫)。"""
        entry = sig["entry"] if entry is None else entry
        reasons = list(self.vetoes(i))
        su = self.suitability(i)
        if su:
            reasons.append(su)
        pl = self.plan(i, entry, sig["stop"])
        if pl["rr"] < self.p["min_rr"]:
            reasons.append(f"停損距離過大（淨虧 {pl['loss_pct']:.2f}%），風報比 {pl['rr']:.2f} < {self.p['min_rr']}")
        if pl["resistance"] and pl["resistance"] < pl["tp2"]:
            reasons.append(f"上方壓力 {pl['resistance']:g} 低於 2% 停利價 {pl['tp2']:g}，獲利空間不足")
        reasons += self.filter_notes(i, {**sig, "entry": entry})["block"]
        return (not reasons), reasons, pl

    # ---------------- 回測 ----------------
    FUNNEL_ORDER = ["出現訊號", "否決（跌停／破底／衰退）", "股性不符", "風報比不足", "上方壓力太近",
                    "停損太近（<1 ATR）", "族群弱勢", "開高不追", "漲停買不到", "開低破停損", "限價未成交", "成交"]

    @staticmethod
    def _reason_key(r):
        if "ATR】" in r or "倍 ATR" in r:
            return "停損太近（<1 ATR）"
        if "弱勢族群" in r:
            return "族群弱勢"
        if "股性" in r:
            return "股性不符"
        if "風報比" in r:
            return "風報比不足"
        if "壓力" in r:
            return "上方壓力太近"
        return "否決（跌停／破底／衰退）"

    def _locked_up(self, e):
        return self.O[e] == self.H[e] == self.L[e] and self.CHG[e] >= TH["limit"]

    def _simulate(self, i, c, hold):
        """回傳 (交易 dict 或 None, 出場 index, 未成交原因)。資料不足回傳 (None, None, None)。"""
        O, H, L, C, n = self.O, self.H, self.L, self.C, self.n
        stop = round_down(c["stop"], self.etf)
        if c["mode"] == "close":
            e = i + 1
            if e >= n:
                return None, None, None
            if self._locked_up(e):
                return None, None, "漲停買不到"
            if O[e] > C[i] * (1 + TH["no_chase"]):
                return None, None, "開高不追"
            if O[e] <= stop:
                return None, None, "開低破停損"
            entry, start = float(O[e]), e
        else:
            limit, start = c["entry"], None
            for e in range(i + 1, min(i + 1 + TH["limit_valid"], n)):
                if O[e] <= limit:
                    entry, start = float(O[e]), e
                    break
                if L[e] <= limit:
                    entry, start = float(limit), e
                    break
            if start is None:
                return (None, None, "限價未成交") if i + TH["limit_valid"] < n else (None, None, None)
        tp = target_price(entry, self.p["target_net"], self.buy_c, self.sell_c, self.etf)
        last = start + hold - 1
        if last >= n:
            return None, None, None
        exit_ = why = xj = None
        mae = mfe = 0.0
        for j in range(start, last + 1):
            mae = min(mae, L[j] / entry - 1)
            mfe = max(mfe, H[j] / entry - 1)
            if O[j] <= stop and (j > start or c["mode"] == "limit"):
                exit_, why, xj = float(min(O[j], entry if j == start else O[j])), "停損（跳空）", j
                break
            if L[j] <= stop:
                exit_, why, xj = float(stop), "停損", j
                break
            if c["mode"] == "limit" and j == start:
                continue  # 掛單成交當天保守處理：不計停利
            if H[j] >= tp:
                exit_, why, xj = float(O[j] if O[j] >= tp else tp), "停利", j
                break
        if exit_ is None:
            exit_, why, xj = float(C[last]), "期滿", last
        tag = "、".join(PATTERNS[k] for k in c["ids"]) or c.get("note", "")[:12]
        return {
            "訊號日": self.dates[i], "進場日": self.dates[start], "進場價": round(entry, 2),
            "停損": stop, "停利": tp, "出場日": self.dates[xj], "出場價": round(exit_, 2),
            "結果": why, "淨報酬%": round(net_return(entry, exit_, self.buy_c, self.sell_c), 2),
            "持有天數": xj - start + 1, "最大不利%": round(mae * 100, 2), "最大有利%": round(mfe * 100, 2),
            "買點": c["type"].split("（")[0], "型態": tag,
        }, xj, None

    def backtest(self):
        hold = int(self.p["hold_days"])
        trades, funnel = [], {k: 0 for k in self.FUNNEL_ORDER}
        i = 70
        while i < self.n - 1:
            cs = self.candidates(i, live=False)
            if not cs:
                i += 1
                continue
            funnel["出現訊號"] += 1
            passing, first = [], None
            for c in cs:
                ok, why, pl = self.tradable(i, c)
                if ok:
                    passing.append((pl["rr"], c))
                elif first is None:
                    first = why[0]
            if not passing:
                funnel[self._reason_key(first)] += 1
                i += 1
                continue
            c = max(passing, key=lambda x: x[0])[1]
            t, xj, miss = self._simulate(i, c, hold)
            if t is None and miss is None:
                break  # 資料尾端，交易尚未結束
            if t is None:
                funnel[miss] += 1
                i += 1
                continue
            funnel["成交"] += 1
            # 研究用標記（訊號當天的狀態，不影響結果）
            k = self.stop_atr(c["entry"], round_down(c["stop"], self.etf), i)
            t["停損/ATR"] = round(k, 2) if np.isfinite(k) else None
            ms = self.MKT[i]
            t["大盤"] = {1.0: "多頭", 0.0: "空頭", 0.5: "盤整"}.get(ms, "無資料") if np.isfinite(ms) else "無資料"
            rs = self.SEC_RS[i]
            t["族群RS"] = round(float(rs), 2) if np.isfinite(rs) else None
            trades.append(t)
            i = xj + 1
        self.funnel = funnel
        return pd.DataFrame(trades)

    @staticmethod
    def summarize(bt: pd.DataFrame):
        if bt is None or bt.empty:
            return {"n": 0}
        r = bt["淨報酬%"]
        wins, losses = r[r > 0], r[r <= 0]
        streak = best = 0
        for x in r:
            streak = streak + 1 if x <= 0 else 0
            best = max(best, streak)
        return {
            "n": len(bt),
            "tp_rate": (bt["結果"] == "停利").mean() * 100,
            "sl_rate": bt["結果"].str.startswith("停損").mean() * 100,
            "win_rate": (r > 0).mean() * 100,
            "avg": r.mean(), "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0,
            "max_streak": best, "avg_days": bt["持有天數"].mean(),
            "worst_mae": bt["最大不利%"].min(),
            "first": bt["訊號日"].iloc[0], "last": bt["訊號日"].iloc[-1],
        }

    # ---------------- 綜合報告 ----------------
    def report(self, market=None):
        i = self.n - 1
        C = self.C
        st = self.structure(i)
        stage, stage_txt = self.lifecycle(i, st)
        dec = self.decline(i)
        cands = self.candidates(i, live=True)
        evals = [(c,) + self.tradable(i, c) for c in cands]  # (c, ok, why, pl)
        passing = [e for e in evals if e[1]]
        if passing:
            sig, ok, why, pl = max(passing, key=lambda e: e[3]["rr"])
        elif evals:
            sig, ok, why, pl = max(evals, key=lambda e: e[3]["rr"])
        else:
            sig, ok, why, pl = None, False, list(self.vetoes(i)), None
            su0 = self.suitability(i)
            if su0:
                why.append(su0)
        bt = self.backtest()
        stats = self.summarize(bt)
        enough = stats["n"] >= 30

        recent = [(self.dates[j], h) for j in range(max(70, i - 9), i + 1) for h in self.hits[j]]

        # ---- 10 模組評分 ----
        mods = []

        def mod(no, name, score, text):
            mods.append({"no": no, "name": name, "score": score, "text": text})

        fnote = self.filter_notes(i, sig)
        rs_txt = next((x for x in fnote["info"] if "族群" in x or "大盤 " in x), "")
        if market:
            mod("01", "市場環境", market["score"],
                f"加權指數 {market['state']}（{market['close']:,.0f}，MA20 {market['ma20']:,.0f}／MA60 {market['ma60']:,.0f}，{market['date']}）"
                + (f"；{rs_txt}" if rs_txt else "") + "；法人資料未納入")
        else:
            mod("01", "市場環境", None, "無法取得加權指數，無法判定" + (f"；{rs_txt}" if rs_txt else ""))
        mod("02", "趨勢結構", st["score"],
            st["state"] + (f"；前高 {st['last_high']:g}、前低 {st['last_low']:g}" if st["last_high"] else ""))
        if self.hits[i] or any(("回測" in c["type"] or "拉回" in c["type"]) for c in cands):
            s3 = 1.0
        elif recent:
            s3 = 0.5
        else:
            s3 = 0.0
        mod("03", "18 種空轉多", s3,
            "；".join(f"{d} {h['name']}" for d, h in recent[-4:]) if recent else "近 10 日無符合型態（#13、#14、#16 需人工看圖）")
        mod("04", "多頭生命週期", {5: 1.0, 4: 1.0, 3: 0.5, 2: 0.5, 1: 0.0, 0: 0.0}[stage], stage_txt)
        vr, chg = self.VR[i], self.CHG[i]
        if C[i] > self.O[i] and vr >= 1.0:
            s5, t5 = 1.0, f"收紅、量比 {vr:.2f}（價漲量增）"
        elif C[i] < self.O[i] and vr >= 1.5:
            s5, t5 = 0.0, f"收黑、量比 {vr:.2f}（價跌量增）"
        elif vr < 0.8 and abs(chg) < 2:
            s5, t5 = 0.5, f"量縮（量比 {vr:.2f}），僅觀察"
        else:
            s5, t5 = 0.5, f"漲跌 {chg:+.2f}%、量比 {vr:.2f}"
        if self.UP[i] >= 0.5:
            t5 += "；長上影"
        if self.LOW_SH[i] >= 0.5:
            t5 += "；長下影"
        mod("05", "量價與 K 棒", s5, t5)
        ma_ok = C[i] >= self.MA20[i]
        s6 = 1.0 if ma_ok and self.MA20[i] >= self.MA60[i] else 0.5 if ma_ok else 0.0
        mod("06", "均線與支撐壓力", s6,
            f"MA5 {self.MA5[i]:.2f}／MA10 {self.MA10[i]:.2f}／MA20 {self.MA20[i]:.2f}／MA60 {self.MA60[i]:.2f}"
            + (f"；股性 ATR {self.ATRP[i]:.1f}%" if np.isfinite(self.ATRP[i]) else ""))
        mom = [self.K[i] > self.D[i], self.HIST[i] > self.HIST[i - 1],
               40 <= self.df["rsi14"].iloc[i] <= 75]
        s7 = 1.0 if sum(mom) >= 2 else 0.5 if sum(mom) == 1 else 0.0
        mod("07", "KD／MACD／RSI", s7,
            f"K {self.K[i]:.1f}／D {self.D[i]:.1f}；MACD 柱 {self.HIST[i]:.2f}"
            f"（{'擴大' if self.HIST[i] > self.HIST[i-1] else '縮小'}）；RSI14 {self.df['rsi14'].iloc[i]:.1f}")
        s8 = 0.0 if len(dec) >= 3 else 0.5 if dec else 1.0
        mod("08", "高檔衰退警報", s8, "、".join(dec) if dec else "未出現")
        if enough:
            s9 = 1.0 if stats["avg"] > 0 else 0.0
            mod("09", "交易期望值", s9, f"回測 {stats['n']} 筆，平均每筆 {stats['avg']:+.2f}%")
        else:
            mod("09", "交易期望值", None, f"回測只有 {stats['n']} 筆（< 30），樣本不足，無法判定")

        # ---- 決策 ----
        entry = float(C[i])
        why = list(why)
        s10 = 0.0 if why and any(k in w for w in why for k in ("跌停", "破底", "衰退", "風報比", "股性")) else (1.0 if ok else 0.5)
        mod("10", "風控與資金管理", s10, "；".join(why) if why else ("通過" if ok else "尚無進場訊號"))

        avail = [m["score"] for m in mods if m["score"] is not None]
        fit = sum(avail) / len(avail) * 100 if avail else 0.0

        if ok and enough and stats["avg"] <= 0:
            ok = False
            why.append("本股過去同類訊號平均報酬為負")
        early = False
        halve = []
        if ok and st["score"] == 0.0 and pl:
            early = True  # 空頭結構中的空轉多突破：積極型買點，部位減半
            halve.append("早期空轉多買點")
        if ok and pl and fnote["halve"]:
            halve += fnote["halve"]
        if halve and pl:
            pl = dict(pl)
            pl["shares"] = pl["shares"] // (2 ** len(halve))  # 每一項減半條件各減半一次
            pl["halve"] = halve
            per = pl["entry"] * (1 + self.buy_c)
            pl["cost"] = pl["shares"] * per
            pl["max_loss"] = pl["shares"] * (per - pl["stop"] * (1 - self.sell_c))
            pl["profit"] = pl["shares"] * (pl["tp3"] * (1 - self.sell_c) - per)
        unsuitable = self.suitability(i)
        if ok:
            decision, color = ("可進場（掛單）" if sig["mode"] == "limit" else "可進場"), "green"
        elif self.vetoes(i):
            decision, color = "不可進場", "red"
        elif unsuitable:
            decision, color = "不適合本策略", "red"
        else:
            decision, color = "等待", "orange"

        # ---- 下一步觸發條件（不可只給模糊結論）----
        trig = None
        if not ok:
            vneed = float(self.df["vma5"].iloc[i]) * TH["vol_big"]
            cands = [x for x in (st["last_high"], self.HH20P[i], self.H[i])
                     if x is not None and np.isfinite(x) and x > C[i]]
            lvl = round_up(min(cands) * 1.0001, self.etf) if cands else round_up(self.H[i] * 1.0001, self.etf)
            conf = st["last_high"] if st["last_high"] and st["last_high"] > lvl else None
            bull = self.MA20[i] > self.MA60[i] and self.MA20[i] > self.MA20[i - 5] and C[i] > self.MA60[i]
            if unsuitable:
                text = (f"這檔股票日均波動約 {self.ATRP[i]:.1f}%，與「{self.p['hold_days']} 日內淨賺 "
                        f"{self.p['target_net']:g}%」的策略不匹配，系統不提供買點。"
                        f"本策略適合日均波動 {TH['atr_min']}%～{TH['atr_max']}% 的股票。")
            elif self.vetoes(i) and not (sig and ok):
                text = ("① 先等止跌：某天不跌停、不創新低，且收在當日區間上半部（止跌 K）。"
                        f"② 積極買點：之後收盤突破止跌 K 最高價、收紅、量 ≥ {vneed:,.0f} 張；停損 = 止跌 K 最低價。")
                if conf:
                    text += f" ③ 確認買點：收盤站上前波高點 {conf:g}。"
            else:
                text = (f"積極買點：收盤站上 {lvl:g}、收紅、量 ≥ {vneed:,.0f} 張（5 日均量 × {TH['vol_big']}）。"
                        "保守買點：突破後回測不破突破價、再出紅 K。")
                if conf:
                    text += f" 前波高點 {conf:g} 為上方壓力。"
                if bull:
                    text += (f" 多頭回檔買點（策略B）：拉回 MA10 {self.MA10[i]:.2f}～MA20 {self.MA20[i]:.2f} 一帶不破月線，"
                             f"出現第一根收盤高於前一日最高、量比 ≥ {TH['b_vr']} 的紅 K。")
            if not unsuitable:
                if any("ATR" in b for b in fnote["block"]):
                    text += f" 目前訊號的停損離進場價太近（< {TH['stop_atr_min']:g} ATR），等更遠、更明確的結構停損再進場。"
                if any("弱勢族群" in b for b in fnote["block"]):
                    text += f" 族群目前明顯弱於大盤，等族群 {TH['rs_days']} 日表現追上大盤（落後不超過 {abs(TH['rs_weak']):g} 個百分點）再考慮。"
                text += " 觸發當天收盤後再分析一次確認風報比。"
            trig = {"price": float(lvl), "vol": vneed, "text": text}

        # ---- 理由與風險 ----
        pros, cons, conflicts = [], [], []
        for m in mods:
            if m["score"] == 1.0:
                pros.append(f"{m['name']}：{m['text']}")
            elif m["score"] == 0.0:
                cons.append(f"{m['name']}：{m['text']}")
        rsi14 = self.df["rsi14"].iloc[i]
        if rsi14 <= 30 and st["score"] == 0.0:
            conflicts.append(f"RSI14 {rsi14:.1f} 超賣 vs 空頭結構 → 結構優先，超賣只能觀察")
        if self.K[i] > self.D[i] and self.K[i - 1] <= self.D[i - 1] and st["score"] == 0.0:
            conflicts.append("KD 金叉 vs 空頭結構 → 金叉只當反彈觀察訊號")
        if recent and st["last_high"] and C[i] < st["last_high"]:
            conflicts.append(f"有型態訊號 vs 尚未突破前高 {st['last_high']:g} → 只列候選")
        if st["score"] == 1.0 and self.K[i] >= 80:
            conflicts.append("多頭結構 vs KD 超買 → 結構未破前不因 KD 高檔看空")
        if self.long_red(i) and self.big(i) and np.isfinite(self.POS60[i]) and self.POS60[i] >= 0.85:
            conflicts.append("帶量長紅 vs 位在高檔 → 留意高檔出貨換手")
        if early:
            conflicts.append("型態突破 vs 仍是空頭結構 → 屬早期空轉多（積極型）買點，部位已自動減半")
        if ok and fnote["halve"]:
            conflicts.append("個股訊號成立 vs 大盤空頭 → 可進場但股數減半（大盤濾網・測試中）")
        if not ok and fnote["block"]:
            cons.insert(0, "過濾（測試中）：" + "；".join(fnote["block"]))
        if 10 <= stats["n"] < 30 and stats["avg"] < 0:
            cons.insert(0, f"本股過去 {stats['n']} 筆同類訊號平均 {stats['avg']:+.2f}%（為負，樣本雖少仍需警惕）")
        if pl and pl["resistance"] and pl["resistance"] < pl["tp3"]:
            conflicts.append(f"3% 停利價 {pl['tp3']:g} 高於前方壓力 {pl['resistance']:g} → 可改用 2% 停利 {pl['tp2']:g}")

        last_date = self.df.index[-1]
        stale_days = (pd.Timestamp(dt.date.today()) - last_date).days
        return {
            "code": self.code, "date": self.dates[i], "close": entry, "chg": float(self.CHG[i]),
            "volume": float(self.V[i]), "decision": decision, "color": color, "fit": fit,
            "modules": mods, "structure": st, "stage": stage_txt, "signal": sig, "plan": pl,
            "why": why, "trigger": trig, "pros": pros[:3], "cons": cons[:3], "conflicts": conflicts,
            "stats": stats, "enough": enough, "bt": bt, "recent_hits": recent,
            "decline": dec, "stale_days": stale_days, "etf": self.etf, "early": early,
            "costs": (self.buy_c, self.sell_c), "funnel": getattr(self, "funnel", {}),
            "atrp": float(self.ATRP[i]) if np.isfinite(self.ATRP[i]) else None, "candidates": cands,
            "filters": fnote, "filter_flags": {k: bool(self.p.get(k)) for k in FILTERS},
        }
