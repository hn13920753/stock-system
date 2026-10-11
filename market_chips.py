"""
大盤籌碼｜官方盤後資料
------------------------------------------------------------
1. 證交所 BFI82U：三大法人買賣金額統計（上市，元）
2. 期交所 futContractsDate：臺股期貨（TX）三大法人未平倉口數淨額
3. 期交所 futDataDown：臺股期貨近月一般盤收盤價
4. 加權指數收盤：Yahoo Finance ^TWII

全部是盤後資料（約 15:00 後更新），不是盤中即時。
任何來源失敗都會回報原因，不會補假數字。
"""
from __future__ import annotations

import datetime as dt
import io
import time

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}
TWSE_URLS = ["https://www.twse.com.tw/rwd/zh/fund/BFI82U",
             "https://www.twse.com.tw/fund/BFI82U"]
TAIFEX_OI_CSV = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
TAIFEX_OI_HTML = "https://www.taifex.com.tw/cht/3/futContractsDate"
TAIFEX_PRICE_CSV = "https://www.taifex.com.tw/cht/3/futDataDown"


def _num(x) -> float:
    s = str(x).replace(",", "").replace("－", "-").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _group(name: str) -> str | None:
    n = str(name).strip()
    if n.startswith("合計"):
        return "合計"
    if n.startswith("自營"):
        return "自營"
    if n.startswith("投信"):
        return "投信"
    if "外資" in n or "陸資" in n:
        return "外資"
    return None


# ------------------------------------------------------------------
# 1. 證交所三大法人買賣金額（BFI82U）
# ------------------------------------------------------------------
def parse_bfi82u(j: dict) -> pd.DataFrame | None:
    if not isinstance(j, dict) or j.get("stat") != "OK" or not j.get("data"):
        return None
    rows = []
    for a in j["data"]:
        if len(a) < 4:
            continue
        rows.append({"單位名稱": str(a[0]).strip(), "買進金額": _num(a[1]),
                     "賣出金額": _num(a[2]), "買賣差額": _num(a[3])})
    return pd.DataFrame(rows) if rows else None


def fetch_bfi82u_day(day: dt.date, session=None) -> pd.DataFrame | None:
    s = session or requests.Session()
    last_err = None
    for url in TWSE_URLS:
        try:
            r = s.get(url, params={"type": "day", "dayDate": day.strftime("%Y%m%d"), "response": "json"},
                      headers=UA, timeout=10)
            r.raise_for_status()
            return parse_bfi82u(r.json())
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err:
        raise last_err
    return None


def summarize_bfi82u(table: pd.DataFrame) -> dict:
    """單日表 → 外資／投信／自營／合計 買賣超（億元）。"""
    out = {"外資": 0.0, "投信": 0.0, "自營": 0.0, "合計": np.nan}
    for _, r in table.iterrows():
        g = _group(r["單位名稱"])
        if g == "合計":
            out["合計"] = r["買賣差額"] / 1e8
        elif g:
            out[g] += r["買賣差額"] / 1e8
    if not np.isfinite(out["合計"]):
        out["合計"] = out["外資"] + out["投信"] + out["自營"]
    return out


def spot_institutions(days: int = 10, max_lookback: int = 25):
    """最近 N 個交易日現貨三大法人買賣超。回傳 (df, 最新一日原始表, 錯誤列表)。"""
    s = requests.Session()
    today = dt.date.today()
    rows, latest_table, errors = [], None, []
    for k in range(max_lookback):
        d = today - dt.timedelta(days=k)
        if d.weekday() >= 5:
            continue
        try:
            t = fetch_bfi82u_day(d, s)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{d}: {str(e)[:80]}")
            if len(errors) >= 4 and not rows:
                break
            continue
        if t is None:
            continue
        if latest_table is None:
            latest_table = (d, t)
        rows.append({"日期": pd.Timestamp(d), **summarize_bfi82u(t)})
        if len(rows) >= days:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows).sort_values("日期") if rows else None
    return df, latest_table, errors


# ------------------------------------------------------------------
# 2. 期交所臺股期貨三大法人未平倉
# ------------------------------------------------------------------
def _decode(content: bytes) -> str:
    for enc in ("cp950", "big5", "utf-8-sig", "utf-8"):
        try:
            txt = content.decode(enc)
            if "身份別" in txt or "日期" in txt:
                return txt
        except UnicodeDecodeError:
            continue
    return content.decode("cp950", errors="replace")


def parse_taifex_oi_csv(text: str) -> pd.DataFrame | None:
    """期交所『區分各期貨契約』CSV → 每日 外資／投信／自營 未平倉口數淨額。"""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    col = lambda *keys: next((c for c in df.columns if all(k in c for k in keys)), None)
    c_date, c_who = col("日期"), col("身份")
    c_net = col("未平倉", "淨額", "口數")
    c_name = col("商品")
    if not (c_date and c_who and c_net):
        return None
    if c_name:
        df = df[df[c_name].astype(str).str.contains("臺股期貨|台股期貨|TX", regex=True)]
    rows = []
    for (d,), g in df.groupby([c_date]):
        rec = {"日期": pd.to_datetime(str(d).strip())}
        for _, r in g.iterrows():
            who = _group(r[c_who])
            if who in ("外資", "投信", "自營"):
                rec[who] = _num(r[c_net])
        if all(k in rec for k in ("外資", "投信", "自營")):
            rec["合計"] = rec["外資"] + rec["投信"] + rec["自營"]
            rows.append(rec)
    return pd.DataFrame(rows).sort_values("日期") if rows else None


def parse_taifex_oi_html(html: str, day: dt.date) -> dict | None:
    """備援：單日 HTML 報表。取『臺股期貨』區段的三列，倒數第 2 個數字為未平倉淨額口數。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if day.strftime("%Y/%m/%d") not in soup.get_text(" ", strip=True):
        return None
    rec = {"日期": pd.Timestamp(day)}
    for table in soup.find_all("table"):
        active = False
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            line = " ".join(cells)
            if "臺股期貨" in line or "台股期貨" in line:
                active = True
            elif active and any(w in line for w in ("電子期貨", "小型臺指", "小型台指", "金融期貨")):
                active = False
            if not active:
                continue
            who = next((_group(c) for c in cells if _group(c) in ("外資", "投信", "自營")), None)
            nums = [_num(c) for c in cells if np.isfinite(_num(c))]
            if who and len(nums) >= 12:
                rec[who] = nums[-2]
    if all(k in rec for k in ("外資", "投信", "自營")):
        rec["合計"] = rec["外資"] + rec["投信"] + rec["自營"]
        return rec
    return None


def futures_institutions(days: int = 10):
    """回傳 (df, 來源說明, 錯誤列表)。"""
    s = requests.Session()
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2 + 6)
    errors = []
    try:
        r = s.post(TAIFEX_OI_CSV, data={"queryStartDate": start.strftime("%Y/%m/%d"),
                                        "queryEndDate": end.strftime("%Y/%m/%d"),
                                        "commodityId": "TXF"}, headers=UA, timeout=15)
        r.raise_for_status()
        df = parse_taifex_oi_csv(_decode(r.content))
        if df is not None and len(df):
            return df.tail(days), "期交所 區分各期貨契約（CSV）", errors
        errors.append("期交所 CSV 無臺股期貨資料")
    except Exception as e:  # noqa: BLE001
        errors.append(f"期交所 CSV：{str(e)[:80]}")
    rows = []
    for k in range(days * 2 + 6):
        d = end - dt.timedelta(days=k)
        if d.weekday() >= 5:
            continue
        try:
            r = s.get(TAIFEX_OI_HTML, params={"queryType": "1", "doQuery": "1",
                                              "queryDate": d.strftime("%Y/%m/%d"), "commodityId": "TXF"},
                      headers=UA, timeout=15)
            r.raise_for_status()
            rec = parse_taifex_oi_html(r.text, d)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{d} HTML：{str(e)[:60]}")
            if len(errors) >= 5 and not rows:
                break
            continue
        if rec:
            rows.append(rec)
        if len(rows) >= days:
            break
        time.sleep(0.3)
    if rows:
        return pd.DataFrame(rows).sort_values("日期"), "期交所 區分各期貨契約（網頁）", errors
    return None, None, errors


# ------------------------------------------------------------------
# 3. 臺股期貨收盤價（近月、一般盤）
# ------------------------------------------------------------------
def parse_futures_price_csv(text: str) -> pd.Series | None:
    df = pd.read_csv(io.StringIO(text), index_col=False)
    df.columns = [str(c).strip() for c in df.columns]
    col = lambda *keys: next((c for c in df.columns if all(k in c for k in keys)), None)
    c_date, c_con, c_month, c_close, c_sess = col("日期"), col("契約"), col("到期"), col("收盤"), col("時段")
    if not (c_date and c_month and c_close):
        return None
    for c in (c_con, c_month, c_sess):
        if c:
            df[c] = df[c].astype(str).str.strip()
    if c_con:
        df = df[df[c_con] == "TX"]
    if c_sess:
        df = df[df[c_sess].str.contains("一般")]
    df = df[~df[c_month].str.contains("/|W", regex=True)]  # 排除價差與週契約
    df["_close"] = df[c_close].map(_num)
    df = df[np.isfinite(df["_close"])]
    if df.empty:
        return None
    df["_date"] = pd.to_datetime(df[c_date].astype(str).str.strip())
    near = df.sort_values(["_date", c_month]).groupby("_date").first()
    return near["_close"]


def futures_close(days: int = 10):
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2 + 6)
    try:
        r = requests.post(TAIFEX_PRICE_CSV, data={"down_type": "1", "commodity_id": "TX",
                                                  "queryStartDate": start.strftime("%Y/%m/%d"),
                                                  "queryEndDate": end.strftime("%Y/%m/%d")},
                          headers=UA, timeout=15)
        r.raise_for_status()
        s = parse_futures_price_csv(_decode(r.content))
        return s, None
    except Exception as e:  # noqa: BLE001
        return None, f"期貨收盤價：{str(e)[:80]}"


def taiex_close():
    try:
        import yfinance as yf
        h = yf.Ticker("^TWII").history(period="2mo", interval="1d", auto_adjust=False)
        s = h["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception:  # noqa: BLE001
        return None
