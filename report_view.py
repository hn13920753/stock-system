"""
單檔完整報告（畫面）。股票系統 App 與產業掃描 App 共用同一份（兩個倉庫的 report_view.py 要保持相同）。
不含交易邏輯；所有判斷都來自 engine.Analyzer.report()。
"""
import pandas as pd
import streamlit as st

import engine as E


def money(x):
    return f"{x:,.0f}"


def price(x):
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def render_report(r, an, df, src, target, hold, title=None):
    code = title or r["code"]
    # ---------------- 標頭 ----------------
    st.markdown(f"**{code}**　收盤 **{price(r['close'])}**（{r['chg']:+.2f}%）・量 {money(r['volume'])} 張・{r['date']}")
    st.caption(f"資料：{src}・{len(df)} 根日K（{df.index[0]:%Y-%m-%d} 起）"
               + (f"・股性 ATR {r['atrp']:.1f}%（本策略適用 {E.TH['atr_min']}～{E.TH['atr_max']}%）"
                  if r.get("atrp") else ""))
    if r["stale_days"] > 5:
        st.warning(f"最新資料距今 {r['stale_days']} 天，可能不是最新行情。")

    # ---------------- 結論 ----------------
    color = {"green": "v-green", "orange": "v-orange", "red": "v-red"}[r["color"]]
    if r["decision"].startswith("可進場"):
        sg = r["signal"]
        sub = sg["type"] + "・" + ("型態：" + "、".join(E.PATTERNS[k] for k in sg["ids"]) if sg["ids"]
                                   else sg.get("note", ""))
    elif r["why"]:
        sub = "；".join(r["why"][:2])
    else:
        sub = "目前沒有符合規則的進場訊號"
    st.markdown(f'<div class="verdict {color}"><div class="big">{r["decision"]}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    s = r["stats"]
    k1, k2 = st.columns(2)
    if s["n"] == 0:
        win_val, win_note = "—", "本股歷史上沒有出現過同類訊號"
    elif s["n"] < 10:
        win_val, win_note = "樣本不足", f"只有 {s['n']} 筆，不計算勝率"
    else:
        win_val = f"{s['tp_rate']:.0f}%"
        win_note = (f"{s['n']} 筆中先到停利 {round(s['tp_rate'] * s['n'] / 100)} 筆"
                    + ("" if r["enough"] else "・樣本 < 30，僅供參考"))
    k1.markdown(f'<div class="kpi"><div class="lab">歷史回測勝率（{hold} 日內先到 +{target:g}%）</div>'
                f'<div class="val">{win_val}</div><div class="note">{win_note}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="lab">規則符合度（不是勝率）</div>'
                f'<div class="val">{r["fit"]:.0f}%</div><div class="note">10 模組中可判定者的平均</div></div>',
                unsafe_allow_html=True)
    if s["n"] >= 10:
        st.caption(f"回測平均每筆淨報酬 {s['avg']:+.2f}%・平均獲利 {s['avg_win']:+.2f}%／平均虧損 {s['avg_loss']:+.2f}%"
                   f"・最多連虧 {s['max_streak']} 筆・平均持有 {s['avg_days']:.1f} 天")

    # ---------------- 三項過濾（測試中）----------------
    f = r.get("filters") or {}
    flags = r.get("filter_flags") or {}
    if f:
        on = [v[0] for k, v in E.FILTERS.items() if flags.get(k)]
        st.caption("🧪 過濾（測試中）："
                   + ("；".join(f.get("info", [])) or "無資料")
                   + (f"。啟用：{'、'.join(on)}" if on else "。三項過濾都已關閉"))

    # ---------------- 交易計畫 / 下一步 ----------------
    pl = r["plan"]
    if r["decision"].startswith("可進場") and pl:
        st.markdown("#### 交易計畫")
        lots, odd = divmod(pl["shares"], 1000)
        qty = (f"{lots} 張" if lots else "") + (f" {odd} 股" if odd else "") or "0 股"
        half = ("・已減半：" + "、".join(pl["halve"])) if pl.get("halve") else ""
        rows = [
            (("限價掛單", f"{price(pl['entry'])}（{r['signal'].get('days_left', 3)} 個交易日內有效；開盤低於此價就以開盤價成交；"
                          f"未成交前收盤跌破 {price(pl['stop'])} 就取消）")
             if r["signal"].get("mode") == "limit" else
             ("買進價", f"約 {price(pl['entry'])}（明日開盤高於 {price(E.round_up(pl['entry'] * 1.01, r['etf']))} 就不追）")),
            ("買進數量", f"{qty}（約 {money(pl['cost'])} 元）{half}"),
            ("停利", f"{price(pl['tp3'])}（淨 +{pl.get('target', target):g}% 全出）；遇壓力可改 {price(pl['tp2'])}（淨 +{pl.get('tp2_pct', 2.0):g}%）"),
            ("停損", f"{price(pl['stop'])}（碰到即出，建議設觸價停損單；淨 {pl['loss_pct']:.2f}%）"),
            ("最大損失", f"約 {money(pl['max_loss'])} 元"),
            ("風險報酬比", f"{pl['rr']:.2f}"),
            ("前方壓力", price(pl["resistance"]) if pl["resistance"] else "近 120 日內無"),
            ("最晚出場", f"第 {hold} 個交易日收盤"),
            ("撤退條件", "跌破停損、爆量長黑、跌回突破價之下（假突破）"),
        ]
        if pl.get("aggr"):
            st.info(f"⚡ 積極組（測試中）：高波動個股，停損放寬到 2 ATR、淨利 {pl.get('target', 8):g}% 出場，"
                    "股數依「單筆最大損失」反推。尚未證實有優勢，正在紙上交易驗證。")
        st.table(pd.DataFrame(rows, columns=["項目", "內容"]).set_index("項目"))
        if pl["shares"] == 0:
            st.warning("依目前資金與最大損失設定，買不到 1 股，請調整進階設定。")
    else:
        st.markdown("#### 下一步")
        if r["trigger"]:
            st.write("**觸發條件：** " + r["trigger"]["text"])
        st.caption("觸發當天收盤後再分析一次，系統會算出停損、數量與停利。")

    # ---------------- 理由 / 風險 / 衝突 ----------------
    st.markdown("#### 支持進場")
    for x in r["pros"] or ["（無）"]:
        st.markdown(f"- {x}")
    st.markdown("#### 主要風險")
    for x in (r["cons"] + [w for w in r["why"] if w not in " ".join(r["cons"])])[:3] or ["（無）"]:
        st.markdown(f"- {x}")
    if r["conflicts"]:
        st.markdown("#### 衝突訊號與裁決")
        for x in r["conflicts"]:
            st.markdown(f"- {x}")

    # ---------------- 10 模組 ----------------
    with st.expander("10 模組核對"):
        mark = {1.0: "✅", 0.5: "🟡", 0.0: "❌", None: "⚪"}
        for m in r["modules"]:
            st.markdown(f"{mark[m['score']]} **{m['no']} {m['name']}**　{m['text']}")
        st.caption("✅ 支持　🟡 中性／部分　❌ 反對　⚪ 無法判定（不計入符合度）")

    # ---------------- 18 型態 ----------------
    with st.expander("18 種空轉多（近 10 日）"):
        if r["recent_hits"]:
            for d, h in r["recent_hits"]:
                st.markdown(f"- {d}　**#{h['id']} {h['name']}**：{h['note']}")
        else:
            st.write("近 10 日沒有自動判定到的型態。")
        st.caption("#13 雙盤底、#14 雙弧底、#16 突破ABC 尚未自動化，請人工看圖。量化門檻為測試中。")

    # ---------------- 回測 ----------------
    with st.expander("回測明細"):
        fn = r.get("funnel") or {}
        if fn.get("出現訊號"):
            st.markdown("**訊號漏斗**（每一關擋掉幾次）")
            st.table(pd.DataFrame([(k, v) for k, v in fn.items() if v or k in ("出現訊號", "成交")],
                                  columns=["關卡", "次數"]).set_index("關卡"))
        bt = r["bt"]
        if bt is None or bt.empty:
            st.write("本股資料期間內沒有符合規則的歷史進場。")
        else:
            st.write(f"期間 {s['first']} ～ {s['last']}，共 {s['n']} 筆；先停利 {s['tp_rate']:.0f}%、"
                     f"先停損 {s['sl_rate']:.0f}%、淨報酬為正 {s['win_rate']:.0f}%；最大不利波動 {s['worst_mae']:.1f}%。")
            cols = [c for c in ["訊號日", "買點", "進場價", "停損", "停利", "出場日", "結果", "淨報酬%", "持有天數",
                                "停損/ATR", "大盤", "型態"] if c in bt.columns]
            st.dataframe(bt[cols].iloc[::-1], hide_index=True, use_container_width=True)
        st.caption("方法：每天收盤後判斷訊號。收盤型買點隔日開盤進場，開盤高於前收 1% 不追、漲停鎖死買不到；"
                   "掛單型買點 3 個交易日內觸價成交，成交當天只算停損不算停利。逐日檢查，同一天同時碰到停損與停利先算停損；"
                   "跳空跌破停損以開盤價計算；持有期滿以收盤出場。已扣手續費與證交稅，未含滑價。過去結果不保證未來。")

    # ---------------- K 線圖 ----------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        d = an.df.iloc[-120:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                     increasing_line_color="#e53935", decreasing_line_color="#2e9d5b",
                                     name="K線"), row=1, col=1)
        for col, clr in (("ma20", "#8e44ad"), ("ma60", "#e67e22")):
            fig.add_trace(go.Scatter(x=d.index, y=d[col], line=dict(color=clr, width=1.2),
                                     name=col.upper()), row=1, col=1)
        vc = ["#e53935" if c >= o else "#2e9d5b" for c, o in zip(d["close"], d["open"])]
        fig.add_trace(go.Bar(x=d.index, y=d["volume"], marker_color=vc, name="量"), row=2, col=1)
        if pl and r["decision"].startswith("可進場"):
            for y, clr, nm in ((pl["tp3"], "#e53935", "停利"), (pl["stop"], "#2e9d5b", "停損")):
                fig.add_hline(y=y, line_dash="dot", line_color=clr, annotation_text=nm, row=1, col=1)
        fig.update_layout(height=420, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                          xaxis_rangeslider_visible=False, dragmode=False)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:  # noqa: BLE001
        st.line_chart(an.df["close"].iloc[-120:])
