"""build_dataset.py — 把 optscnr 的訊號 log 轉成 jevlike 格式（P0）。

只讀 optscnr（本機 clone 或 GitHub raw，釘 commit），永不寫回。
輸出：
  data/opt/<variant>/<task>.jsonl   variant ∈ {mask_crit, mask_nocrit, nomask_crit, nomask_nocrit}
  data/opt/features.jsonl           每筆的結構欄位 + gold + split + multiples（LR 基線與 EV 用）
  data/opt/MANIFEST.md

gold（凍結）：
  hit  = 任一檢查點 (t5/t10/t20) 的 opt_price / entry_price 峰值 ≥ 2.0（同 optscnr shadow_tracer.make_verdict）
  tier = A 峰值 ≥ 2 / B 1.2–2 / C < 1.2
  mature = 至少一個檢查點有價；EV 只在 t5/t10/t20 三點齊全的列上算（同 optscnr strategy_lab.multiples）
split（凍結）：cal = snapshot_date 2026-06/07/08；test = 2026-09（時間切分，不隨機）
state 只用訊號當下可見欄位；t5/t10/t20/path/verdict/sell_points 一律不進 state。

用法：python3 build_dataset.py [--optscnr /path/to/clone | --raw <commit>]
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "opt")
TASKS = json.load(open(os.path.join(ROOT, "tasks", "opt_tasks.json"), encoding="utf-8"))
TASKS = {k: v for k, v in TASKS.items() if not k.startswith("_")}
SIGNAL_FILES = ["data/iv_log/signals_2026-06.json", "data/iv_log/signals_2026-07.json",
                "data/iv_log/signals_2026-08.json", "data/iv_log/signals_2026-09.json"]
CAL_MONTHS = ("2026-06", "2026-07", "2026-08")
TEST_MONTHS = ("2026-09",)
VARIANTS = [("mask", "crit"), ("mask", "nocrit"), ("nomask", "crit"), ("nomask", "nocrit")]
# 同 optscnr strategy_lab（純函式，複製一份避免 import 到它的 yfinance 相依）
TIER_EDGES = (1.5, 3.0)
IV_EDGE = 50.0
FEATURE_KEYS = {"🚨異常掃貨": "sweep", "🚀點火": "ignition", "🆕新倉暴量": "first_seen_in_feed", "🚀突發暴量": "burst_no_history",
                "🆕低基期": "low_base", "🔭LEAPS": "leaps", "🚬菸屁股": "cigar_butt", "🔥萬人塚": "mass_grave", "🎲末日結算": "gamble"}
FEATURE_ZH = {v: k for k, v in FEATURE_KEYS.items()}
TIER_ZH = {"lottery": "樂透（< $1.5）", "mid": "中間（$1.5–3）", "heavy": "實彈（≥ $3）"}
OI_STATUS_ZH = {"confirmed": "已確認淨增", "unconfirmed": "未確認", "new": "新出現在篩選表", "flat": "持平", "down": "淨減"}


def tier_of(p):
    return None if p is None else ("lottery" if p < TIER_EDGES[0] else "mid" if p < TIER_EDGES[1] else "heavy")


def split_tags(tags):
    out = {"features": [], "warnings": [], "events": [], "ignition_x": None}
    for tok in str(tags or "").split():
        if tok.startswith("⚠️"):
            out["warnings"].append(tok)
        elif tok.startswith("📅"):
            out["events"].append(tok)
        else:
            key = next((v for k, v in FEATURE_KEYS.items() if tok.startswith(k)), None)
            out["features"].append(key or f"other:{tok}")
            m = re.search(r"\(([>0-9.]+)x\)", tok)
            if key == "ignition" and m:
                out["ignition_x"] = float(m.group(1).lstrip(">"))
    return out


def structural_pass(score, dte, iv, otm, oi_d7):
    try:
        return int(score) >= 8 and 21 <= int(dte) <= 120 and float(iv) < IV_EDGE and float(otm) < 25.0 and float(oi_d7) > 0
    except (TypeError, ValueError):
        return False


def multiples(sig):
    e = sig.get("entry_price") or 0
    if e <= 0:
        return None
    out = []
    for k in ("t5", "t10", "t20"):
        r = sig.get(k)
        if not r or r.get("opt_price") is None:
            return None
        out.append(r["opt_price"] / e)
    return out


def peak_loose(sig):
    e = sig.get("entry_price") or 0
    ms = [(sig.get(k) or {}).get("opt_price") for k in ("t5", "t10", "t20")]
    ms = [m / e for m in ms if m is not None and e > 0]
    return max(ms) if ms else None


def load_signals(src, raw_commit):
    rows = []
    for rel in SIGNAL_FILES:
        if raw_commit:
            url = f"https://raw.githubusercontent.com/clarencechien/optscnr/{raw_commit}/{rel}"
            data = json.load(io.TextIOWrapper(urllib.request.urlopen(url, timeout=60), encoding="utf-8"))
        else:
            p = os.path.join(src, rel)
            if not os.path.exists(p):
                continue
            data = json.load(open(p, encoding="utf-8"))
        rows += data if isinstance(data, list) else data.get("signals", [])
    return rows


def fnum(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def render_state(sig, mask):
    sd, exp = sig["snapshot_date"], sig["expiry"]
    dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.strptime(sd, "%Y-%m-%d")).days
    spot = sig.get("entry_spot") or 0
    otm = (sig["strike"] / spot - 1) * 100 if spot > 0 else None
    tg = split_tags(sig.get("tags"))
    feats = sig.get("features") if sig.get("features") is not None else tg["features"]
    warns = sig.get("warnings") if sig.get("warnings") else tg["warnings"]
    events = sig.get("events") if sig.get("events") else tg["events"]
    if mask:
        events = [re.sub(r"\([0-9\-/]+\)", "（到期前）", e) for e in events]
    L = []
    L.append(f"標的：{'股票X（已隱藏）' if mask else sig['ticker']}")
    if mask:
        L.append(f"合約：價外買權，距到期 {dte} 天")
    else:
        L.append(f"合約：{sig['ticker']} {exp} 買權，訊號日 {sd}，距到期 {dte} 天")
    L.append(f"履約價 {sig['strike']}，現貨 {fnum(spot, 2)}，價外 {fnum(otm)}%")
    px = f"權利金 ${sig['entry_price']:.2f}"
    if sig.get("entry_bid") is not None and sig.get("entry_ask") is not None:
        px += f"（bid {sig['entry_bid']:.2f} / ask {sig['entry_ask']:.2f}）"
    L.append(px + f"，等級：{TIER_ZH.get(sig.get('premium_tier') or tier_of(sig['entry_price']), '—')}")
    L.append(f"IV {fnum(sig.get('entry_iv'))}%")
    oi_line = f"未平倉 OI {sig.get('oi')}，七日變化 {sig.get('oi_d7')}"
    if sig.get("oi_delta_status"):
        oi_line += f"（{OI_STATUS_ZH.get(sig['oi_delta_status'], sig['oi_delta_status'])}）"
    L.append(oi_line)
    if sig.get("volume") is not None:
        L.append(f"當日成交量 {sig['volume']}")
    L.append(f"掃描分數 {sig.get('score')}，標籤：{sig.get('tags') or '—'}")
    L.append("特徵：" + ("、".join(FEATURE_ZH.get(f, f) for f in feats) if feats else "無"))
    if sig.get("signal_day_underlying_move") is not None:
        L.append(f"訊號日現貨漲跌 {sig['signal_day_underlying_move']:+.1f}%")
    L.append(f"訊號日有新聞：{'是' if sig.get('news_at_signal') else '否'}")
    L.append("到期前排定事件：" + ("、".join(events) if events else "無"))
    L.append("警告：" + ("、".join(warns) if warns else "無"))
    return "\n".join(L), {"dte": dte, "otm_pct": otm, "features": feats, "n_warnings": len(warns), "has_event": bool(events),
                          "ignition_x": tg["ignition_x"]}


def question_for(task, crit):
    t = TASKS[task]
    opts = {k: ({"label": v["label"], "criteria": v["criteria"]} if crit else v["label"]) for k, v in t["options"].items()}
    return {"type": t["kind"], "instructions": t["instructions"], "options": opts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optscnr", default=os.path.join(os.path.dirname(ROOT), "clarencechien", "optscnr"))
    ap.add_argument("--raw", default=None, help="GitHub raw at this optscnr commit instead of a local clone")
    a = ap.parse_args()
    sigs = load_signals(a.optscnr, a.raw)
    src_desc = f"raw@{a.raw}" if a.raw else a.optscnr
    rows_feat, per_variant = [], {(m, c): {t: [] for t in TASKS} for m, c in VARIANTS}
    n_skip = 0
    for s in sigs:
        pk = peak_loose(s)
        mo = s["snapshot_date"][:7]
        if pk is None or mo not in CAL_MONTHS + TEST_MONTHS:
            n_skip += 1
            continue
        split = "cal" if mo in CAL_MONTHS else "test"
        hit = pk >= 2.0
        tier_gold = "A" if pk >= 2.0 else ("B" if pk >= 1.2 else "C")
        gold = {"opt_play": "A" if hit else "B", "opt_tier": tier_gold}
        m3 = multiples(s)
        st_mask, meta = render_state(s, mask=True)
        st_nomask, _ = render_state(s, mask=False)
        sid = s["signal_id"]
        rule_b = bool(s["structural_pass"]) if s.get("structural_pass") is not None else structural_pass(
            s.get("score"), meta["dte"], s.get("entry_iv"), meta["otm_pct"], s.get("oi_d7"))
        rows_feat.append({"id": sid, "snapshot_date": s["snapshot_date"], "split": split, "ticker": s["ticker"], "event": f"{s['ticker']}_{s['snapshot_date']}",
                          "hit": hit, "peak": pk, "tier_gold": tier_gold, "multiples": m3, "rule_b": rule_b,
                          "entry_price": s["entry_price"], "entry_iv": s.get("entry_iv"), "entry_spot": s.get("entry_spot"), "strike": s["strike"],
                          "dte": meta["dte"], "otm_pct": meta["otm_pct"], "oi": s.get("oi"), "oi_d7": s.get("oi_d7"), "volume": s.get("volume"),
                          "score": s.get("score"), "premium_tier": s.get("premium_tier") or tier_of(s["entry_price"]), "features": meta["features"],
                          "n_warnings": meta["n_warnings"], "has_event": meta["has_event"], "ignition_x": meta["ignition_x"],
                          "news_at_signal": bool(s.get("news_at_signal")), "underlying_move": s.get("signal_day_underlying_move"),
                          "checkpoints": sum((s.get(k) or {}).get("opt_price") is not None for k in ("t5", "t10", "t20"))})
        for (m, c) in VARIANTS:
            for t in TASKS:
                per_variant[(m, c)][t].append({
                    "id": sid, "task": t, "state": st_mask if m == "mask" else st_nomask, "gold": gold[t],
                    "difficulty": "easy", "hard_type": None, "lang": "zh", "pair_id": f"{s['ticker']}_{s['snapshot_date']}",
                    "split": split, "source": "optscnr", "question": question_for(t, c == "crit")})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "features.jsonl"), "w", encoding="utf-8") as f:
        for r in rows_feat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for (m, c), tasks in per_variant.items():
        d = os.path.join(OUT, f"{m}_{c}")
        os.makedirs(d, exist_ok=True)
        for t, rows in tasks.items():
            with open(os.path.join(d, f"{t}.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows_feat)
    cal = [r for r in rows_feat if r["split"] == "cal"]; te = [r for r in rows_feat if r["split"] == "test"]
    def stats(R):
        rb = [r for r in R if r["rule_b"]]
        m3 = [r for r in R if r["multiples"]]
        return (f"n={len(R)}，命中 {sum(r['hit'] for r in R) / max(1, len(R)):.3f}；規則 B n={len(rb)}，命中 {sum(r['hit'] for r in rb) / max(1, len(rb)):.3f}；"
                f"三點齊全 n={len(m3)}；tier A/B/C = {sum(r['tier_gold'] == 'A' for r in R)}/{sum(r['tier_gold'] == 'B' for r in R)}/{sum(r['tier_gold'] == 'C' for r in R)}")
    man = [
        "# data/opt — 由 optscnr 訊號 log 建構（build_dataset.py）", "",
        f"來源：`{src_desc}`，檔案 {', '.join(SIGNAL_FILES)}；讀到 {len(sigs)} 筆，跳過（無檢查點或月份外）{n_skip} 筆，保留 {n} 筆。", "",
        "## 凍結的定義", "",
        "- gold `hit`：t5/t10/t20 任一檢查點 opt_price / entry_price 峰值 ≥ 2.0（同 optscnr `shadow_tracer.make_verdict`）。",
        "- gold `tier`：A 峰值 ≥ 2 / B 1.2–2 / C < 1.2。",
        "- EV：只在三點齊全（`multiples` 非空）的列上，用 optscnr `strategy_lab` 的六種出場政策算。",
        f"- split：cal = {'/'.join(CAL_MONTHS)}，test = {'/'.join(TEST_MONTHS)}（時間切分）。",
        "- state 只用訊號當下欄位；`mask` 變體遮 ticker、日曆日與事件日期，只給 DTE。",
        "- 注意：九月晚期的列只有 t5（或 t10）檢查點，峰值上限較低，各方法一體適用。", "",
        "## 樣本", "",
        f"- cal：{stats(cal)}", f"- test：{stats(te)}", "",
        "## 變體", "", "| 目錄 | 遮蔽 | 判準 |", "|---|---|---|",
    ] + [f"| `{m}_{c}` | {'遮 ticker/日期' if m == 'mask' else '不遮'} | {'選項帶 playbook 判準' if c == 'crit' else '只有 label'} |" for m, c in VARIANTS] + [
        "", "## state 範例（mask）", "", "```", per_variant[("mask", "crit")]["opt_play"][0]["state"], "```", "",
        "## question 範例（crit）", "", "```json", json.dumps(question_for("opt_play", True), ensure_ascii=False, indent=1), "```"]
    open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(man) + "\n")
    print(f"kept {n} (cal {len(cal)}, test {len(te)}), skipped {n_skip}; wrote {OUT}")
    print("cal:", stats(cal)); print("test:", stats(te))


if __name__ == "__main__":
    main()
