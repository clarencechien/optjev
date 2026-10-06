"""E3 / E4 analysis for v12 (gates in docs/handoff-jevlike-v12.md §4–§5, written before the runs). Called by analyze_v12.py."""
import json
import math
import os

import numpy as np

from analyze_v12 import ACC, EPS, JEV, STRONG, TASKS, V12, WEAK, f, load_task, mcnemar_p, nll_T, GRID
from analyze import ece, group_split, load_jsonl, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

LAT_26B_L4 = os.path.join(JEV, "results/modal/latency/latency_swafull.json")          # b11118, L4, --swa-full
LAT_26B_L40S = os.path.join(JEV, "results/modal/v11/latency-26b/latency_b11371_l40s.json")


def fit_T(Z, y):
    return float(GRID[int(np.argmin([nll_T([Z], [y], T) for T in GRID]))])


def d0(model_dir):
    """per task: dict id -> (calibrated P, gold idx, split) with T fitted on that model's cal half."""
    out = {}
    for t in TASKS:
        got = load_task(model_dir, t)
        if not got:
            return None
        rows, letters, Z, y = got
        ca = group_split(rows, 0)
        T = fit_T(Z[ca], y[ca]); P = softmax(Z / T)
        out[t] = {"ids": [r["id"] for r in rows], "P": P, "y": y, "ca": ca, "T": T,
                  "acc_test": float((P.argmax(1)[~ca] == y[~ca]).mean()), "mass_p5": float(np.percentile([r.get("option_mass") or 1 for r in rows], 5))}
    return out


def cascade(first, big, max_send=0.25):
    """first answers; rows whose calibrated confidence is below tau go to big. tau = the max_send quantile of the pooled
    cal confidences of `first` (chosen on cal), applied to test."""
    conf_cal = np.concatenate([first[t]["P"].max(1)[first[t]["ca"]] for t in TASKS])
    tau = float(np.quantile(conf_cal, max_send))
    ok, sent, n = 0, 0, 0
    per = {}
    for t in TASKS:
        a, b = first[t], big[t]
        bi = {i: k for k, i in enumerate(b["ids"])}
        te = ~a["ca"]
        idx = np.where(te)[0]
        s_ok = s_sent = 0
        for k in idx:
            c = a["P"][k].max()
            if c < tau:
                kb = bi[a["ids"][k]]; good = b["P"][kb].argmax() == b["y"][kb]; s_sent += 1
            else:
                good = a["P"][k].argmax() == a["y"][k]
            s_ok += int(good)
        per[t] = {"acc": s_ok / len(idx), "sent": s_sent / len(idx)}
        ok += s_ok; sent += s_sent; n += len(idx)
    return {"tau": tau, "acc": ok / n, "sent": sent / n, "per_task": per}


def mean_acc(m, tasks):
    return float(np.mean([m[t]["acc_test"] for t in tasks]))


def jevbench_acc(d):
    n = ok = 0
    for tier in ("original", "easy", "hard"):
        for r in load_jsonl(os.path.join(d, f"{tier}.jsonl")):
            if "fwd" in r:
                n += 1; ok += bool(r.get("correct_fwd"))
    return (ok / n if n else None), n


def latency(p):
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    return {"single_p50": r["L1_single_100tok_2opt"]["summary"]["p50"], "shared10_per_q": r["L4_shared_state_10q_cache_on"]["per_call"]["p50"],
            "decode_floor": r.get("L1_ref_same_prompt_repeated", {}).get("summary", {}).get("p50")}


def d2_acc(d):
    accs = {}
    for t in TASKS:
        rows = [r for r in load_jsonl(os.path.join(d, f"{t}.jsonl")) if "correct" in r]
        if rows:
            accs[t] = sum(r["correct"] for r in rows) / len(rows)
    return float(np.mean(list(accs.values()))) if accs else None


def run_e3():
    m26, e4b = d0(ACC), d0(os.path.join(ACC, "e4b/D0"))
    res = {"26b": {"d0": mean_acc(m26, TASKS), "weak": mean_acc(m26, WEAK), "d2": d2_acc(os.path.join(ACC, "D2")), "jevbench": jevbench_acc(os.path.join(JEV, "results/modal/jevbench"))[0],
                   "lat_L4": latency(LAT_26B_L4), "lat_L40S": latency(LAT_26B_L40S)}}
    if e4b:
        res["e4b"] = {"d0": mean_acc(e4b, TASKS), "weak": mean_acc(e4b, WEAK), "cascade": cascade(e4b, m26)}
    for name in ("12b", "31b"):
        m = d0(os.path.join(V12, "d0", name))
        if not m:
            continue
        jb_dir = os.path.join(V12, "jevbench-1slot", name)  # 31B rerun with one 8192-token slot (36 long items overflowed 2048)
        jb, jn = jevbench_acc(jb_dir if os.path.isdir(jb_dir) else os.path.join(V12, "jevbench", name))
        res[name] = {"d0": mean_acc(m, TASKS), "weak": mean_acc(m, WEAK), "strong": mean_acc(m, STRONG), "per_task": {t: m[t]["acc_test"] for t in TASKS},
                     "T": {t: m[t]["T"] for t in TASKS}, "mass_p5_min": min(m[t]["mass_p5"] for t in TASKS),
                     "d2": d2_acc(os.path.join(V12, "d2", name)), "jevbench": jb, "jevbench_n": jn,
                     "latency": latency(os.path.join(V12, "latency", name, "latency.json")),
                     "mcnemar_vs_26b": {t: mcnemar_p(m[t]["P"].argmax(1)[~m[t]["ca"]] == m[t]["y"][~m[t]["ca"]],
                                                     m26[t]["P"].argmax(1)[~m26[t]["ca"]] == m26[t]["y"][~m26[t]["ca"]])[0] for t in TASKS}}
        if name == "12b":
            res[name]["cascade"] = cascade(m, m26)
    v = {}
    if "12b" in res:
        c = res["12b"]["cascade"]
        v["12b_first_stage"] = c["acc"] >= res["26b"]["d0"] - 0.003 and c["sent"] <= 0.25 + 1e-9
        v["12b_cygnet_sanity"] = res["12b"]["jevbench"] is not None and abs(res["12b"]["jevbench"] - 0.879) <= 0.02
    if "31b" in res:
        v["31b_weak_candidate"] = res["31b"]["weak"] >= res["26b"]["weak"] + 0.03 and res["31b"]["d0"] >= res["26b"]["d0"]
    res["gates"] = v
    res["verdict"] = {"12b": ("取代 E4B 當第一階" if v.get("12b_first_stage") else "不取代") if "12b" in res else "未跑",
                      "31b": ("列為弱題候選" if v.get("31b_weak_candidate") else "不列") if "31b" in res else "未跑"}
    L = ["# 17-E3 — 階梯補 Gemma 4 12B 與 31B", "",
         "llama.cpp b11371（jevlike v11 編的同一份），`-np 4 --jinja --reasoning-budget 0 --swa-full`；12B Q8_0 在 L4（ctx 16384），31B Q8_0 在 L40S（ctx 8192，16384 加 swa-full 的 KV 12.8 GB 放不下）。"
         "26B 對照用 jevlike 既有 D0（UD-Q4_K_M、b11118，v11 已驗 b11371 1,998/2,000 同答）。每個模型各自在 cal 半擬每類溫度。", "",
         f"## 判定：12B **{res['verdict']['12b']}**；31B **{res['verdict']['31b']}**", ""]
    if "12b" in res:
        c, r = res["12b"]["cascade"], res["12b"]
        L += [f"- 12B 級聯（12B 先答，cal 上 25% 分位的校準信心以下送 26B）：test acc {c['acc']:.3f}、送 26B {c['sent']:.0%}（門檻：≥ 26B {res['26b']['d0']:.3f} − 0.003 且 ≤ 25%：{'✓' if v['12b_first_stage'] else '✗'}）。"
              + (f"同規則的 E4B 級聯：acc {res['e4b']['cascade']['acc']:.3f}、送 {res['e4b']['cascade']['sent']:.0%}。" if "e4b" in res else ""),
              f"- 12B 重現 Cygnet（健全性，不判定）：JevBench 公開 fwd {f(r['jevbench'])}（{r['jevbench_n']} 題），Cygnet 自報 0.879 ± 0.02：{'✓' if v['12b_cygnet_sanity'] else '✗'}。"]
    if "31b" in res:
        r = res["31b"]
        L += [f"- 31B 弱題三類 test 平均 {r['weak']:.3f} vs 26B {res['26b']['weak']:.3f}（門檻 +0.03）；十類平均 {r['d0']:.3f} vs 26B {res['26b']['d0']:.3f}（門檻 ≥）→ {'✓' if v['31b_weak_candidate'] else '✗'}。"]
    if "12b" in res and res["12b"].get("latency") and res["26b"].get("lat_L4"):
        l12, l26 = res["12b"]["latency"]["single_p50"], res["26b"]["lat_L4"]["single_p50"]
        L += [f"- **速度但書（門檻沒寫到，照實記）**：同一張 L4 上 12B 單題 p50 {l12:.0f} ms，26B {l26:.0f} ms。12B 是 dense，每個 token 算 12B；26B-A4B 每個 token 只算約 4B。"
              "所以 12B 當第一階在 L4 上不省延遲，只省一點記憶體（12.7 GB 對 17 GB）。準確率門檻照預登記判「取代」，實際要不要換，等 GB10 上的延遲。"]
    L += ["", "## D0 test（每類 100 筆）", "", "| task | E4B | 12B | 26B | 31B | 12B vs 26B p | 31B vs 26B p |", "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        row = [f(e4b[t]["acc_test"]) if e4b else "—", f(res.get("12b", {}).get("per_task", {}).get(t)), f(m26[t]["acc_test"]), f(res.get("31b", {}).get("per_task", {}).get(t)),
               f(res.get("12b", {}).get("mcnemar_vs_26b", {}).get(t)), f(res.get("31b", {}).get("mcnemar_vs_26b", {}).get(t))]
        L.append(f"| {t} | " + " | ".join(row) + " |")
    L.append(f"| **十類平均** | {f(res.get('e4b', {}).get('d0'))} | {f(res.get('12b', {}).get('d0'))} | {f(res['26b']['d0'])} | {f(res.get('31b', {}).get('d0'))} | | |")
    L.append(f"| **弱題三類** | {f(res.get('e4b', {}).get('weak'))} | {f(res.get('12b', {}).get('weak'))} | {f(res['26b']['weak'])} | {f(res.get('31b', {}).get('weak'))} | | |")
    L += ["", "## 其他尺與速度", "", "| 模型 | D2 盲寫平均 | JevBench 公開 fwd | 卡 | 單題 p50 | 共用 state 第 2 題起每題 | 字母總機率 p5 最低 |", "|---|---|---|---|---|---|---|"]
    l26 = res["26b"]
    L.append(f"| 26B | {f(l26['d2'])} | {f(l26['jevbench'])} | L4 / L40S | {f(l26['lat_L4']['single_p50'], 0)} / {f(l26['lat_L40S']['single_p50'], 0)} ms | "
             f"{f(l26['lat_L4']['shared10_per_q'], 0)} / {f(l26['lat_L40S']['shared10_per_q'], 0)} ms | — |")
    for name, card in (("12b", "L4"), ("31b", "L40S")):
        if name in res:
            r = res[name]; lat = r["latency"] or {}
            L.append(f"| {name.upper()} | {f(r['d2'])} | {f(r['jevbench'])}（{r['jevbench_n']} 題） | {card} | {f(lat.get('single_p50'), 0)} ms | {f(lat.get('shared10_per_q'), 0)} ms | {f(r['mass_p5_min'], 4)} |")
    return res, L


# ---------------------------------------------------------------- E4
def e4_task(rows, trigger):
    for r in rows:  # a stage-B call that failed (e.g. context overflow) leaves the row on stage A
        if r.get("B") is not None and "raw_logprobs" not in r["B"]:
            r["B_error"] = r["B"].get("error"); r["B"] = None
    letters = rows[0]["letters"]
    y = np.array([letters.index(r["gold"]) for r in rows])
    ZA = np.array([[r["A"]["raw_logprobs"].get(L) if r["A"]["raw_logprobs"].get(L) is not None else -30.0 for L in letters] for r in rows])
    ca = np.array([r["split"] == "cal" for r in rows]); te = ~ca
    TA = fit_T(ZA[ca], y[ca]); PA = softmax(ZA / TA); confA = PA.max(1)
    trig = np.array([(confA[i] < trigger) and (rows[i]["B"] is not None) for i in range(len(rows))])
    missingB = int(((confA < trigger) & np.array([r["B"] is None for r in rows])).sum())
    ZB = np.array([[(r["B"] or r["A"])["raw_logprobs"].get(L) if (r["B"] or r["A"])["raw_logprobs"].get(L) is not None else -30.0 for L in letters] for r in rows])
    m = ca & trig
    TB = fit_T(ZB[m], y[m]) if m.sum() >= 5 else TA
    PB = softmax(ZB / TB)
    P = np.where(trig[:, None], PB, PA)
    okA = PA.argmax(1) == y; okF = P.argmax(1) == y
    def cov(conf, ok):
        cut, _ = risk_threshold(conf[ca], ok[ca], EPS); h = conf[te] >= cut
        return float(h.mean()), (float((~ok[te][h]).mean()) if h.any() else 0.0)
    covA, errA = cov(confA, okA); covF, errF = cov(P.max(1), okF)
    B = [r["B"] for r, t in zip(rows, trig) if t]
    lat = [b["gen_ms"] + b["read_ms"] for b in B]
    return {"n_test": int(te.sum()), "TA": TA, "TB": TB, "trigger_rate_test": float(trig[te].mean()), "missing_B": missingB,
            "accA": float(okA[te].mean()), "accF": float(okF[te].mean()),
            "fixed": int((~okA & okF & te).sum()), "broken": int((okA & ~okF & te).sum()),
            "covA": covA, "covF": covF, "errA": errA, "errF": errF,
            "acc_triggered_A": float(okA[te & trig].mean()) if (te & trig).any() else None, "acc_triggered_F": float(okF[te & trig].mean()) if (te & trig).any() else None,
            "tokens_mean": float(np.mean([b["n_tokens"] for b in B])) if B else None, "truncated_rate": float(np.mean([b["truncated"] for b in B])) if B else None,
            "mass_B_median": float(np.median([b["option_mass"] or 0 for b in B])) if B else None,
            "lat_p50_ms": float(np.percentile(lat, 50)) if lat else None, "lat_p95_ms": float(np.percentile(lat, 95)) if lat else None,
            "okA": okA[te], "okF": okF[te]}


def run_e4():
    d = os.path.join(V12, "think_tail", "26b")
    res = {}
    for trigger in (0.7, 0.9):
        per = {}
        for t in WEAK:
            rows = load_jsonl(os.path.join(d, f"{t}.jsonl"))
            if rows:
                per[t] = e4_task(rows, trigger)
        if not per:
            return {"error": "no think_tail results"}, ["# 17-E4 — 結果尚未拉回"]
        okA = np.concatenate([per[t].pop("okA") for t in per]); okF = np.concatenate([per[t].pop("okF") for t in per])
        p, n01, n10 = mcnemar_p(okA, okF)
        pooled = {"n": int(len(okA)), "accA": float(okA.mean()), "accF": float(okF.mean()), "mcnemar_p": p, "fixed": n01, "broken": n10,
                  "covA": float(np.mean([per[t]["covA"] for t in per])), "covF": float(np.mean([per[t]["covF"] for t in per])),
                  "trigger_rate": float(np.mean([per[t]["trigger_rate_test"] for t in per]))}
        # JevBench public (fwd): T = jevlike v8 T_med, same trigger
        jb = load_jsonl(os.path.join(d, "jevbench.jsonl"))
        jA = jF = jn = jt = 0
        for r in jb:
            L = r["letters"]; yi = r["labels"].index(r["expected"])
            za = np.array([r["A"]["raw_logprobs"].get(x) if r["A"]["raw_logprobs"].get(x) is not None else -30.0 for x in L])
            a = int(za.argmax()); fa = a
            if r["A_conf_T"] < trigger and r["B"] and "raw_logprobs" in r["B"]:
                zb = np.array([r["B"]["raw_logprobs"].get(x) if r["B"]["raw_logprobs"].get(x) is not None else -30.0 for x in L]); fa = int(zb.argmax()); jt += 1
            jn += 1; jA += int(a == yi); jF += int(fa == yi)
        res[str(trigger)] = {"per_task": per, "pooled": pooled, "jevbench": {"n": jn, "accA": jA / jn if jn else None, "accF": jF / jn if jn else None, "triggered": jt}}
    # strong tasks' trigger rate from jevlike's existing 26B D0 (stage A is the same readout)
    strong_rate = {}
    for t in STRONG:
        rows, letters, Z, y = load_task(ACC, t)
        ca = group_split(rows, 0); T = fit_T(Z[ca], y[ca]); conf = softmax(Z / T).max(1)
        strong_rate[t] = float((conf[~ca] < 0.7).mean())
    res["strong_trigger_rate_0.7"] = strong_rate
    P7 = res["0.7"]["pooled"]
    g = {"acc_plus2": P7["accF"] >= P7["accA"] + 0.02, "cov_plus5": P7["covF"] >= P7["covA"] + 0.05, "broken_le_half_fixed": P7["broken"] <= P7["fixed"] / 2,
         "weak_trigger_le_30": P7["trigger_rate"] <= 0.30, "strong_trigger_le_5": max(strong_rate.values()) <= 0.05,
         "jevbench_not_worse": (res["0.7"]["jevbench"]["accF"] or 0) >= (res["0.7"]["jevbench"]["accA"] or 0)}
    adopt = g["acc_plus2"] and g["cov_plus5"] and g["broken_le_half_fixed"]
    verdict = "採用（弱題 opt-in 慢路徑）" if adopt else ("變差" if P7["accF"] < P7["accA"] - 0.01 or P7["broken"] > P7["fixed"] else "持平（不採用）")
    res["gates"] = g; res["verdict"] = verdict
    L = ["# 17-E4 — 26B 只在低信心時先想再答", "",
         "26B UD-Q4_K_M、llama.cpp b11371、L4，伺服器不帶 `--reasoning-budget 0`（會強制關思考）。A 段 = 現行讀法；B 段 = 思考開的模板，預填 `<|channel>thought\\n`，"
         "貪婪生成 ≤ 512 token、遇 `<channel|>` 停，補上關閉標記後讀字母。溫度每類在 cal 半擬（A 段 T_A；B 段在 cal 的觸發列上另擬 T_B）。步驟 0 格式檢查通過（見 §步驟 0）。", "",
         f"## 判定：**{verdict}**", "",
         f"- 主門檻（觸發 = 校準後信心 < 0.7）：弱題三類 test 合併 {P7['n']} 筆 acc {P7['accA']:.3f} → {P7['accF']:.3f}（+2 點：{'✓' if g['acc_plus2'] else '✗'}，McNemar p {P7['mcnemar_p']:.3f}）；"
         f"翻對 {P7['fixed']}、翻錯 {P7['broken']}（翻錯 ≤ 翻對一半：{'✓' if g['broken_le_half_fixed'] else '✗'}）；ε=5% 平均 coverage {P7['covA']:.2f} → {P7['covF']:.2f}（+5 點：{'✓' if g['cov_plus5'] else '✗'}）。",
         f"- 觸發率：弱題 {P7['trigger_rate']:.0%}（≤ 30%：{'✓' if g['weak_trigger_le_30'] else '✗'}）；強題最高 {max(strong_rate.values()):.0%}（≤ 5%：{'✓' if g['strong_trigger_le_5'] else '✗'}）。",
         f"- JevBench 公開 fwd：{res['0.7']['jevbench']['accA']:.3f} → {res['0.7']['jevbench']['accF']:.3f}（觸發 {res['0.7']['jevbench']['triggered']} 題；不變差：{'✓' if g['jevbench_not_worse'] else '✗'}）。", "",
         "## 逐類", "", "| 觸發 | task | test n | 觸發率 | acc A → 最終 | 觸發列 acc A → B | 翻對 / 翻錯 | ε=5% coverage A → 最終 | T_A / T_B | 平均思考 token | 截斷 | B 字母總機率中位數 | 觸發列延遲 p50 / p95 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for trig in ("0.7", "0.9"):
        for t, v in res[trig]["per_task"].items():
            L.append(f"| {trig} | {t} | {v['n_test']} | {v['trigger_rate_test']:.0%} | {v['accA']:.3f} → {v['accF']:.3f} | {f(v['acc_triggered_A'])} → {f(v['acc_triggered_F'])} | "
                     f"{v['fixed']} / {v['broken']} | {v['covA']:.2f} → {v['covF']:.2f} | {v['TA']:.2f} / {v['TB']:.2f} | {f(v['tokens_mean'], 0)} | {f(v['truncated_rate'], 2)} | "
                     f"{f(v['mass_B_median'], 4)} | {f(v['lat_p50_ms'] / 1000 if v['lat_p50_ms'] else None, 1)} / {f(v['lat_p95_ms'] / 1000 if v['lat_p95_ms'] else None, 1)} s |")
    P9 = res["0.9"]["pooled"]
    L += ["", f"次要觸發 0.9：合併 acc {P9['accA']:.3f} → {P9['accF']:.3f}，翻對 {P9['fixed']}、翻錯 {P9['broken']}，觸發率 {P9['trigger_rate']:.0%}；"
          f"JevBench {res['0.9']['jevbench']['accA']:.3f} → {res['0.9']['jevbench']['accF']:.3f}（觸發 {res['0.9']['jevbench']['triggered']} 題）。",
          "", "強題（jevlike 既有 26B D0，A 段同讀法）信心 < 0.7 的比例：" + "、".join(f"{t} {v:.0%}" for t, v in strong_rate.items()) + "。",
          "", "延遲是 L4 上 26B 的實測，只報不判（handoff §5）；GB10 的判定另寫。"]
    return res, L


def main(which, save):
    if which in ("e3", "all"):
        save("e3", *run_e3())
    if which in ("e4", "all"):
        save("e4", *run_e4())
