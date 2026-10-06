"""v12 analysis (docs/handoff-jevlike-v12.md). Gates were written before any number was computed.

  python3 v12/analyze_v12.py e1      # ordinal expected-level readout, offline
  python3 v12/analyze_v12.py e2      # temperature granularity, offline
  python3 v12/analyze_v12.py e3      # 12B / 31B rungs (reads results/modal/v12)
  python3 v12/analyze_v12.py e4      # think-on-the-tail (reads results/modal/v12/think_tail)
  python3 v12/analyze_v12.py all
Writes v12/results/v12.json and v12/results/17-v12-<exp>.md. Uses jevlike (submodule) code and committed results.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTJEV = os.path.dirname(HERE)
JEV = os.path.join(OPTJEV, "jevlike")
sys.path.insert(0, os.path.join(JEV, "bench"))
os.environ.setdefault("MPLBACKEND", "Agg")
from analyze import TASKS, ece, group_split, load_jsonl, logit_matrix, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

ACC = os.path.join(JEV, "results/modal/accuracy")
JB = os.path.join(JEV, "results/modal/jevbench")
V12 = os.path.join(OPTJEV, "results/modal/v12")
OUT = os.path.join(HERE, "results")
GRID = np.exp(np.linspace(math.log(0.05), math.log(50), 200))  # same grid as jevlike analyze.fit_temperature
KIND = {t: TASKS[t]["kind"] for t in TASKS}
WEAK = ["m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
STRONG = [t for t in TASKS if t not in WEAK]
EPS = 0.05


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def load_task(acc_dir, task):
    rows = [r for r in load_jsonl(os.path.join(acc_dir, f"{task}.jsonl")) if "raw_logprobs" in r]
    if not rows:
        return None
    letters = list(TASKS[task]["options"])
    Z = logit_matrix(rows, letters)
    y = np.array([letters.index(r["gold"]) for r in rows])
    return rows, letters, Z, y


def nll_T(Zs, ys, T):
    tot, n = 0.0, 0
    for Z, y in zip(Zs, ys):
        P = softmax(Z / T)
        tot += -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).sum(); n += len(y)
    return tot / n


def fit_T_pooled(Zs, ys):
    vals = [nll_T(Zs, ys, T) for T in GRID]
    return float(GRID[int(np.argmin(vals))])


def mcnemar_p(a_correct, b_correct):
    """Exact two-sided McNemar on discordant pairs."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p), n01, n10


# ---------------------------------------------------------------- E1
def readouts(P):
    """R0 argmax, R1 round(E) with .5 toward the lighter level (lower index), R2 median level."""
    K = P.shape[1]
    E = (P * np.arange(K)).sum(1)
    r1 = np.ceil(E - 0.5).astype(int)  # x.5 -> x (lighter); otherwise nearest
    r1 = np.clip(r1, 0, K - 1)
    r2 = (np.cumsum(P, 1) >= 0.5).argmax(1)
    return P.argmax(1), r1, r2, E


def e1_metrics(pred, y, conf=None, cut=None):
    ok = pred == y
    err = ~ok
    adj = (np.abs(pred - y) == 1) & err
    out = {"n": int(len(y)), "acc": float(ok.mean()), "mae": float(np.abs(pred - y).mean()),
           "adjacent_share_of_errors": float(adj.sum() / err.sum()) if err.any() else None}
    if conf is not None and cut is not None:
        h = conf >= cut
        out["coverage_eps5"] = float(h.mean()); out["error_eps5"] = float(err[h].mean()) if h.any() else 0.0
    return out


def run_e1():
    res = {"gate": "pooled severity acc R1 >= R0 + 2pt, MAE not up, no set worse by >2pt; JevBench score correct R1 >= R0 - 1"}
    sets = {}
    for task in ("m_alarm_severity", "q_spc_action"):
        rows, letters, Z, y = load_task(ACC, task)
        ca = group_split(rows, 0); te = ~ca
        T = float(GRID[int(np.argmin([nll_T([Z[ca]], [y[ca]], t) for t in GRID]))])
        Pc = softmax(Z[ca] / T)
        r0c, r1c, _, _ = readouts(Pc)
        cut0, _ = risk_threshold(Pc.max(1), r0c == y[ca], EPS)
        cut1, _ = risk_threshold(Pc[np.arange(len(r1c)), r1c], r1c == y[ca], EPS)
        cand = {"D0 test": (Z[te], y[te])}
        for name, sub in (("heldout v1", "heldout_v1"), ("heldout v2", "heldout_v2"), ("D2 blind", "D2")):
            got = load_task(os.path.join(ACC, sub), task)
            if got:
                cand[name] = (got[2], got[3])
        for name, (Zs, ys) in cand.items():
            P = softmax(Zs / T)
            r0, r1, r2, E = readouts(P)
            p, n01, n10 = mcnemar_p(r0 == ys, r1 == ys)
            sets[(task, name)] = {"T": T, "R0": e1_metrics(r0, ys, P.max(1), cut0), "R1": e1_metrics(r1, ys, P[np.arange(len(r1)), r1], cut1),
                                  "R2": e1_metrics(r2, ys), "mcnemar_p_R0_R1": p, "R1_fixes": n01, "R1_breaks": n10}
    pooled = {}
    for task in ("m_alarm_severity", "q_spc_action"):
        keys = [k for k in sets if k[0] == task]
        n = sum(sets[k]["R0"]["n"] for k in keys)
        acc0 = sum(sets[k]["R0"]["acc"] * sets[k]["R0"]["n"] for k in keys) / n
        acc1 = sum(sets[k]["R1"]["acc"] * sets[k]["R1"]["n"] for k in keys) / n
        mae0 = sum(sets[k]["R0"]["mae"] * sets[k]["R0"]["n"] for k in keys) / n
        mae1 = sum(sets[k]["R1"]["mae"] * sets[k]["R1"]["n"] for k in keys) / n
        worst = min((sets[k]["R1"]["acc"] - sets[k]["R0"]["acc"]) for k in keys)
        fixes = sum(sets[k]["R1_fixes"] for k in keys); breaks = sum(sets[k]["R1_breaks"] for k in keys)
        pooled[task] = {"n": n, "acc_R0": acc0, "acc_R1": acc1, "mae_R0": mae0, "mae_R1": mae1, "worst_set_delta": worst, "fixes": fixes, "breaks": breaks}
    # JevBench score items (fwd), T = jevlike v8 T_med
    an = json.load(open(os.path.join(JEV, "results/analysis.json")))
    T_med = float(np.median([r["temperature"] for r in an]))
    jb = {"T": T_med, "R0": 0, "R1": 0, "n": 0, "items": []}
    for tier in ("original", "easy", "hard"):
        for r in load_jsonl(os.path.join(JB, f"{tier}.jsonl")):
            if r.get("type") != "score" or "fwd" not in r:
                continue
            labels = r["labels"]
            lp = np.array([[r["fwd"]["raw_logprobs"].get(L) if r["fwd"]["raw_logprobs"].get(L) is not None else -30.0 for L in labels]])
            P = softmax(lp / T_med)
            r0, r1, _, E = readouts(P)
            yi = labels.index(str(r["expected"]))
            jb["n"] += 1; jb["R0"] += int(r0[0] == yi); jb["R1"] += int(r1[0] == yi)
            jb["items"].append({"id": r["task_id"], "expected": yi, "R0": int(r0[0]), "R1": int(r1[0]), "E": float(E[0])})
    sev = pooled["m_alarm_severity"]
    g_acc = sev["acc_R1"] >= sev["acc_R0"] + 0.02
    g_mae = sev["mae_R1"] <= sev["mae_R0"] + 1e-9
    g_worst = sev["worst_set_delta"] >= -0.02
    g_jb = jb["R1"] >= jb["R0"] - 1
    if g_acc and g_mae and g_worst and g_jb:
        verdict = "採用"
    elif sev["acc_R1"] < sev["acc_R0"] - 0.02 or not g_worst or not g_jb:
        verdict = "變差"
    else:
        verdict = "持平"
    res.update({"sets": {f"{k[0]} / {k[1]}": v for k, v in sets.items()}, "pooled": pooled, "jevbench_score": {k: v for k, v in jb.items() if k != "items"},
                "jevbench_items": jb["items"], "gates": {"acc_plus2": g_acc, "mae_not_up": g_mae, "no_set_worse_2pt": g_worst, "jevbench_ok": g_jb}, "verdict": verdict})
    L = ["# 17-E1 — 順序型題目：機率加權等級 vs argmax（離線）", "",
         "資料：jevlike 26B（UD-Q4_K_M、b11118）既有原始 logprob。溫度在 D0 急迫度 cal 半上擬，套到 D0 test、heldout 兩組、D2 盲寫。",
         "R0 = argmax；R1 = round(Σ k·P_k)，.5 往較輕一級；R2 = 中位等級（探索）。ε=5% 門檻在 D0 cal 上各自用該讀法的信心算。", "",
         f"## 判定：**{verdict}**", "",
         f"- 急迫度合併 {sev['n']} 筆：acc R0 {sev['acc_R0']:.3f} → R1 {sev['acc_R1']:.3f}（門檻 +0.02：{'✓' if g_acc else '✗'}）；"
         f"MAE {sev['mae_R0']:.3f} → {sev['mae_R1']:.3f}（不增：{'✓' if g_mae else '✗'}）；單一集合最差差 {sev['worst_set_delta'] * 100:+.1f} 點（≥ −2：{'✓' if g_worst else '✗'}）；"
         f"R1 翻對 {sev['fixes']}、翻錯 {sev['breaks']}。",
         f"- JevBench 分數題 {jb['n']} 題（T={T_med:.2f}）：R0 對 {jb['R0']}、R1 對 {jb['R1']}（R1 ≥ R0 − 1：{'✓' if g_jb else '✗'}）。", "",
         "## 逐集合", "", "| 題目 / 集合 | n | T | acc R0 / R1 / R2 | MAE R0 / R1 | 相鄰錯佔比 R0 / R1 | ε=5% coverage R0 / R1 | 翻對 / 翻錯 | McNemar p |", "|---|---|---|---|---|---|---|---|---|"]
    for (task, name), v in sets.items():
        a, b, c = v["R0"], v["R1"], v["R2"]
        L.append(f"| {task} / {name} | {a['n']} | {v['T']:.2f} | {a['acc']:.3f} / {b['acc']:.3f} / {c['acc']:.3f} | {a['mae']:.3f} / {b['mae']:.3f} | "
                 f"{f(a['adjacent_share_of_errors'], 2)} / {f(b['adjacent_share_of_errors'], 2)} | {f(a.get('coverage_eps5'), 2)} / {f(b.get('coverage_eps5'), 2)} | "
                 f"{v['R1_fixes']} / {v['R1_breaks']} | {v['mcnemar_p_R0_R1']:.3f} |")
    spc = pooled["q_spc_action"]
    L += ["", f"SPC 動作（choice，探索、不判門檻）合併 {spc['n']} 筆：acc R0 {spc['acc_R0']:.3f} → R1 {spc['acc_R1']:.3f}，翻對 {spc['fixes']}、翻錯 {spc['breaks']}。"]
    return res, L


# ---------------------------------------------------------------- E2
def risk_eval(conf_ca, ok_ca, conf_te, ok_te):
    cut, _ = risk_threshold(conf_ca, ok_ca, EPS)
    h = conf_te >= cut
    err = float((~ok_te[h]).mean()) if h.any() else 0.0
    return {"cut": cut, "coverage": float(h.mean()), "error": err, "holds": err <= EPS}


def run_e2():
    data = {}
    for t in TASKS:
        rows, letters, Z, y = load_task(ACC, t)
        ca = group_split(rows, 0)
        data[t] = (Z, y, ca, ~ca)
    def fit_on(tasks):
        return fit_T_pooled([data[t][0][data[t][2]] for t in tasks], [data[t][1][data[t][2]] for t in tasks])
    kinds = sorted(set(KIND.values()))
    T_kind = {k: fit_on([t for t in TASKS if KIND[t] == k]) for k in kinds}
    T_global = fit_on(list(TASKS))
    arms = {}
    for t in TASKS:
        Z, y, ca, te = data[t]
        same = [u for u in TASKS if KIND[u] == KIND[t] and u != t]
        T_loo = fit_on(same) if same else fit_on([u for u in TASKS if u != t])
        Ts = {"T_task": fit_on([t]), "T_kind": T_kind[KIND[t]], "T_global": T_global, "T_loo": T_loo}
        arms[t] = {}
        for name, T in Ts.items():
            P = softmax(Z / T); ok = P.argmax(1) == y; conf = P.max(1)
            arms[t][name] = {"T": T, "nll_test": nll_T([Z[te]], [y[te]], T), "ece_test": ece(conf[te], ok[te]), **risk_eval(conf[ca], ok[ca], conf[te], ok[te])}
    def avg(name, key):
        return float(np.mean([arms[t][name][key] for t in TASKS]))
    summ = {n: {"ece": avg(n, "ece_test"), "nll": avg(n, "nll_test"), "coverage": avg(n, "coverage"), "holds": int(sum(arms[t][n]["holds"] for t in TASKS))}
            for n in ("T_task", "T_kind", "T_global", "T_loo")}
    # JevBench transfer: per-type T_kind from D0 cal vs v8 single T_med, same ECE code
    an = json.load(open(os.path.join(JEV, "results/analysis.json")))
    T_med = float(np.median([r["temperature"] for r in an]))
    conf_k, conf_m, conf_1, okk = [], [], [], []
    for tier in ("original", "easy", "hard"):
        for r in load_jsonl(os.path.join(JB, f"{tier}.jsonl")):
            if "fwd" not in r:
                continue
            labels = r["labels"]
            lp = np.array([[r["fwd"]["raw_logprobs"].get(L) if r["fwd"]["raw_logprobs"].get(L) is not None else -30.0 for L in labels]])
            yi = labels.index(str(r["expected"]))
            Tk = T_kind.get(r["type"], T_global)
            for T, store in ((Tk, conf_k), (T_med, conf_m), (1.0, conf_1)):
                P = softmax(lp / T); store.append(float(P.max()))
            okk.append(int(np.argmax(lp) == yi))
    okk = np.array(okk, bool)
    transfer = {"n": int(len(okk)), "ece_raw": ece(np.array(conf_1), okk), "ece_Tmed_v8": ece(np.array(conf_m), okk), "ece_T_kind": ece(np.array(conf_k), okk), "T_med": T_med}
    s, t_, k_ = summ["T_loo"], summ["T_task"], summ["T_kind"]
    g_loo = s["ece"] <= t_["ece"] + 0.01 and s["holds"] >= t_["holds"] and s["coverage"] >= t_["coverage"] - 0.03
    g_kind = (k_["ece"] <= t_["ece"] + 0.01 and k_["holds"] >= t_["holds"] and k_["coverage"] >= t_["coverage"] - 0.03 and k_["nll"] <= t_["nll"])
    g_transfer = transfer["ece_T_kind"] <= 0.044
    verdict = "T_kind 取代 T_task" if (g_loo and g_kind) else ("T_loo 只給新題用、舊題維持 T_task" if g_loo else "維持 T_task")
    res = {"T_kind": T_kind, "T_global": T_global, "per_task": arms, "summary": summ, "jevbench_transfer": transfer,
           "gates": {"loo_ok": g_loo, "kind_replaces": g_kind, "transfer_ok": g_transfer}, "verdict": verdict}
    L = ["# 17-E2 — 溫度顆粒度：每 task / 每題型 / 全域 / 留一（離線）", "",
         "資料：jevlike 26B D0 原始 logprob，pair_id 分組切半（seed 0）。擬合格點同 jevlike（0.05–50 對數 200 點），NLL 最小。ε=5% 用 v9 的選擇性風險控制。", "",
         f"## 判定：**{verdict}**", "",
         f"- 題型溫度：choice {T_kind.get('choice', float('nan')):.2f}、noul {T_kind.get('noul', float('nan')):.2f}、score {T_kind.get('score', float('nan')):.2f}；全域 {T_global:.2f}。",
         f"- T_loo 當新題預設：{'✓' if g_loo else '✗'}（ECE ≤ T_task + 0.01、守住數不少、coverage 不少於 −3 點）。T_kind 取代 T_task：{'✓' if g_kind else '✗'}（再加 NLL 不高於 T_task）。",
         f"- JevBench 轉移（{transfer['n']} 題 fwd，top-label ECE，同一支程式）：raw {transfer['ece_raw']:.3f}、v8 單一 T={T_med:.2f} {transfer['ece_Tmed_v8']:.3f}、"
         f"題型 T {transfer['ece_T_kind']:.3f}（門檻 ≤ 0.044：{'✓' if g_transfer else '✗'}）。", "",
         "## 平均（10 類 test）", "", "| 臂 | ECE | NLL | ε=5% 守住 | ε=5% 平均 coverage |", "|---|---|---|---|---|"]
    for n in ("T_task", "T_kind", "T_global", "T_loo"):
        v = summ[n]
        L.append(f"| {n} | {v['ece']:.4f} | {v['nll']:.4f} | {v['holds']}/10 | {v['coverage']:.3f} |")
    L += ["", "## 逐類", "", "| task | 題型 | T_task | T_kind | T_loo | ECE task / kind / loo | coverage task / kind / loo | 守住 task / kind / loo |", "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        a = arms[t]
        L.append(f"| {t} | {KIND[t]} | {a['T_task']['T']:.2f} | {a['T_kind']['T']:.2f} | {a['T_loo']['T']:.2f} | "
                 f"{a['T_task']['ece_test']:.3f} / {a['T_kind']['ece_test']:.3f} / {a['T_loo']['ece_test']:.3f} | "
                 f"{a['T_task']['coverage']:.2f} / {a['T_kind']['coverage']:.2f} / {a['T_loo']['coverage']:.2f} | "
                 f"{'✓' if a['T_task']['holds'] else '✗'} / {'✓' if a['T_kind']['holds'] else '✗'} / {'✓' if a['T_loo']['holds'] else '✗'} |")
    return res, L


def save(key, res, L):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "v12.json")
    allres = json.load(open(p)) if os.path.exists(p) else {}
    allres[key] = res
    json.dump(allres, open(p, "w"), ensure_ascii=False, indent=1, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else (float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))
    open(os.path.join(OUT, f"17-v12-{key}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:10]))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        save("e1", *run_e1())
    if which in ("e2", "all"):
        save("e2", *run_e2())
    if which in ("e3", "all") or which in ("e4", "all"):
        import analyze_v12_gpu  # noqa: F401  (E3/E4 live in a second file, written once their runs exist)
        analyze_v12_gpu.main(which, save)
