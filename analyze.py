"""analyze.py — P1 回溯校準的分析（本機 CPU）。門檻 G0–G2b 照 PLAN.md §4 凍結，跑完照填。

輸入：results/modal/retro/<variant>/<task>.jsonl（jevlike bench.accuracy 的原始 logprobs）、data/opt/features.jsonl
輸出：results/01-retro.md、results/retro.json、results/thresholds.lock.json

程序（跑前寫死）：
1. 每個 variant × task：logit 矩陣（jevlike analyze.logit_matrix），cal 上以 NLL 擬合溫度 T，P = softmax(Z/T)。
   P(噴) = P[A]（opt_play 的 A=賭、opt_tier 的 A=高）。溫度單調，AUROC 不受 T 影響；ECE 受影響。
2. G0 洩漏：opt_play 的 test acc（argmax）mask vs nomask 差 ≤ 2 點（crit 與 nocrit 各看）。補報 AUROC 差。
3. G1 鑑別力：test 上 P(噴) 對 hit 的 AUROC ≥ 0.65（< 0.60 = 不做）。主變體 = mask_crit；其餘為消融。
4. G2 賭籃子：門檻在 cal 上定——目標命中率 = cal 規則 B 命中率 + 5pp；取 cal 上「P(噴) ≥ thr 的子集命中率 ≥ 目標」中 coverage 最大的 thr；
   若不存在，取 cal coverage 20% 分位的 thr。套到 test：子集命中率 ≥ test 規則 B 命中率 + 5pp、子集 ≥ 30 筆、coverage ≥ 20%；
   EV(C_bound) 在三點齊全列上 ≥ test 規則 B 的 EV(C_bound)。
5. G2b 基線：同 15 個結構欄位的 logistic regression（cal 訓練）；門檻用「與 jevlike 相同 cal coverage」對齊；test 上 jevlike 籃子命中率 ≥ LR 籃子 + 3pp。
6. 2×2 配對表：規則 B × jevlike 賭 四格的 n、命中率、EV。
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "jevlike", "bench"))
sys.path.insert(0, os.path.join(ROOT, "jevlike"))
os.environ.setdefault("MPLBACKEND", "Agg")
from analyze import ece, fit_temperature, load_jsonl, logit_matrix, softmax  # noqa: E402  (jevlike)

RETRO = os.path.join(ROOT, os.environ.get("RETRO_DIR", "results/modal/retro"))
FEATS = os.path.join(ROOT, "data/opt/features.jsonl")
VARIANTS = ["mask_crit", "mask_nocrit", "nomask_crit", "nomask_nocrit"]
MAIN = "mask_crit"
TASKS = {"opt_play": "AB", "opt_tier": "ABC"}
G = {"g0_acc_diff_max": 2.0, "g1_auroc_min": 0.65, "g1_auroc_stop": 0.60, "g2_pp": 5.0, "g2_min_n": 30, "g2_min_cov": 0.20, "g2b_pp": 3.0, "fallback_cov": 0.20}
FEATURE_KEYS = ["sweep", "ignition", "first_seen_in_feed", "burst_no_history", "low_base", "leaps", "cigar_butt", "mass_grave", "gamble"]


# ---------------------------------------------------------------- optscnr 出場政策（同 strategy_lab，純函式複製）
def _tier(p):
    return "lottery" if p < 1.5 else ("mid" if p < 3.0 else "heavy")


def pol_hold(m, f):
    return m[-1]


def pol_half(m, f, th=2.0):
    return (th + m[-1]) / 2 if max(m) >= th else m[-1]


def pol_rungs(m, f, th=(2, 4, 8)):
    fired = [t for t in th if max(m) >= t]
    return (sum(fired) + (len(th) - len(fired)) * m[-1] + m[-1]) / (len(th) + 1)


def pol_stop_half(m, f, stop=0.5, th=2.0):
    for x in m:
        if x >= th:
            return (th + m[-1]) / 2
        if x <= stop:
            return stop
    return m[-1]


def pol_bound(m, f):
    t = _tier(f["entry_price"])
    if t == "lottery" and f.get("entry_iv") is not None and f["entry_iv"] < 50:
        return pol_hold(m, f)
    if t == "heavy":
        return pol_rungs(m, f)
    return pol_stop_half(m, f)


POLICIES = {"A_hold": pol_hold, "B_half2x": pol_half, "C_bound": pol_bound, "stop_half": pol_stop_half}


def basket_stats(feats, mask):
    R = [f for f, m in zip(feats, mask) if m]
    n = len(R)
    out = {"n": n, "hit": (sum(r["hit"] for r in R) / n) if n else None, "n_ev": 0, "ev": {}}
    R3 = [r for r in R if r["multiples"]]
    out["n_ev"] = len(R3)
    if R3:
        out["ev"] = {p: float(np.mean([fn(r["multiples"], r) for r in R3])) for p, fn in POLICIES.items()}
    return out


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- 門檻選法（凍結）
def pick_threshold(conf_cal, hit_cal, target):
    """cal 上：conf ≥ thr 的子集命中率 ≥ target 中 coverage 最大的 thr；沒有就取 coverage 20% 分位。"""
    order = np.sort(np.unique(conf_cal))[::-1]
    best = None
    for thr in order:
        h = conf_cal >= thr
        if h.sum() < G["g2_min_n"]:
            continue
        if hit_cal[h].mean() >= target:
            best = (float(thr), float(h.mean()))  # 越往下 coverage 越大；保留最後一個滿足的
    if best is not None:
        return best[0], "target_met"
    thr = float(np.quantile(conf_cal, 1 - G["fallback_cov"]))
    return thr, "fallback_cov20"


# ---------------------------------------------------------------- LR 基線
def feature_matrix(feats):
    X = []
    for r in feats:
        x = [math.log(max(r["entry_price"], 0.01)), (r.get("entry_iv") or 0) / 100, math.log1p(r["dte"]), (r.get("otm_pct") or 0) / 100,
             math.log1p(max(r.get("oi") or 0, 0)), math.copysign(math.log1p(abs(r.get("oi_d7") or 0)), r.get("oi_d7") or 0),
             math.log1p(max(r.get("volume") or 0, 0)), float(r.get("score") or 0), float(r["rule_b"]), float(r["news_at_signal"]),
             float(r["has_event"]), float(r["n_warnings"]), math.log1p(min(r.get("ignition_x") or 0, 1000)), float(r.get("underlying_move") or 0) / 10,
             float(r["premium_tier"] == "lottery"), float(r["premium_tier"] == "heavy")]
        x += [float(k in (r["features"] or [])) for k in FEATURE_KEYS]
        X.append(x)
    return np.array(X)


def lr_baseline(feats, ca, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = feature_matrix(feats)
    y = np.array([r["hit"] for r in feats], int)
    sc = StandardScaler().fit(X[ca])
    clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[ca]), y[ca])
    p = clf.predict_proba(sc.transform(X))[:, 1]
    return p, {"auroc_cal": auroc(y[ca], p[ca]), "auroc_test": auroc(y[te], p[te]), "n_features": X.shape[1]}


# ---------------------------------------------------------------- 主流程
def analyze_variant(variant, task, feats, by_id, ca, te, hit, rule_b, lr_p):
    rows = [r for r in load_jsonl(os.path.join(RETRO, variant, f"{task}.jsonl")) if "raw_logprobs" in r]
    if not rows:
        return None
    idx = {r["id"]: r for r in rows}
    keep = np.array([f["id"] in idx for f in feats])
    if keep.sum() < len(feats):
        print(f"[{variant}/{task}] only {keep.sum()}/{len(feats)} rows have results", flush=True)
    R = [idx[f["id"]] for f in feats if f["id"] in idx]
    F = [f for f in feats if f["id"] in idx]
    ca_v, te_v, hit_v, rb_v, lr_v = ca[keep], te[keep], hit[keep], rule_b[keep], lr_p[keep]
    letters = list(TASKS[task])
    Z = logit_matrix(R, letters)
    y = np.array([letters.index(r["gold"]) for r in R])
    T = fit_temperature(Z[ca_v], y[ca_v])
    P = softmax(Z / T)
    Praw = softmax(Z)
    pred = P.argmax(1)
    conf = P[:, 0]  # P(噴) = P[A]
    out = {"variant": variant, "task": task, "n": len(R), "T": T, "missing_rate": float(np.mean([bool(r["missing"]) for r in R])),
           "option_mass_p5": float(np.percentile([r.get("option_mass") or 0 for r in R], 5)),
           "acc_test": float((pred[te_v] == y[te_v]).mean()), "acc_cal": float((pred[ca_v] == y[ca_v]).mean()),
           "acc_raw_test": float((Praw.argmax(1)[te_v] == y[te_v]).mean()),
           "pred_A_rate_test": float((pred[te_v] == 0).mean()), "raw_pred_A_rate_test": float((Praw.argmax(1)[te_v] == 0).mean()),
           "auroc_test": auroc(hit_v[te_v], conf[te_v]), "auroc_cal": auroc(hit_v[ca_v], conf[ca_v]),
           "ece_test_raw": ece(Praw[:, 0][te_v], hit_v[te_v]), "ece_test_T": ece(conf[te_v], hit_v[te_v]),
           "mean_conf_raw_test": float(Praw[:, 0][te_v].mean()), "mean_conf_T_test": float(conf[te_v].mean())}
    # G2 賭籃子
    rb_cal_hit = hit_v[ca_v & rb_v].mean() if (ca_v & rb_v).any() else float("nan")
    rb_test = basket_stats(F, te_v & rb_v)
    target = rb_cal_hit + G["g2_pp"] / 100
    thr, how = pick_threshold(conf[ca_v], hit_v[ca_v], target)
    bet = conf >= thr
    cal_b, test_b = basket_stats(F, ca_v & bet), basket_stats(F, te_v & bet)
    cov_test = float(bet[te_v].mean())
    # LR 對齊 coverage
    cal_cov = float(bet[ca_v].mean())
    lr_thr = float(np.quantile(lr_v[ca_v], 1 - cal_cov)) if 0 < cal_cov < 1 else float("inf")
    lr_bet = lr_v >= lr_thr
    lr_test_b = basket_stats(F, te_v & lr_bet)
    # 補充（非判準）：LR 門檻改為對齊 jevlike 的 test coverage
    lr_thr_te = float(np.quantile(lr_v[te_v], 1 - cov_test)) if 0 < cov_test < 1 else float("inf")
    lr_test_b_te = basket_stats(F, te_v & (lr_v >= lr_thr_te))
    # 2x2
    grid = {}
    for rbv in (True, False):
        for bv in (True, False):
            grid[f"ruleB={int(rbv)}/jev={int(bv)}"] = basket_stats(F, te_v & (rb_v == rbv) & (bet == bv))
    out.update({"target_hit_cal": float(target), "rule_b_cal_hit": float(rb_cal_hit), "threshold": thr, "threshold_how": how,
                "cal_basket": cal_b, "cal_coverage": cal_cov, "test_basket": test_b, "test_coverage": cov_test, "rule_b_test": rb_test,
                "test_basket_ci": wilson(int(round((test_b["hit"] or 0) * test_b["n"])), test_b["n"]),
                "lr_threshold": lr_thr, "lr_test_basket": lr_test_b, "lr_test_coverage": float(lr_bet[te_v].mean()), "lr_test_basket_matched_test_cov": lr_test_b_te, "grid2x2": grid,
                "top_conf_ids_test": [F[i]["id"] for i in np.argsort(-conf * te_v)[:10]]})
    return out


def verdicts(res, lr_info):
    V = {}
    mp = res.get(("mask_crit", "opt_play")); nmp = res.get(("nomask_crit", "opt_play"))
    mn = res.get(("mask_nocrit", "opt_play")); nmn = res.get(("nomask_nocrit", "opt_play"))
    if mp and nmp:
        d = abs(mp["acc_test"] - nmp["acc_test"]) * 100
        V["G0"] = {"acc_diff_pt_crit": d, "acc_diff_pt_nocrit": (abs(mn["acc_test"] - nmn["acc_test"]) * 100) if (mn and nmn) else None,
                   "auroc_diff_crit": nmp["auroc_test"] - mp["auroc_test"], "pass": d <= G["g0_acc_diff_max"],
                   "note": "過：遮/不遮差 ≤ 2 點。沒過：模型在用記憶，只認遮的版本（本報告一律以 mask_crit 為主變體）。"}
    if mp:
        a = mp["auroc_test"]
        V["G1"] = {"auroc_test": a, "pass": a >= G["g1_auroc_min"], "stop": a < G["g1_auroc_stop"], "auroc_by_variant": {f"{v}/{t}": r["auroc_test"] for (v, t), r in res.items()}}
        tb, rb = mp["test_basket"], mp["rule_b_test"]
        rb_hit = rb["hit"] if rb["hit"] is not None else float("nan")
        c1 = tb["hit"] is not None and tb["hit"] >= rb_hit + G["g2_pp"] / 100
        c2 = tb["n"] >= G["g2_min_n"]; c3 = mp["test_coverage"] >= G["g2_min_cov"]
        ev_j, ev_r = tb["ev"].get("C_bound"), rb["ev"].get("C_bound")
        c4 = (ev_j is not None and ev_r is not None and ev_j >= ev_r)
        V["G2"] = {"basket_hit": tb["hit"], "basket_n": tb["n"], "coverage": mp["test_coverage"], "rule_b_hit_test": rb_hit, "rule_b_n_test": rb["n"],
                   "ev_c_bound_basket": ev_j, "ev_c_bound_rule_b": ev_r, "n_ev_basket": tb["n_ev"], "n_ev_rule_b": rb["n_ev"],
                   "hit_ok": c1, "n_ok": c2, "cov_ok": c3, "ev_ok": c4, "pass": c1 and c2 and c3 and c4}
        lb = mp["lr_test_basket"]
        V["G2b"] = {"jev_basket_hit": tb["hit"], "lr_basket_hit": lb["hit"], "lr_n": lb["n"], "lr_auroc_test": lr_info["auroc_test"], "jev_auroc_test": a,
                    "pass": (tb["hit"] is not None and lb["hit"] is not None and tb["hit"] >= lb["hit"] + G["g2b_pp"] / 100)}
        if V["G1"]["stop"]:
            V["decision"] = "不做（G1 < 0.60，與擲硬幣無異）"
        elif V["G1"]["pass"] and V["G2"]["pass"] and V["G2b"]["pass"]:
            V["decision"] = "上線快選器（G1、G2、G2b 全過）→ 進 P2 前瞻配對"
        elif V["G1"]["pass"] and not V["G2b"]["pass"]:
            V["decision"] = "G1 過但 G2b 沒過：用 LR 取代 GPU；jevlike 至多當第二意見"
        elif V["G1"]["pass"]:
            V["decision"] = "G1 過、G2 沒過：只當第二意見，不當快選器"
        else:
            V["decision"] = "G1 0.60–0.65：不進 P2；記錄後停"
    return V


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_report(res, lr_info, V, feats, ca, te, path, run_meta):
    L = ["# 01 — P1 回溯校準：jevlike 在 optscnr 訊號 log 上的賭／不賭", "",
         f"結果目錄 `{os.path.relpath(RETRO, ROOT)}`；資料 `data/opt/`（cal {int(ca.sum())} 筆 = 2026-06/07/08，test {int(te.sum())} 筆 = 2026-09）。"
         f"門檻 PLAN.md §4 凍結；程序見本檔案頂端。run：{run_meta}", "",
         "## 判定", "", f"**{V.get('decision', '（結果不全）')}**", ""]
    if "G0" in V:
        g = V["G0"]
        L += [f"- G0 洩漏：opt_play test acc 遮/不遮差 {f(g['acc_diff_pt_crit'], 1)} 點（crit）、{f(g['acc_diff_pt_nocrit'], 1)} 點（nocrit）；AUROC 差（不遮 − 遮）{f(g['auroc_diff_crit'])} → {'✓ 過' if g['pass'] else '✗ 沒過'}。{g['note']}"]
    if "G1" in V:
        g = V["G1"]
        L += [f"- G1 鑑別力：mask_crit/opt_play test AUROC {f(g['auroc_test'])}（門檻 0.65，< 0.60 停）→ {'✓ 過' if g['pass'] else ('✗ 停' if g['stop'] else '✗ 沒過')}"]
    if "G2" in V:
        g = V["G2"]
        L += [f"- G2 賭籃子：test 命中 {f(g['basket_hit'])}（n={g['basket_n']}，coverage {f(g['coverage'], 2)}）vs 規則 B 同視窗 {f(g['rule_b_hit_test'])}（n={g['rule_b_n_test']}）+5pp → 命中 {'✓' if g['hit_ok'] else '✗'}、n≥30 {'✓' if g['n_ok'] else '✗'}、coverage≥20% {'✓' if g['cov_ok'] else '✗'}；"
              f"EV(C_bound) 籃子 {f(g['ev_c_bound_basket'], 2)}（n={g['n_ev_basket']}）vs 規則 B {f(g['ev_c_bound_rule_b'], 2)}（n={g['n_ev_rule_b']}）{'✓' if g['ev_ok'] else '✗'} → {'✓ 過' if g['pass'] else '✗ 沒過'}"]
    if "G2b" in V:
        g = V["G2b"]
        L += [f"- G2b 基線：jevlike 籃子命中 {f(g['jev_basket_hit'])} vs LR 同 coverage 籃子 {f(g['lr_basket_hit'])}（n={g['lr_n']}）+3pp → {'✓ 過' if g['pass'] else '✗ 沒過'}；AUROC jevlike {f(g['jev_auroc_test'])} vs LR {f(g['lr_auroc_test'])}"]
    L += ["", "## 所有變體", "", "| variant | task | n | T | missing | acc test（校準後 / raw） | 預測 A 比例（校準後 / raw） | AUROC cal / test | ECE test raw → T | 平均 P(A) raw → T |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (v, t), r in res.items():
        L.append(f"| {v} | {t} | {r['n']} | {r['T']:.2f} | {r['missing_rate']:.3f} | {r['acc_test']:.3f} / {r['acc_raw_test']:.3f} | {r['pred_A_rate_test']:.2f} / {r['raw_pred_A_rate_test']:.2f} "
                 f"| {f(r['auroc_cal'])} / {f(r['auroc_test'])} | {r['ece_test_raw']:.3f} → {r['ece_test_T']:.3f} | {r['mean_conf_raw_test']:.2f} → {r['mean_conf_T_test']:.2f} |")
    L += ["", "## 賭籃子（每變體，門檻在 cal 上定）", "", "| variant | task | 門檻 | 選法 | cal 籃子 n / 命中 / coverage | test 籃子 n / 命中 [95% CI] / coverage | test EV C_bound（n） | 規則 B test n / 命中 / EV | LR 同 cal-coverage test n / 命中 | LR 同 test-coverage n / 命中（補充） |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (v, t), r in res.items():
        cb, tb, rb, lb = r["cal_basket"], r["test_basket"], r["rule_b_test"], r["lr_test_basket"]
        ci = r["test_basket_ci"]
        L.append(f"| {v} | {t} | {r['threshold']:.3f} | {r['threshold_how']} | {cb['n']} / {f(cb['hit'])} / {r['cal_coverage']:.2f} | {tb['n']} / {f(tb['hit'])} [{f(ci[0], 2)}, {f(ci[1], 2)}] / {r['test_coverage']:.2f} "
                 f"| {f(tb['ev'].get('C_bound'), 2)}（{tb['n_ev']}） | {rb['n']} / {f(rb['hit'])} / {f(rb['ev'].get('C_bound'), 2)} | {lb['n']} / {f(lb['hit'])} |")
    m = res.get((MAIN, "opt_play"))
    if m:
        L += ["", f"## 2×2 配對表（{MAIN}/opt_play，test）", "", "| 規則 B | jevlike 賭 | n | 命中 | EV C_bound（n） |", "|---|---|---|---|---|"]
        for k, b in m["grid2x2"].items():
            rbv, jv = k.split("/")
            L.append(f"| {'過' if rbv.endswith('1') else '不過'} | {'賭' if jv.endswith('1') else '不賭'} | {b['n']} | {f(b['hit'])} | {f(b['ev'].get('C_bound'), 2)}（{b['n_ev']}） |")
    hit = np.array([r["hit"] for r in feats])
    rb = np.array([r["rule_b"] for r in feats])
    L += ["", "## 基線與背景", "",
          f"- 基礎命中率：cal {hit[ca].mean():.3f}、test {hit[te].mean():.3f}；規則 B：cal {hit[ca & rb].mean():.3f}（n={int((ca & rb).sum())}）、test {hit[te & rb].mean():.3f}（n={int((te & rb).sum())}）。",
          f"- LR 基線（{lr_info['n_features']} 個結構欄位，cal 訓練，class_weight=balanced）：AUROC cal {f(lr_info['auroc_cal'])}、test {f(lr_info['auroc_test'])}。",
          "- 九月 test 的規則 B 只有十幾筆，+5pp 的比較統計上很弱；籃子命中率附 Wilson 95% CI。九月晚期列只有 t5/t10 檢查點，峰值上限較低，對所有方法一體適用。",
          "- 溫度 T 是在 cal 上以 NLL 擬合；AUROC 與 T 無關，ECE 與平均 P(A) 才看得出校準前後差異。",
          "- 系統提示沿用 jevlike 的 `SYSTEM`（「你是產線決策引擎。只回答一個字母。」），模板未改（PLAN §5）。"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    feats = load_jsonl(FEATS)
    by_id = {r["id"]: r for r in feats}
    ca = np.array([r["split"] == "cal" for r in feats]); te = ~ca
    hit = np.array([r["hit"] for r in feats]); rule_b = np.array([r["rule_b"] for r in feats])
    lr_p, lr_info = lr_baseline(feats, ca, te)
    res = {}
    for v in VARIANTS:
        for t in TASKS:
            r = analyze_variant(v, t, feats, by_id, ca, te, hit, rule_b, lr_p)
            if r:
                res[(v, t)] = r
    V = verdicts(res, lr_info)
    run_meta = ""
    rp = os.path.join(RETRO, "_run.json")
    if os.path.exists(rp):
        runs = load_jsonl(rp)
        run_meta = "; ".join(f"{r.get('gpu')} {r.get('model_file')} build {r.get('build_info')} bench {r.get('bench_s')}s" for r in runs[-2:])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    json.dump({"commit": commit, "thresholds": G, "lr": lr_info, "verdicts": V, "results": {f"{v}/{t}": r for (v, t), r in res.items()}},
              open(os.path.join(ROOT, "results/retro.json"), "w"), ensure_ascii=False, indent=1, default=float)
    lock = {"commit": commit, "retro_dir": os.path.relpath(RETRO, ROOT), "main_variant": MAIN,
            "tasks": {f"{v}/{t}": {"T": r["T"], "threshold": r["threshold"], "threshold_how": r["threshold_how"], "n_cal": int(ca.sum()), "n_test": int(te.sum()),
                                  "auroc_test": r["auroc_test"], "test_basket": r["test_basket"], "test_coverage": r["test_coverage"]} for (v, t), r in res.items()}}
    json.dump(lock, open(os.path.join(ROOT, "results/thresholds.lock.json"), "w"), ensure_ascii=False, indent=1, default=float)
    write_report(res, lr_info, V, feats, ca, te, os.path.join(ROOT, "results/01-retro.md"), run_meta)
    print(json.dumps(V, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
