"""E4 (docs/handoff-jevlike-v12.md §5): think only on the low-confidence tail, then read the letter.

Stage A = jevlike's current readout (thinking off, first-token letter logprob).
Stage B = for rows whose calibrated stage-A confidence is below --trigger: thinking-on template, prefill the thought
channel, generate up to --budget tokens greedily, close the channel, read the letter logprobs at the answer position.
Calibration inside the run: per task, T fitted by NLL on the pair_id cal half (same split and grid as jevlike).
The run records B for every row under --trigger (default 0.9) so the analysis can apply the pre-registered 0.7 and 0.9.

args: "--tasks m_alarm_severity,q_spc_action,p_uph_anomaly --trigger 0.9 --budget 512 --workers 4 --jevbench 1 --jb-T 3.28 --step0 0 --limit 0"
Out: <out_dir>/<task>.jsonl, <out_dir>/jevbench.jsonl, <out_dir>/_step0.json, <out_dir>/_summary.json
"""
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import read_option_probs  # noqa: E402
from decide.prompt import SYSTEM_EN, TemplateRenderer, letters_for  # noqa: E402

ROOT = "/root" if os.path.isdir("/root/data") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
GRID = np.exp(np.linspace(math.log(0.05), math.log(50), 200))


def group_split(rows, seed=0):  # == jevlike bench/analyze.group_split (inlined: analyze imports matplotlib)
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed); rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return np.array([(r.get("pair_id") or r["id"]) in cal for r in rows])


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def fit_T(Z, y):
    best, bt = float("inf"), 1.0
    for T in GRID:
        P = softmax(Z / T); v = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
        if v < best:
            best, bt = v, float(T)
    return bt


def logits(raw, letters):
    return np.array([raw.get(L) if raw.get(L) is not None else -30.0 for L in letters], dtype=float)


class Thinker:
    def __init__(self, base_url, budget):
        self.base_url, self.budget = base_url, budget
        r = requests.post(f"{base_url}/tokenize", json={"content": CLOSE, "add_special": False, "parse_special": True, "with_pieces": True}, timeout=30).json()
        toks = r.get("tokens", [])
        self.close_ids = {t["id"] if isinstance(t, dict) else t for t in toks} if len(toks) == 1 else set()
        self.close_tokenization = toks

    def think(self, prompt_think, session):
        body = {"prompt": prompt_think + OPEN, "n_predict": self.budget, "temperature": 0, "cache_prompt": True,
                "return_tokens": True, "stop": [CLOSE]}
        t0 = time.perf_counter()
        r = session.post(f"{self.base_url}/completion", json=body, timeout=600).json()
        gen_ms = (time.perf_counter() - t0) * 1000
        toks = r.get("tokens") or []
        cut = next((i for i, t in enumerate(toks) if t in self.close_ids), None)
        closed = cut is not None or r.get("stop_type") == "word" or bool(r.get("stopping_word"))
        if cut is not None:
            toks = toks[:cut]
            text = session.post(f"{self.base_url}/detokenize", json={"tokens": toks}, timeout=60).json().get("content", "")
        else:
            text = r.get("content", "")
            if CLOSE in text:
                text = text.split(CLOSE)[0]; closed = True
        return {"thought": text, "n_tokens": int(r.get("tokens_predicted") or len(toks)), "closed": closed, "gen_ms": gen_ms,
                "stop_type": r.get("stop_type"), "content_head": (r.get("content") or "")[:200]}


def stage_b(th, tr1, prompt_msgs_or_state, letters, session, render):
    p1 = render(tr1, prompt_msgs_or_state)
    t = th.think(p1, session)
    t0 = time.perf_counter()
    rd = read_option_probs(th.base_url, p1 + OPEN + t["thought"] + CLOSE, letters, session=session)
    return {"raw_logprobs": rd["raw_logprobs"], "probs": rd["probs"], "option_mass": rd.get("option_mass"), "first_token": rd["first_token"],
            "n_tokens": t["n_tokens"], "truncated": not t["closed"], "gen_ms": t["gen_ms"], "read_ms": (time.perf_counter() - t0) * 1000,
            "stop_type": t["stop_type"], "thought": t["thought"][:3000]}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    tasks = a.get("--tasks", "m_alarm_severity,q_spc_action,p_uph_anomaly").split(",")
    trigger = float(a.get("--trigger", 0.9)); budget = int(a.get("--budget", 512)); workers = int(a.get("--workers", 4))
    step0 = a.get("--step0", "0") == "1"; limit = int(a.get("--limit", 0)); do_jb = a.get("--jevbench", "1") == "1"
    jb_T = float(a.get("--jb-T", 3.28))
    os.makedirs(out_dir, exist_ok=True)
    tr0 = TemplateRenderer(base_url)
    tr1 = TemplateRenderer(base_url, enable_thinking=True)
    th = Thinker(base_url, budget)
    tls = threading.local()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        return tls.s

    render_state = lambda tr, rq: tr.render(rq[0], rq[1])  # noqa: E731
    render_msgs = lambda tr, m: tr.render_messages(m)  # noqa: E731

    if step0:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/synthetic/m_alarm_severity.jsonl"), encoding="utf-8")][:3]
        info = {"template_nothink_tail": tr0.template[-120:], "template_think_head": tr1.template[:160], "template_think_tail": tr1.template[-120:],
                "think_kwargs_honored": tr1.server_honored_kwargs, "close_tokenization": th.close_tokenization, "rows": []}
        for r in rows:
            letters = letters_for(len(r["question"]["options"]))
            A = read_option_probs(base_url, tr0.render(r["state"], r["question"]), letters, session=sess())
            B = stage_b(th, tr1, (r["state"], r["question"]), letters, sess(), render_state)
            info["rows"].append({"id": r["id"], "gold": r["gold"], "A_probs": A["probs"], "A_mass": A.get("option_mass"),
                                 "B_probs": B["probs"], "B_mass": B["option_mass"], "B_first_token": B["first_token"], "B_tokens": B["n_tokens"],
                                 "B_truncated": B["truncated"], "B_stop_type": B["stop_type"], "B_thought_head": B["thought"][:400], "B_gen_ms": B["gen_ms"]})
        json.dump(info, open(os.path.join(out_dir, "_step0.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps(info, ensure_ascii=False, indent=1)[:6000], flush=True)
        return info

    summary = {}
    for task in tasks:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8") if l.strip()]
        if limit:
            rows = rows[:limit]
        letters = letters_for(len(rows[0]["question"]["options"]))
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            A = list(ex.map(lambda r: read_option_probs(base_url, tr0.render(r["state"], r["question"]), letters, session=sess()), rows))
        Z = np.array([logits(x["raw_logprobs"], letters) for x in A]); y = np.array([letters.index(r["gold"]) for r in rows])
        ca = group_split(rows); T = fit_T(Z[ca], y[ca]); conf = softmax(Z / T).max(1)
        trig = [i for i in range(len(rows)) if conf[i] < trigger]
        print(f"[{task}] stage A done ({time.time() - t0:.0f}s), T={T:.2f}, {len(trig)}/{len(rows)} below {trigger}", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            B = dict(zip(trig, ex.map(lambda i: stage_b(th, tr1, (rows[i]["state"], rows[i]["question"]), letters, sess(), render_state), trig)))
        with open(os.path.join(out_dir, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": r["id"], "task": task, "gold": r["gold"], "pair_id": r.get("pair_id"), "split": "cal" if ca[i] else "test",
                                    "letters": letters, "A": {k: A[i].get(k) for k in ("raw_logprobs", "probs", "option_mass", "latency_ms")},
                                    "A_T": T, "A_conf_T": float(conf[i]), "B": B.get(i)}, ensure_ascii=False) + "\n")
        summary[task] = {"n": len(rows), "T": T, "n_triggered": len(trig), "s": round(time.time() - t0, 1)}
        print(f"[{task}] {summary[task]}", flush=True)

    if do_jb:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bench.jevbench import user_text
        items = []
        for tier in ("original", "easy", "hard"):
            items += [dict(json.loads(l), tier=tier) for l in open(os.path.join(ROOT, f"data/jevbench/{tier}.jsonl"), encoding="utf-8") if l.strip()]
        if limit:
            items = items[:limit]
        t0 = time.time()

        def msgs(it):
            return [{"role": "system", "content": SYSTEM_EN}, {"role": "user", "content": user_text(it, list(range(len(it["labels"]))))}]

        def a_read(it):
            return read_option_probs(base_url, tr0.render_messages(msgs(it)), letters_for(len(it["labels"])), session=sess())
        with ThreadPoolExecutor(workers) as ex:
            A = list(ex.map(a_read, items))
        confs = []
        for it, x in zip(items, A):
            letters = letters_for(len(it["labels"])); z = logits(x["raw_logprobs"], letters)[None, :]
            confs.append(float(softmax(z / jb_T).max()))
        trig = [i for i, c in enumerate(confs) if c < trigger]
        with ThreadPoolExecutor(workers) as ex:
            B = dict(zip(trig, ex.map(lambda i: stage_b(th, tr1, msgs(items[i]), letters_for(len(items[i]["labels"])), sess(), render_msgs), trig)))
        with open(os.path.join(out_dir, "jevbench.jsonl"), "w", encoding="utf-8") as f:
            for i, it in enumerate(items):
                letters = letters_for(len(it["labels"]))
                f.write(json.dumps({"task_id": it["id"], "tier": it["tier"], "type": it["question"]["type"], "labels": it["labels"], "expected": str(it["expected"]),
                                    "letters": letters, "A": {k: A[i].get(k) for k in ("raw_logprobs", "probs", "option_mass", "latency_ms")},
                                    "A_conf_T": confs[i], "B": B.get(i)}, ensure_ascii=False) + "\n")
        summary["jevbench"] = {"n": len(items), "T": jb_T, "n_triggered": len(trig), "s": round(time.time() - t0, 1)}
        print(f"[jevbench] {summary['jevbench']}", flush=True)
    json.dump(summary, open(os.path.join(out_dir, "_summary.json"), "w"), indent=1)
    return summary
