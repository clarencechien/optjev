"""v12 (docs/handoff-jevlike-v12.md): Gemma 4 12B / 31B ladder rungs (E3) and the 26B think-on-the-tail readout (E4).

Run from the optjev repo root (paths below are relative to it):
  scripts/modal.sh run v12/modal_v12.py::download --name 12b          # CPU container -> models volume /models/v12/
  scripts/modal.sh run v12/modal_v12.py::run_l4   --name 12b --which smoke
  scripts/modal.sh run --detach v12/modal_v12.py::suite_l4   --name 12b      # D0 + D2 + JevBench + latency
  scripts/modal.sh run --detach v12/modal_v12.py::suite_l40s --name 31b
  scripts/modal.sh run --detach v12/modal_v12.py::run_l4 --name 26b --which think_tail --args "..."
  scripts/modal.sh volume get optjev-results v12 results/modal/
Server: the llama.cpp b11371 build that jevlike v11 compiled into the models volume (CUDA archs 89/90 = L4, L40S).
Same flags as jevlike's 26B runs (-c 16384 -np 4 --jinja --reasoning-budget 0) plus --swa-full for the shared-state latency.
Only synthetic data is mounted.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

LLAMA_TAG = "b11371"
GGUF = {  # name -> (repo, revision, file); pinned 2026-10-06 from the HF API
    "12b": ("ggml-org/gemma-4-12B-it-GGUF", "e3e681731089efaa3f0917336944ac64752db8ba", "gemma-4-12B-it-Q8_0.gguf"),
    "31b": ("unsloth/gemma-4-31B-it-GGUF", "c1ac76e99d5513b141e8adde7288b85c3f9c32ec", "gemma-4-31B-it-Q8_0.gguf"),
    "31b-q4": ("unsloth/gemma-4-31B-it-GGUF", "c1ac76e99d5513b141e8adde7288b85c3f9c32ec", "gemma-4-31B-it-Q4_K_M.gguf"),
}
OURS_26B = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"  # models volume root (jevlike modal_app.py)

app = modal.App("optjev-v12")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("optjev-results", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("libgomp1")
    .pip_install("numpy", "requests")
    .add_local_dir("jevlike/decide", remote_path="/root/decide")
    .add_local_dir("jevlike/bench", remote_path="/root/bench")
    .add_local_dir("jevlike/data", remote_path="/root/data")
    .add_local_file("v12/think_tail.py", remote_path="/root/bench/think_tail.py")
)
BUILD_DIR = f"/models/llama-{LLAMA_TAG}"
SERVER_BIN = f"{BUILD_DIR}/bin/llama-server"
PORT = 8080


@app.function(image=cpu_image, volumes={"/models": models}, timeout=60 * 90, cpu=4)
def download(name: str = "12b"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    repo, rev, fname = GGUF[name]
    if os.path.exists(f"/models/v12/{fname}"):
        return {"name": name, "cached": True, "gb": round(os.path.getsize(f"/models/v12/{fname}") / 1e9, 2)}
    t0 = time.time()
    p = hf_hub_download(repo, fname, revision=rev, local_dir="/models/v12")
    models.commit()
    out = {"name": name, "path": p, "gb": round(os.path.getsize(p) / 1e9, 2), "s": round(time.time() - t0)}
    print(out)
    return out


def _model_path(name):
    return f"/models/{OURS_26B}" if name == "26b" else f"/models/v12/{GGUF[name][2]}"


CTX = {"31b": "8192"}  # 31B Q8 (32.6 GB) + --swa-full KV at 16384 OOMs a 48 GB L40S (12.8 GB KV); 8192 = 2048 per slot


def _start(model_path, extra=(), think=False, ctx="16384", np_="4"):
    """think=True (E4) drops --reasoning-budget 0, which would force the thought channel shut during generation.
    The letter readout goes through /completion with our own template, so stage A is unaffected by that flag."""
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    budget = [] if think else ["--reasoning-budget", "0"]
    cmd = [SERVER_BIN, "-m", model_path, "-ngl", "99", "-c", ctx, "-np", np_, "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", *budget, "--swa-full", "--metrics", *extra]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    for _ in range(1500):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, round(time.time() - t0, 1)
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def _bench(name, which, args, out_sub):
    import sys
    sys.path.insert(0, "/root")
    mod = __import__(f"bench.{which}", fromlist=["main"])
    out_dir = f"/results/v12/{out_sub or which}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    ret = mod.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=args, commit=results.commit, model=name, backend="llama")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
    with open("/results/v12/_run.jsonl", "a") as f:
        f.write(json.dumps({"which": which, "out_sub": out_sub, "name": name, "args": args, "gpu": smi, "model": _model_path(name),
                            "server": (ver.stdout + ver.stderr).strip()[-160:], "bench_s": round(time.time() - t0, 1),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    results.commit()
    return json.loads(json.dumps(ret, default=str))


def _run(name, which, args, out_sub="", np_="4"):
    """np_: fewer slots = longer context per slot (llama-server splits -c across slots). Used for the JevBench reruns:
    some public items are ~3.9k tokens, which exceeds a 2048-token slot (31B, ctx 8192 / 4) or 4096 + a 512-token thought (E4)."""
    os.makedirs("/results/v12", exist_ok=True)
    proc, start_s = _start(_model_path(name), think=(which == "think_tail"), ctx=CTX.get(name, "16384"), np_=np_)
    try:
        return {"server_start_s": start_s, "ret": _bench(name, which, args, out_sub)}
    finally:
        proc.terminate()


# The suite: smoke gate first (stop if it fails), then D0, D2, JevBench, latency. One server start.
SUITE = [
    ("smoke", "", "smoke"),
    ("accuracy", "--control-n 0 --workers 4", "d0"),
    ("accuracy", "--data-dir /root/data/blind/D2 --control-n 0 --workers 4", "d2"),
    ("jevbench", "--workers 4", "jevbench"),
    ("latency", "--n 200 --warmup 20 --only L1,L4", "latency"),
]


def _suite(name, skip=""):
    os.makedirs("/results/v12", exist_ok=True)
    proc, start_s = _start(_model_path(name), ctx=CTX.get(name, "16384"))
    out = {"server_start_s": start_s}
    try:
        for which, args, sub in SUITE:
            if sub in skip.split(","):
                continue
            print(f"=== {name}: {sub}", flush=True)
            out[sub] = _bench(name, which, args, sub)
        return out
    finally:
        proc.terminate()


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l4(name: str = "12b", which: str = "smoke", args: str = "", out_sub: str = "", slots: str = "4"):
    return _run(name, which, args, out_sub, slots)


@app.function(image=image, gpu="L40S", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l40s(name: str = "31b", which: str = "smoke", args: str = "", out_sub: str = "", slots: str = "4"):
    return _run(name, which, args, out_sub, slots)


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 4)
def suite_l4(name: str = "12b", skip: str = ""):
    return _suite(name, skip)


@app.function(image=image, gpu="L40S", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 4)
def suite_l40s(name: str = "31b", skip: str = ""):
    return _suite(name, skip)
