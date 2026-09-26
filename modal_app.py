"""Modal entry point for optjev: llama-server (Gemma 4 26B-A4B, same volume as jevlike) + jevlike's bench.accuracy on data/opt.

Usage (via scripts/modal.sh so token env vars are mapped):
  scripts/modal.sh run modal_app.py --which smoke                      # 20 rows of one variant, prints acc + first tokens
  scripts/modal.sh run --detach modal_app.py --which retro             # P1: every variant x task, resumable
  scripts/modal.sh volume get optjev-results / results/modal/
P2 (daily schedule) is not wired yet: add a function with schedule=modal.Cron(...) once daily_pick.py exists.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

MODELS = {  # same files as jevlike/modal_app.py, already in volume gb10-decide-models
    "26b": ("unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
    "e4b": ("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q8_0.gguf"),
}
GPU = os.environ.get("OPTJEV_GPU", "L4")
SERVER_BIN = "/app/llama-server"
VARIANTS = ["mask_crit", "mask_nocrit", "nomask_crit", "nomask_nocrit"]
TASKS = ["opt_play", "opt_tier"]

app = modal.App("optjev")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=True)
results = modal.Volume.from_name("optjev-results", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    .entrypoint([])
    .pip_install("huggingface_hub", "numpy", "requests")
    .add_local_dir("jevlike/decide", remote_path="/root/decide")
    .add_local_dir("jevlike/bench", remote_path="/root/bench")
    .add_local_dir("data/opt", remote_path="/root/data/opt")
)


@app.function(image=image, volumes={"/models": models}, timeout=60 * 60)
def download(model: str = "26b"):
    from huggingface_hub import hf_hub_download
    repo, fname = MODELS[model]
    p = hf_hub_download(repo, fname, local_dir="/models")
    models.commit()
    return {"path": p, "bytes": os.path.getsize(p)}


def start_server(model_file, n_parallel=4, ctx=8192, port=8080, extra=()):
    cmd = [SERVER_BIN, "-m", f"/models/{model_file}", "-ngl", "99", "-c", str(ctx), "-np", str(n_parallel),
           "--port", str(port), "--host", "127.0.0.1", "--jinja", "--reasoning-budget", "0", "--metrics", *extra]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    for _ in range(900):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


@app.function(image=image, gpu=GPU, volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_retro(variants: list[str], tasks: list[str], limit: int = 0, workers: int = 4, model: str = "26b", server_extra: str = "--swa-full", out_root: str = "retro"):
    """For each variant: bench.accuracy.main(data_dir=/root/data/opt/<variant>) -> /results/<out_root>/<variant>/<task>.jsonl (raw logprobs)."""
    import sys
    sys.path.insert(0, "/root")
    if not os.path.exists(f"/models/{MODELS[model][1]}"):
        raise RuntimeError(f"model file missing in volume: {MODELS[model][1]} (run --which download)")
    t_start = time.time()
    proc = start_server(MODELS[model][1], n_parallel=workers, extra=tuple(server_extra.split()))
    t_ready = time.time()
    summary = {}
    try:
        from bench import accuracy
        for v in variants:
            out_dir = f"/results/{out_root}/{v}"
            os.makedirs(out_dir, exist_ok=True)
            args = f"--data-dir /root/data/opt/{v} --tasks {','.join(tasks)} --control-n 0 --workers {workers} --resume 1 --limit {limit}"
            print(f"=== variant {v}: {args}", flush=True)
            summary[v] = accuracy.main(base_url="http://127.0.0.1:8080", out_dir=out_dir, args=args, commit=results.commit, model=model, backend="llama")
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        props = json.load(urllib.request.urlopen("http://127.0.0.1:8080/props", timeout=10))
        with open(f"/results/{out_root}/_run.json", "a") as f:
            f.write(json.dumps({"variants": variants, "tasks": tasks, "limit": limit, "gpu": smi or GPU, "workers": workers, "model": model,
                                "model_file": MODELS[model][1], "server_extra": server_extra, "build_info": props.get("build_info"),
                                "server_start_s": round(t_ready - t_start, 1), "bench_s": round(time.time() - t_ready, 1),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        results.commit()
        return summary
    finally:
        proc.terminate()


@app.local_entrypoint()
def main(which: str = "smoke", variants: str = ",".join(VARIANTS), tasks: str = ",".join(TASKS), limit: int = 0, workers: int = 4, model: str = "26b"):
    if which == "download":
        print(json.dumps(download.remote(model), indent=2))
    elif which == "smoke":
        ret = run_retro.remote(["mask_crit"], tasks.split(","), limit=20, workers=workers, model=model, out_root="smoke")
        print(json.dumps(ret, indent=2, ensure_ascii=False, default=str))
    elif which == "retro":
        ret = run_retro.remote(variants.split(","), tasks.split(","), limit=limit, workers=workers, model=model, out_root="retro")
        print(json.dumps(ret, indent=2, ensure_ascii=False, default=str)[:4000])
    else:
        raise SystemExit(f"unknown --which {which}")
