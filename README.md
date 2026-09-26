# optjev — 用 jevlike 當第二個選股器，在 optscnr 的訊號 log 上校準「賭／不賭」

上游：[jevlike](https://github.com/clarencechien/jevlike)（typed decision：讀選項字母第一個 token 的 logprob）、[optscnr](https://github.com/clarencechien/optscnr)（每日 US 選擇權掃描 + T+5/10/20 影子追蹤）。
optjev **只讀** optscnr，永不寫回（它的紅線 1、4、7、9）。規劃與凍結門檻：`PLAN.md`。

## 狀態

| 階段 | 內容 | 狀態 |
|---|---|---|
| P0 | 骨架：`jevlike/` submodule、`tasks/opt_tasks.json`、`build_dataset.py`、`modal_app.py`、`analyze.py` | 做完 |
| P1 | 回溯校準：1,467 筆訊號 × 2 題 × 4 變體，Modal L4；判定 G0–G2b | 見 `results/01-retro.md` |
| P2 | 前瞻配對：每日排程、兩個選股器同進影子帳本 | 未開始（P1 判定決定） |

## 跑一次

```bash
git submodule update --init --depth 1
pip install 'modal[api-proxy-support]' numpy scikit-learn matplotlib
python3 build_dataset.py --optscnr ../clarencechien/optscnr        # 或 --raw <optscnr commit>
scripts/modal.sh run modal_app.py --which smoke                    # 20 筆，看 first token / option mass
scripts/modal.sh run --detach modal_app.py --which retro           # 全量，可 --resume
scripts/modal.sh volume get optjev-results retro results/modal/    # 拉回原始 logprobs
python3 analyze.py                                                  # → results/01-retro.md、retro.json、thresholds.lock.json
```

`scripts/modal.sh` 把本環境的 `modal` / `modal_secret` 映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`。模型檔沿用 jevlike 的 volume `gb10-decide-models`。

## 結構

- `tasks/opt_tasks.json` 兩題：`opt_play`（賭／不賭）、`opt_tier`（峰值等級高／中／低）；選項帶 optscnr playbook 判準（crit）或只有 label（nocrit）
- `build_dataset.py` → `data/opt/<mask|nomask>_<crit|nocrit>/*.jsonl`（jevlike 格式）、`features.jsonl`（LR 基線與 EV）、`MANIFEST.md`（凍結定義與樣本數）
- `modal_app.py` llama-server + jevlike `bench/accuracy.py`，結果存 volume `optjev-results`
- `analyze.py` 溫度校準、AUROC、ECE、賭籃子門檻（cal 上定）、LR 基線、2×2 配對表、G0–G2b 判定
- `results/` `01-retro.md`、`retro.json`、`thresholds.lock.json`、`modal/`（原始 logprobs）
