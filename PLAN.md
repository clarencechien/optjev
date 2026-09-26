# optjev — 用 jevlike 做選擇權「快選器」的可行性規劃（只計劃，未動手）

日期：2026-09-26。上游：`clarencechien/jevlike`（typed decision 引擎）、`clarencechien/optscnr`（每日 US 選擇權掃描 + 影子追蹤）。

## 0. 一句話

**可以做，但不是「讓模型猜漲跌」，而是「讓模型當第二個選股器，用 optscnr 已累積的 1,676 筆訊號（755 筆已到 T+20）校準它的信心，再和規則 B 配對比較」。** 回溯校準一次不到 1 美元、一天內做完；獨立選股的配對比較要往前跑三個月才有答案。跑的地方建議 Modal 原生排程（每天約 5 分鐘 L4，每月約 2 美元，在免費額度內），不需要 CF Worker 驅動；Worker 只留一個手動觸發鈕。

## 1. 三個問題的直接回答

### Q1 jevlike 真的有東西嗎？能做「賭／不賭／勝率幾成」的快選器嗎？

jevlike 有的東西（`results/REPORT.md`）：
- 讀第一個 token 的選項 logprob，一次 forward 就得到一個分布，L4 上單題 137 ms。
- 有清楚判準（criteria）的封閉題，26B 零樣本七類 ≥ 98%。
- **raw 信心沒有鑑別力**（平均 0.99，溫度要拉到 T≈6），門檻只能用校準後的信心；conformal 門檻給了「錯誤預算 → 自動處理比例」的保證。
- 弱題（判準模糊）順序敏感 20%，校準也救不回，要改判準。

套到選擇權的意思：
- jevlike 擅長的是「依照一本 playbook 判斷這個 setup 合不合格」，不是「預測價格」。所以快選器的題目要寫成 **playbook 合規／setup 品質** 的封閉題，選項帶 optscnr 的判準（規則 B、premium tier、Δ7d、事件、警告）。
- 「勝率幾成」**不能**直接拿 logprob 當機率。要用 optscnr 的 `verdict`（✅噴了 = 峰值 ≥ 2x）當 gold，做溫度校準，輸出的是「校準後 P(噴)」，而且只顯示成粗桶（二成／三成／四成以上），不顯示假精確的小數。
- 「賭／不賭」= 校準後信心過門檻。門檻不是 jevlike 的 conformal 錯誤預算（基礎命中率只有 17.6%，錯誤預算會變成 70% 沒意義），而是「賭的那一籃子命中率 ≥ 目標」反推的門檻，在 held-out 上驗證。

**能不能做的判準是實驗結果，不是現在能拍板的。** 先寫死門檻（§4），跑回溯校準，沒過就不做。

### Q2 拿 optscnr 累積的資料，讓「另一個 jevlike 選出」再 pair，有可能嗎？

資料盤點（optscnr）：

| 資料 | 筆數 | 有結果？ |
|---|---|---|
| `data/iv_log/signals_2026-{06..09}.json`（score ≥ 8 的訊號） | 1,676 筆，755 筆 T+20 成熟；verdict ✅210 ➖880 ❌246 💀131 | **有**：t5/t10/t20 價格、peak、verdict、2026-08 起每日 path |
| `data/decisions/*.json`（規則 B 候選 + LLM 三題） | 11 筆候選 | 有 outcome，尚未成熟 |
| `data/strategy_matrix.json` | 每日重算 | 整體命中 0.176、EV(C_bound) 1.119；規則 B n=136、命中 0.309、EV 1.692；對照組 edge −0.1pp |
| `data/YYYY-MM-DD.csv`（每日 score > 0 全表） | 251 天、約 389k 列 | **沒有**結果（只有快照） |

所以「pair」分兩層，能做的程度不同：

1. **回溯層（現在就能做）**：只在 1,676 筆訊號上做。這些是規則掃描已經選過的，jevlike 在上面做的是「再判一次」，pair 的意思是 **規則 B 判定 × jevlike 判定** 的 2×2 表，四格各自的命中率與 EV。
2. **前瞻層（要往前跑）**：jevlike 當獨立選股器，每天讀當日全表（約 1.5k 列）自己選；規則 B 也選；兩邊的選單一起進同一本影子帳本，用同一個 tracer 補 T+5/10/20。**歷史全表無法補結果**（yfinance 沒有歷史選擇權價格），所以獨立選股只能從上線那天開始累積，和 optscnr 自己的預先登記一樣要 100 筆成熟、約三個月。

另外兩條 optscnr 的紅線決定了 optjev 必須是獨立 repo、獨立影子帳本：
- 紅線 9「LLM 只答事實題，不做可玩／跳過判斷、不填機率」→ jevlike 的判斷**不能**寫進 `data/decisions/`，只能在 optjev 自己的 log。
- 紅線 1「不做盤中即時 alert」→ 每日盤後批次，不做即時。
- 紅線 4、7「append-only、預先登記期間不改規則 B」→ optjev 只讀 optscnr，永不寫回。

### Q3 有地方跑 jevlike 嗎？CF Worker 驅動 Modal 拿數值回來？

| 方案 | 怎麼跑 | 每日成本 | 評價 |
|---|---|---|---|
| **A. Modal 原生排程（建議）** | `modal deploy`，`@app.function(gpu="L4", schedule=modal.Cron("50 23 * * 1-5", timezone="UTC"))`，排在 optscnr scanner（22:17 UTC）與 Worker decision（23:05）之後；函式內起 llama-server（模型已在 `gb10-decide-models` volume）、拉 optscnr raw、跑完把結果 commit 到 optjev（GitHub token 放 Modal secret） | 冷啟 1–2 分 + 約 2k 題 × 0.14 s ≈ 5 分 L4 ≈ $0.07；月約 $2，Starter 每月 $30 免費額度內 | 最少零件；不用 Worker 也能跑；Modal 儀表板有「run now」 |
| B. CF Worker → Modal HTTP 端點 | Modal 開 `@modal.fastapi_endpoint()`，Worker cron POST；GPU 工作用 `.spawn()` 回 call id，結果由 Modal 端自己寫回 GitHub，Worker 不等 | 同 A | 只有想「在 dashboard 按一下手動重跑」時才需要；建議 A + 一個手動端點，不要用 Worker 當主排程 |
| C. GB10 本機 + cloudflared tunnel | jevlike 已有 `bench/run_local.py`；Worker 或 Modal 直接打 tunnel | 電費 | GB10 到位後的遷移路徑；排程時間機器要開著 |
| D. Workers AI / AI Studio | — | — | **不行**：都不給 logprob（jevlike `PLAN.md` 實測 AI Studio 回 `Logprobs is not enabled`） |

「CF Worker 驅動 Modal 跑完拿數值回來」技術上可行，但 Worker 同步等 GPU 冷啟會撞 Worker 自己的執行時間上限，一定要走 spawn + 回寫；既然要回寫 GitHub，讓 Modal 排程自己做就好，Worker 只讀。

## 2. 設計

### 2.1 資料流（optjev 只讀 optscnr，永不寫回）

```
optscnr (GitHub raw)                          optjev (本 repo)
  data/iv_log/signals_*.json  ──回溯──▶  build_dataset.py ──▶ data/opt/*.jsonl（jevlike 格式）
  data/YYYY-MM-DD.csv         ──每日──▶  daily_pick.py     ──▶ log/picks/YYYY-MM-DD.json（規則B + jev 兩欄）
  shadow_tracer.py 的 T+N 邏輯 ──借用──▶  tracer.py         ──▶ log/picks/*.json 補 outcome（append-only）
                                            analyze.py        ──▶ results/*.md、thresholds.lock.json
```

### 2.2 題目（task family `opt_*`，格式同 jevlike `data/seeds/tasks.json`）

| task | 選項 | gold 來源 | 用途 |
|---|---|---|---|
| `opt_play` | A 賭 / B 不賭 | verdict ✅ → A，其餘 → B | 賭／不賭 |
| `opt_tier` | A 高 / B 中 / C 低 | peak ≥ 2x → A；1.2–2x → B；< 1.2x → C | 勝率桶（顯示用） |
| `opt_exit`（選配） | A 抱到 DTE21 / B 2x 賣半 / C 綁定 | 每筆用 path 算三種出場的倍數取最大 | 出場建議；需重算 path，第二階段 |

選項的 criteria 寫 optscnr 的 playbook（規則 B 條件、tier、Δ7d 狀態、事件與警告）。另做一組**無 criteria** 的變體當消融：看模型是抄規則還是有多的判斷。

### 2.3 state（只能用訊號當下看得到的欄位）

允許：`entry_price / entry_bid / entry_ask / entry_iv / entry_spot / strike / expiry(換算 DTE) / oi / oi_d7 / oi_delta_status / volume / score / tags / features / premium_tier / structural_pass / news_at_signal / events / warnings / why_it_popped / signal_day_underlying_move`。
**禁止**：`t5 / t10 / t20 / path / verdict / sell_points / outcome`。

**洩漏控制（G0）**：ticker 與日期要遮（`股票X`、只給 DTE 不給日曆日）。Gemma 4 的訓練資料可能涵蓋 2026 年中，不遮就可能靠記憶答。遮／不遮各跑一次，差 > 2 點就只認遮的版本。

### 2.4 切分

**時間切分，不用隨機切**：cal = 2026-06/07/08 成熟訊號（約 500 筆），test = 2026-09 成熟訊號。隨機切會讓同一天、同一 ticker 的孿生訊號跨到兩邊，命中率虛高。

### 2.5 基線（沒有這個，結果無法解讀）

- 規則 B 本身（同一 test 視窗的命中率與 EV，不是全期的 0.309）。
- 同 15 個結構欄位上的 logistic regression / 小型 GBM。jevlike 對 SMT 題也做過 TF-IDF+LR 基線；**若 LR 就打得贏 jevlike，答案是「用 LR，不用 GPU」**。jevlike 可能多出來的只有文字欄位（`why_it_popped`、`events`、`warnings`）的理解。

## 3. 階段

| 階段 | 做什麼 | GPU | 時程 |
|---|---|---|---|
| P0 骨架 | optjev repo：`decide/` 用 git submodule 指 jevlike；`tasks/opt_tasks.json`；`build_dataset.py`；`modal_app.py`（複製 jevlike 的，加 schedule）；本文件 §4 門檻凍結 | 0 | 半天 |
| P1 回溯校準 | 1,676 筆 × 2 題 × (遮/不遮) × (有/無 criteria) ≈ 13k 題 ≈ 30 分 L4 ≈ $0.4；`analyze.py` 做溫度校準、AUROC、ECE、2×2 配對表、LR 基線 | ≈ 0.6 h | 一天 |
| P1 判定 | 依 §4 G0–G2 三選一：上線快選器 / 只當第二意見 / 不做 | | |
| P2 前瞻配對 | 每日 Modal 排程；兩個選股器同時進影子帳本；tracer 補結果；dashboard 一頁（可掛在 optscnr Worker `/api/brief` 旁邊，但資料來自 optjev） | 每日 ≈ 5 分 | 三個月累積 |
| P2 判定 | 100 筆成熟後看 G3 | | |

## 4. 預先登記的門檻（跑前寫死，跑完照填）

| 編號 | 門檻 | 沒過的意思 |
|---|---|---|
| G0 洩漏 | 遮 ticker/日期 vs 不遮，`opt_play` test acc 差 ≤ 2 點 | 差 > 2 點：模型在用記憶，只認遮的版本 |
| G1 鑑別力 | 校準後 P(噴) 在 test 上 AUROC ≥ 0.65 | < 0.60：跟擲硬幣一樣，**不做** |
| G2 賭籃子 | test 視窗內，「賭」的子集命中率 ≥ 同視窗規則 B 命中率 + 5 個百分點，且子集 ≥ 30 筆、coverage ≥ 20%；EV(C_bound) ≥ 規則 B 同視窗 EV | 沒過但 G1 過：只當「第二意見」，不當快選器 |
| G2b 基線 | jevlike 賭籃子命中率 ≥ LR 基線 + 3 個百分點 | 沒過：用 LR 取代 GPU |
| G3 前瞻 | 100 筆成熟配對後，jev-only、rule-only、both-agree 三籃子各報命中與 EV；both-agree ≥ rule-only + 5pp 才算「配對有加分」 | 沒過：拆掉排程，留報告 |
| 順序 | `bench/permute.py` 量一次；≥ 20% 就固定選項順序並註明 | |

## 5. 不做的事

- 不改 optscnr 任何檔案、不寫回它的 `data/`、不碰規則 B 門檻（紅線 3、4、7）。
- 不做即時（紅線 1）、不做個人損益（紅線 2）。
- 不 fine-tune、不改 jevlike 的模板（`SYSTEM`、`<bos>`、空 thought channel 原樣）。
- 不在 P1 沒過 G1 的情況下進 P2。

## 6. 風險（誠實版）

1. **最可能的結果**是 G1 勉強過、G2b 沒過：模型讀結構欄位的能力不會比 LR 強，值錢的只有文字欄位，而文字欄位目前很薄（`why_it_popped` 一句話）。
2. 九月的環境很差（optscnr 自己九月命中 12%、EV 0.89x），test 視窗只有九月會把所有方法都打成沒 edge；要報同視窗相對值，不報絕對值。
3. 合成資料到真實資料掉 5–15 點是 jevlike 自己寫的預期；這裡連合成資料都沒有，是直接真實資料零樣本。
4. 出場題（`opt_exit`）要從 path 重算三種政策的倍數，2026-08 以前沒有 path，樣本會少一半，所以放第二階段。

## 7. 參考

- jevlike：`README.md` §1–3、`results/REPORT.md` §0、§3f、`results/11-thresholds.md`、`bench/thresholds.py`、`docs/handoff-gb10.md` §5
- optscnr：`CONTEXT.md` 第九節（紅線）、`strategy_lab.py:58-69`（規則 B）、`shadow_tracer.py:176-197`（verdict 邏輯）、`data/strategy_matrix.json`、`cloudflare/src/index.js:372-458`（decision 流程）
- Modal：`@modal.fastapi_endpoint()`、`schedule=modal.Cron(..., timezone=...)` 需 `modal deploy`；L4 $0.000222/s（≈ $0.80/h）；Starter 每月 $30 免費額度
