# Handoff v12 — 借 Gemma 4 新做法的四件事（2026-10-06，跑前寫死）

給接手 jevlike 的 Claude Code。本檔寫在 optjev（`docs/handoff-jevlike-v12.md`），因為產生它的工作階段只能推 optjev；
**搬進 jevlike 的 `docs/handoff-v12-borrowed-gemma.md` 後照做**，結果放 jevlike 的 `results/17-v12*.md`。
以下所有路徑都是 jevlike repo 內的路徑。

來源：2026-10-06 的生態調查（摘要在 §7）。這十天 Gemma 4 這條線的變化是：官方 JevBench v1.5.4 第一名 Cygnet 是**凍結的 Gemma 4 12B**，
並列第二的 Winnow-12B、Jev-Omni 也是 12B；Decision Index 上凍結的 Gemma 4 31B（decisio 0.6.0）比凍結 12B 高 8 分；
Surogate Rune 在跟我們同一顆 26B-A4B 上做了「低信心才思考」。本輪只借零訓練能用的四件：

| 編號 | 問題 | GPU | 估時／費用 |
|---|---|---|---|
| E1 | 順序型題目改讀「機率加權等級」，比 argmax 準嗎？ | 不用 | 半天 / $0 |
| E2 | 溫度改成「每題型一個」，能取代「每 task 一個」嗎？ | 不用 | 半天 / $0 |
| E3 | 階梯補 Gemma 4 12B 與 31B：12B 能當級聯第一階嗎？31B 能當弱題候選嗎？ | L4 + L40S | 半天 / ≈ $3 |
| E4 | 26B 只在低信心時先想 ≤ 512 token 再讀字母，弱題會變準嗎？ | L4 | 半天 / ≈ $1 |

順序：E1、E2 先做（離線，結果決定 E3、E4 用哪個溫度）；E3、E4 可以同一天在 Modal 上跑。

## 0. 規則（同前幾輪，不能違反）

1. 只用合成資料（D0、D2、heldout、JevBench 公開題）。真實資料不上 Modal。
2. 每個實驗先照本檔門檻跑，跑完在 §結論三選一，沒過也照實寫。門檻不得事後改。
3. **不改 prompt 模板**（`decide/prompt.py` 的 `SYSTEM`、`GEMMA4_TEMPLATE_NOTHINK_BOS`、`ANSWER_LINE`）。E4 只在觸發的列上多一段思考，未觸發的列輸入與現在逐 byte 相同。
4. 不訓練。不追 Winnow-12B、Rune、Jev-Omni 這類訓練過的模型（v11 的 jevify 已量過：英文通用 LoRA 在我們的中文題上低 2–3 點）。
5. 分析腳本 `bench/analyze_v12.py`，結果 `results/17-v12.md` + `results/v12.json`；每個實驗做完 commit + push。
6. 切分沿用 `bench/analyze.py` 的 `group_split(rows, seed=0)`（pair_id 分組，cal/test 各半，D0 test 每 task 100 筆）。
7. 門檻方法沿用 v9 修正後的 `bench/thresholds.py`（選擇性風險控制：cal 上加一修正錯誤率 ≤ ε 的最低信心切點）。

## 1. 共用輸入（都已在 repo，不重跑）

| 資料 | 路徑 | 內容 |
|---|---|---|
| 26B D0 | `results/modal/accuracy/<task>.jsonl` | UD-Q4_K_M、b11118；每列 `raw_logprobs`（top-40 內每個字母）、`probs`、`gold`、`pair_id` |
| 26B D0（新 build） | v11 的 b11371 重跑目錄（見 `results/16-clef-llamacpp.md` §D） | 1,998/2,000 與上列同答，E1/E2 以上列為主、這份當重現 |
| E4B / E2B D0 | `results/modal/accuracy/e4b/`、`e2b/` | 級聯比較用 |
| heldout | `results/modal/accuracy/heldout_v1/`、`heldout_v2/` | 急迫度與 SPC 的另一組種子 |
| D2 盲寫 | `results/modal/accuracy/D2/` | Gemini 寫、Claude 沒看過的題 |
| JevBench 公開 | `results/modal/jevbench/{original,easy,hard}.jsonl` | 每列 `type`（choice / noul / score）、`fwd` / `rev` 的 `raw_logprobs` |
| 題型 | `data/seeds/tasks.json` 的 `kind` | score：`m_alarm_severity`（3 級）；noul：`m_needs_dispatch`、`p_uph_anomaly`、`x_escalate`；其餘 6 類 choice |

## 2. E1 順序型題目：機率加權等級（離線）

**為什麼**：Cygnet 對 score 題回傳「機率加權後的等級」，不取 argmax。我們最弱的急迫度正是 score 題，而且 README §3 寫明錯誤集中在相鄰等級（模型偏向較嚴重的一級）。原始 logprob 都在，不用重跑。

**做法**（每列，P = 校準後機率，溫度 T 在 cal 上用 NLL 擬合，同 `analyze.py`）：
- R0（現行）：argmax P。
- R1（主臂，Cygnet 式）：等級編號 k = 0…K−1（A 最輕），E = Σ k·P_k，預測 = round(E)（.5 往較輕的一級捨，跑前寫死）。
- R2（探索，不判門檻）：中位等級，累積機率首次 ≥ 0.5 的 k。
- R1 的信心：P[預測等級]；選擇性風險控制照 §0 第 7 條。

**資料**：D0 `m_alarm_severity` test 100 筆 + heldout 兩組的急迫度 + D2 的急迫度 + JevBench 三個檔的 score 題（fwd）。
`q_spc_action` 的四個選項也是升級階梯，**只做探索**（報數字、不判門檻），因為它被定義成 choice。

**指標**：acc、平均絕對等級誤差（MAE）、相鄰錯佔全部錯的比例、ε=5% 時的 coverage、McNemar p（R1 對 R0）。

**門檻（跑前寫死）**
- 採用：所有急迫度集合合併後 R1 acc ≥ R0 + 2 點，且 MAE 不增加，且沒有任何單一集合比 R0 差 > 2 點。
- JevBench score 題：R1 答對題數 ≥ R0 − 1（題數少，只當「沒變壞」的檢查）。
- 結論三選一：**採用**（score 題預設改 R1，`decide/` 加一個讀法函式，模板不動）／**持平**（R0 留著，R1 只記錄）／**變差**（記下原因：例如偏向中間級）。

## 3. E2 溫度顆粒度（離線）

**為什麼**：decisio 用「選擇題一個溫度、其他一個」；我們是每 task 一個。新 task 上線時沒有標註，題型溫度能直接套用就省掉校準。v8 已看到合成資料的溫度搬到 JevBench 有用（ECE 0.093 → 0.044）。另外有幾個 task 的 cal 幾乎全對，擬出 T=0.05 這種退化值（`11-thresholds.md`），題型溫度可能更穩。

**臂**
- T_task（現行）：每 task 在自己 cal 上擬。
- T_kind（主臂）：choice / noul / score 各一個，在該題型所有 task 的 cal 合併上擬。
- T_global：全部 cal 合併擬一個。
- T_loo（主臂的「新題」模擬）：對每個 task，用**其他** task 的同題型 cal 擬溫度（leave-one-task-out）。score 題只有一類，T_loo 對它用 T_global 的 leave-one-out。

**指標**（D0 test，每 task 再平均）：NLL、ECE（top_prob，10 bins）、ε=5% 守住的 task 數與平均 coverage。轉移：D0 cal 擬的 T_kind 套到 JevBench（依 `type`）的 ECE。

**門檻（跑前寫死）**
- **T_loo 可當新題預設**：平均 ECE ≤ T_task + 0.01，ε=5% 守住數與 T_task 相同（10/10），平均 coverage 不少於 T_task − 3 點。
- **T_kind 取代 T_task**：上一條成立，且 T_kind 的平均 NLL ≤ T_task 的平均 NLL。
- JevBench 轉移：T_kind 轉過去的 ECE ≤ 0.044（v8 單一溫度的數字）。
- 結論三選一：**T_kind 取代**／**T_loo 只給新題用、舊題維持 T_task**／**維持 T_task**。

E3、E4 的校準一律用 E2 判定後的預設溫度方法。

## 4. E3 階梯補 12B 與 31B（GPU）

**為什麼**：官方 JevBench 第一名是凍結 12B（Cygnet），我們的階梯只有 E2B、E4B、26B。Decision Index 上凍結 31B 比凍結 12B 高 8 分（57.58 對 49.43）。31B 可以跟 Clef 27B（D0 96.6%，SPC 97%）搶弱題候選的位置，而且不用新的推論程式。

**檔案（2026-10-06 查 HF API，全部釘 revision）**

| 模型 | repo @ revision | 檔案 | 大小 | 卡 |
|---|---|---|---|---|
| 12B | `ggml-org/gemma-4-12B-it-GGUF` @ `e3e681731089efaa3f0917336944ac64752db8ba` | `gemma-4-12B-it-Q8_0.gguf` | 12.67 GB | L4 |
| 31B | `unsloth/gemma-4-31B-it-GGUF` @ `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` | `gemma-4-31B-it-Q8_0.gguf` | 32.64 GB | L40S |
| 31B（L4 對照，選配） | 同上 | `gemma-4-31B-it-Q4_K_M.gguf` | 18.32 GB | L4 |

上游權重：12B 是 `google/gemma-4-12B-it` @ `707f0a3b…`（與 Winnow-12B 同底），31B 是 `google/gemma-4-31B-it` @ `842da379…`（與 decisio 0.6.0 同底）。
llama.cpp 用 v11 的 `b11371` build（volume 裡已有 binary）。`modal_app.py` 的 `MODELS` 加 `12b`、`31b`、`31b-q4` 三個鍵。

**步驟 0：smoke（每個模型，必須全過才往下）**
- `bench/smoke.py`：十個字母都是單 token、id 與 26B 相同（同一個 tokenizer 才相同；不同就記下來，不是錯）。
- 採用變體 5 題 first token 全是字母、`option_mass_check.ok`。
- `/apply-template` 學到的模板：12B 是 6 月出的「Unified」版，**先比對它的 chat template 與 26B 是否一樣**（空 thought channel、`<bos>`）。不一樣就照它自己的模板（`TemplateRenderer` 本來就從 server 學），並在報告 §0 記下差異。
- Cygnet 提醒的坑：llama.cpp 預設開 thinking，一定要帶 `enable_thinking: false`（我們的 renderer 已經做了，smoke 會查 option mass）。

**跑什麼**：D0（2,000 題）、D2、JevBench 公開 231 題正反序；延遲 L1（單題）與 L4（共用 state K=10），12B 在 L4、31B 在 L40S，26B 的 L40S 數字沿用 v11（56 ms）。

**門檻（跑前寫死）**
- **12B 當級聯第一階**：用 v3 的級聯邏輯（`bench/analyze_ladder.py`，12B 先答，校準後把握 < 門檻才送 26B），在送 26B 比例 ≤ 25% 的門檻下，級聯 D0 acc ≥ 全 26B − 0.3 點。對照 E4B 級聯（34% 送 26B 才到 0.952）。
- **12B 重現 Cygnet**（健全性檢查，不影響判定）：JevBench 公開 fwd 在 87.9% ± 2 點內；超出範圍就在報告寫原因（prompt 差異、state 用 `json.dumps(indent=1)` 等）。
- **31B 當弱題候選**：弱題三類（急迫度、SPC、UPH）D0 test 平均 ≥ 26B + 3 點（26B 是 87.0%），且十類平均 ≥ 26B（95.5%）。過了就與 Clef 27B 並列，到 GB10 量延遲再選。
- 結論三選一（各自）：12B **取代 E4B 當第一階**／**不取代**；31B **列為弱題候選**／**不列**。

**費用**：12B（L4，下載 + D0/D2/JevBench + 延遲）≈ 0.6 h ≈ $0.5；31B Q8（L40S）≈ 1 h ≈ $2；31B Q4 選配 ≈ $0.5。

## 5. E4 低信心才思考（26B，GPU）

**為什麼**：Rune（同一顆 26B-A4B 的微調）只在信心 < 0.7 時先想最多 512 token，約一成題目觸發，Decision Index 從 53.39 升到 54.89。
我們的弱題錯誤集中在低信心的尾巴；v5 量過 TypeLLM 的「全部開思考」每題 919 token、產線用不起，這裡只在尾巴開。

**做法（兩段，同一個 llama-server）**
1. A 段（現行）：不思考讀字母，得 P_A，用 E2 判定後的溫度方法校準。
2. 觸發條件：校準後 top 機率 < 0.7（主），另報 < 0.9 當次要。
3. B 段（只對觸發的列）：
   - 步驟 0 先確認 Gemma 4 思考格式：`/apply-template` 帶 `enable_thinking: true`，system turn 開頭會有 `<|think|>\n`；模型的思考寫在 `<|channel>thought\n … <channel|>` 之間。確認 26B 在這個模板下生成時確實先寫 thought 再關 channel。
   - 生成：思考開的模板 + state + 題目，`/completion` 的 `n_predict=512`、`temperature=0`、`stop=["<channel|>"]`。
   - 讀字母：思考開的模板前綴 + `<|channel>thought\n` + 生成的 thought + `<channel|>`，`n_predict=1` 讀字母 logprob（`read_option_probs`）。512 token 內沒關 channel 就截斷後照樣補 `<channel|>`，該列記 `truncated=true`。
   - B 段讀字母的 option mass 必須 ≥ 0.9（中位數）。不到就是模型想在 thought 後寫字，停下來修格式，不要往下跑。
4. 最終機率：觸發列用 P_B，溫度在 cal 的觸發列上另外擬一個 T_B（思考會改變信心尺度）；未觸發列維持 P_A。

**資料**：D0 弱題三類（各 200 筆，cal/test 各半）+ JevBench 公開 231 題 fwd。強題七類只跑 A 段統計觸發率，不跑 B 段。

**指標**：觸發率、B 段平均 thought token 數、截斷率、acc（觸發列與全部）、ε=5% 平均 coverage、觸發列延遲 p50/p95、翻對與翻錯的題數。

**門檻（跑前寫死）**
- **採用為弱題的慢路徑**：弱題三類 test 合併 acc ≥ A 段 + 2 點（n=300，報 McNemar p），且 ε=5% 平均 coverage ≥ A 段 + 5 點，且翻錯的題數 ≤ 翻對的一半。
- 觸發率：弱題 ≤ 30%；強題 ≤ 5%（超過表示 0.7 對強題太高，報告裡要寫）。
- JevBench 公開：觸發後整體 acc 不低於 A 段（只當沒變壞的檢查）。
- **延遲不判門檻**：L4 上 26B 生成約 30–40 token/s，512 token 會到十幾秒；只報數字，GB10 的判定寫進 `docs/handoff-gb10.md`。
- 結論三選一：**採用（弱題 opt-in 慢路徑）**／**持平（不採用）**／**變差（記下翻錯的型態）**。

**費用**：D0 弱題 600 列 × 觸發約 25% × 512 token + JevBench 約 20 分鐘，L4 約 1 h ≈ $0.8。

## 6. 本輪不做（記下理由）

- Winnow-12B、Rune v3、Jev-Omni：訓練過的模型，jevify 的教訓是英文通用訓練在我們的題上會掉分。要試以 Winnow 最便宜（有 GGUF、讀答案 token logit），但不在本輪。
- llama.cpp `/v1/systemone`（PR #29818，10/2 合併）：只支援 Laya、Julia-1、Lev、OpenJev 27B、Kev 五個專用模型，一般 Gemma 不能用，繼續用自己的 client。
- DiffusionGemma 26B：外部實測落在合法答案上的機率只有 37–59%，不能用讀 logit 的方式。
- MTPLX（Apple MLX）：Gemma 4 讀第一個 token 已可用，但每題約 1.3 秒，除非要部署在 Mac。
- **官方 JevBench 封閉題評測**：Cygnet 是開 bench request issue 送審（`fstandhartinger/jevbench` #71）。我們的 26B 零訓練版也可以送，拿一把外部的尺；要準備一個可自架的 package，不花 GPU 但要花人力，另開一輪。
- JevOut（arXiv 2609.30243）：刻意優化的自然語境能翻掉 Jev 61% 的決策，翻錯的近一半信心 ≥ 0.7，信心門檻擋不住；中性的一句話只翻 2.2%。產線機台訊息風險低，但護欄（v9 G）與任何吃外部文字的題要記住這點。

## 7. 調查來源（2026-10-06）

- JevBench v1.5.4：<https://benchlm.ai/benchmarks/jevbench>、各軸分數 <https://benchmarkheaven.com/jev-models/v1.5.4>
- Cygnet：<https://github.com/blockbrain-ai/cygnet-recipe>（字母 token 機率加總、`json.dumps(indent=1)`、noul 固定 false 在前、score 回機率加權等級、T=3.4 在 241 題自出題上擬）
- decisio 0.4.0 / 0.6.0：<https://github.com/apolinario/decision-index/pull/62>（choice T=4.672、其他 T=5.252）
- Rune 26B-A4B v3：<https://huggingface.co/michaelfeil/rune-26b-a4b>（信心 < 0.7 才思考、≤ 512 token、ECE 12.5% → T=2 後 2.2%）
- Winnow-12B：<https://huggingface.co/EldanRing/Winnow-12B>；Jev-Omni：<https://jev-ai.pro/model/jev-omni>
- llama.cpp 決策模型：<https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp>
- JevOut：<https://arxiv.org/abs/2609.30243>
- 純 Gemma 4 26B 對 Jev（L4，3,880 題）：<https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6>

## 8. 交付清單

- [ ] `bench/analyze_v12.py`（E1、E2 離線；E3、E4 讀 Modal 結果）
- [ ] E1 結論、E2 結論（決定 E3、E4 的溫度方法）
- [ ] `modal_app.py` 加 `12b`、`31b`、`31b-q4`；`bench/think_tail.py`（E4 兩段讀法）
- [ ] E3 smoke ×2、D0/D2/JevBench、延遲；E4 步驟 0 格式確認、主跑
- [ ] `results/17-v12.md`、`results/v12.json`、`results/cost.md` 加一列、README §2 加 v12 一列
