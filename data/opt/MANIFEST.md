# data/opt — 由 optscnr 訊號 log 建構（build_dataset.py）

來源：`optscnr local clone @ 6c1037f466a34261e8077e8ca9c79a6173716a11`，檔案 data/iv_log/signals_2026-06.json, data/iv_log/signals_2026-07.json, data/iv_log/signals_2026-08.json, data/iv_log/signals_2026-09.json；讀到 1676 筆，跳過（無檢查點或月份外）209 筆，保留 1467 筆。

## 凍結的定義

- gold `hit`：t5/t10/t20 任一檢查點 opt_price / entry_price 峰值 ≥ 2.0（同 optscnr `shadow_tracer.make_verdict`）。
- gold `tier`：A 峰值 ≥ 2 / B 1.2–2 / C < 1.2。
- EV：只在三點齊全（`multiples` 非空）的列上，用 optscnr `strategy_lab` 的六種出場政策算。
- split：cal = 2026-06/2026-07/2026-08，test = 2026-09（時間切分）。
- state 只用訊號當下欄位；`mask` 變體遮 ticker、日曆日與事件日期，只給 DTE。
- 注意：九月晚期的列只有 t5（或 t10）檢查點，峰值上限較低，各方法一體適用。

## 樣本

- cal：n=920，命中 0.157；規則 B n=132，命中 0.311；三點齊全 n=686；tier A/B/C = 144/194/582
- test：n=547，命中 0.121；規則 B n=13，命中 0.385；三點齊全 n=69；tier A/B/C = 66/96/385

## 變體

| 目錄 | 遮蔽 | 判準 |
|---|---|---|
| `mask_crit` | 遮 ticker/日期 | 選項帶 playbook 判準 |
| `mask_nocrit` | 遮 ticker/日期 | 只有 label |
| `nomask_crit` | 不遮 | 選項帶 playbook 判準 |
| `nomask_nocrit` | 不遮 | 只有 label |

## state 範例（mask）

```
標的：股票X（已隱藏）
合約：價外買權，距到期 574 天
履約價 90.0，現貨 44.23，價外 103.5%
權利金 $1.95，等級：中間（$1.5–3）
IV 52.0%
未平倉 OI 784，七日變化 784
掃描分數 9，標籤：🚨異常掃貨 🚀點火(656.5x) 🔭LEAPS
特徵：🚨異常掃貨、🚀點火、🔭LEAPS
訊號日有新聞：否
到期前排定事件：無
警告：無
```

## question 範例（crit）

```json
{
 "type": "noul",
 "instructions": "這張價外買權（call）在接下來 20 天內，權利金是否會漲到進場價的 2 倍以上？值得賭嗎？",
 "options": {
  "A": {
   "label": "賭",
   "criteria": "符合結構候選（分數 ≥ 8、到期 21–120 天、IV < 50%、價外 < 25%、OI 七日淨增）；或有異常掃貨／點火且流動性正常、無 IV 頂峰、無尾段價外警告、到期前有排定催化事件"
  },
  "B": {
   "label": "不賭",
   "criteria": "到期 ≤ 20 天或 > 120 天、IV ≥ 50%、價外 ≥ 25%、OI 七日淨減或不明、流動性稀薄、IV 頂峰、只靠單日暴量沒有 OI 跟上、末日結算"
  }
 }
}
```
