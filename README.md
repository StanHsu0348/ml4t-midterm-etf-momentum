# 跨資產 ETF 動能 × 波動狀態 — ML4T 期中 Proposal

完整走一次 ML4T 流程：100 檔跨資產 ETF（2006–2025，Yahoo Finance），月頻、只做多的排名策略。
流程依序是：假說 → `setup.yaml` v1 → 時間點正確的資料 → 標籤與特徵篩選 → 規則 / Ridge / LightGBM 階梯 → 含成本與風控的回測 → 洩漏稽核 → **只跑一次的 holdout**。

| 繳交物 | 檔案 |
|---|---|
| Proposal 文件（PDF，正文 10 頁 + 附錄） | [`report/proposal.pdf`](report/proposal.pdf)（原始檔 `report/proposal.md`，數字由 `results/tables` 自動填入） |
| 簡報（11 張，含 1 張備用；附講稿 notes） | [`slides/midterm_presentation.pptx`](slides/midterm_presentation.pptx) |
| 交易設定 v1 | [`setup.yaml`](setup.yaml) |
| 研究帳本 / 執行紀錄 | [`run_log.csv`](run_log.csv)（196 列 run，93 個試驗） |
| Holdout 存取紀錄 | [`results/holdout_access.log`](results/holdout_access.log) |

## 結果摘要

| | 驗證 2015–2023 | Holdout 2024–2025 |
|---|---|---|
| 日 IC [NW 95% CI]（LightGBM） | 0.078 [0.042, 0.113] | 0.064 [−0.014, 0.142] |
| 淨 Sharpe [95% CI] | 0.63 [−0.02, 1.29] | 1.08 [−0.31, 2.47] |
| 12-1 規則 / 60/40 淨 Sharpe | 0.46 / 0.76 | 1.35 / 1.39 |
| Deflated Sharpe（N = 93） | 0.28 | 0.17 |

預先登錄的確認檢定（holdout IC > 0 且淨 Sharpe > 0）**通過**，但兩段淨 Sharpe 的信賴區間都含 0：Chapter 20 型態為 **never resolved**。在 holdout 裡，部署模型輸給規則與 60/40（portfolio-translation drift）。另外，ML 的相對優勢對資料雜訊不穩健（見下方「數字可能不同的原因」）。唯一的下一步：分數先做 3 個月指數平滑以降低換手，再重新測試投組關卡（setup v2）。

## 重現方式（從 repo 根目錄執行）

需要 [uv](https://docs.astral.sh/uv/)（或用 `pip install -r requirements.txt` + Python 3.12）。

**A. 重現 holdout 與總表（建議的評分流程）**：使用已 commit 的凍結規格與開發期產物，不重跑 01–05。

```bash
uv sync                                        # 建立 .venv（Python 3.12，版本鎖在 uv.lock）
uv run python scripts/download_data.py         # 重新下載資料 -> data/raw/（約 10 秒）
HOLDOUT_RERUN=1 uv run python scripts/run_notebooks.py 06 07   # holdout 與結果（記為新的 touch）
uv run python scripts/build_report.py          # 產生 report/proposal.pdf（預設用 macOS 的 Google Chrome；其他系統設定 CHROME=路徑，否則只產生 HTML）
uv run python scripts/build_slides.py          # 產生 slides/midterm_presentation.pptx
```

最終審查從乾淨的 clone 跑過這個流程：holdout 淨 Sharpe 1.10（原值 1.08），確認檢定同樣通過。

**B. 重跑開發期（01–05）**：`uv run python scripts/run_notebooks.py`（不帶參數時只跑 01–05，約 3 分鐘）。注意：
- 這會覆寫 `results/artifacts/frozen_hyperparams.json` 和 `deployed.json`。之後的 06 會因為凍結規格的雜湊不符而拒絕執行，用 `git checkout results/` 可以還原。
- 每次重跑都會在 `run_log.csv` 新增列；資料快照不同時，試驗數 N 可能增加，DSR 也會跟著改變。
- notebook 02 的課本快照比對需要 `../machine-learning-for-trading-main/data/etfs/market/etf_universe.parquet`；找不到時 `cross_check` 為 null，其餘不受影響。

**Holdout 閘門**：只要 `results/holdout_access.log` 已有紀錄，**沒有** `HOLDOUT_RERUN=1` 時 notebook 06 會拒絕執行；重跑也會被寫進紀錄。notebook 06 會重新計算凍結規格的雜湊（`results/artifacts/frozen_spec.json`）。只要 `src/`、notebooks 01–05、`setup.yaml`，或 `results/artifacts/frozen_hyperparams.json` / `deployed.json` 在凍結後改過，就會拒絕執行。

**數字可能不同的原因**：(1) 每次從 Yahoo 下載，調整因子都有約 10⁻⁶ 的浮點差異（同一天下載也一樣）；實際成交價和成交量則完全相同。原始資料是 2026-11-08 15:19 UTC 下載的，快照雜湊 `e85bda8973a2` 記錄在 `data/raw/metadata.json` 和帳本每一列。**這麼小的差異就足以改變 Optuna/LightGBM 的選擇**：用新下載的資料重跑 01–05，實測 LightGBM 驗證 IC 由 0.078 變成 0.065，部署模型由 LightGBM 變成 12-1 規則。這是本研究的發現之一，報告第 5 節有討論。(2) LightGBM 已設定 `deterministic=True`、`seed=42`，但不同 CPU 或版本仍可能有很小的浮點差異。(3) Optuna 用固定 seed 的 TPE，可重現。原始 Yahoo 資料依授權不放進 git（`data/raw/` 在 `.gitignore` 中）。

## 程式庫地圖

```
setup.yaml                     交易設定 v1：假說、股票池、時程、成本、驗證協定、確認檢定
src/etfmom/
  config.py                    路徑與 setup 載入
  data.py                      下載、清理（標記並排除）、資格、成本；load_panel() 只回傳開發期資料，
                               load_panel_with_holdout() 是通往 holdout 的唯一入口（需凍結規格、有紀錄）
  features.py                  標籤與 22 個候選特徵（REGISTRY ＝ 特徵假說表）
  models.py                    walk-forward fold、purge、選擇窗末端 purge、Rule/Ridge/LightGBM/集成
  backtest.py                  逐日事件迴圈回測（隔日開盤成交、權重漂移、逐檔成本、配置器、波動目標）
  metrics.py                   IC + Newey-West、BH、Sharpe CI、PSR/DSR、MinTRL、回撤
  ledger.py                    研究帳本（Strategy → Trial family → Trial → Run）
notebooks/                     01_setup … 07_results（.py 為 jupytext 原始碼，.ipynb 為執行後結果）
scripts/                       download_data / run_notebooks / freeze_spec / build_report / build_slides
results/tables, results/figures, results/artifacts/frozen_spec.json
report/, slides/
```

## 研究歷程（git log 即為時間軸）

所有階段都在 2026-11-08 同一個工作階段內完成，不是橫跨六週。能證明「協定先於結果」的是 commit 的順序，git 歷史沒有改寫過。W1 先 commit 設定、協定和空的帳本，之後依序是 W2 資料、W3 特徵篩選、W4 模型、W5 回測。接著在 holdout 前做了多代理人洩漏稽核：確認 3 個中度問題與 7 個低度問題，全部修正後重跑 03–05。然後凍結規格、跑一次 holdout，最後撰寫報告。稽核的發現與修正前後的差異寫在報告附錄 A3。
