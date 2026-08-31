# Real Consultation Keyword Evaluation

將醫生 appointment notes 同診症錄音放入 `input/`，然後執行一條命令：

```bash
python3 real_consultation_keyword_evaluate.py
```

程式會自動完成以下工作：

1. 用 `.env` 指定的 Keyword Selection Model，從 appointment notes 揀選醫療關鍵字。
2. 將錄音送到 `.env` 指定的 ASR Model，並輸出 TXT transcript。
3. 比較每個 transcript 的醫療關鍵字 recall。
4. 在 `results/` 產生 Markdown report、CSV 詳細結果及可追溯的 run manifest。

正常執行不需要任何 command-line parameters。

## 1. 系統要求

- Python 3.10 或以上
- 可連接 Google Gemini API 及 Alibaba Cloud Model Studio / DashScope
- `ffmpeg`（建議安裝；大型 WAV 會自動轉成較細的 16 kHz mono AAC upload copy）

`ffmpeg` 不存在時，程式會保留並直接上傳原始錄音。

## 2. 安裝

在 project 目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Windows PowerShell 啟用 virtual environment：

```powershell
.venv\Scripts\Activate.ps1
```

## 3. 設定 `.env`

複製範例：

```bash
cp .env.example .env
```

填入以下設定：

```dotenv
GEMINI_API_KEY=your-gemini-api-key
KEYWORD_MODEL=gemini-2.5-flash

DASHSCOPE_API_KEY=your-dashscope-api-key
ASR_MODELS=qwen3-asr-flash,qwen-audio-3.0-asr-flash

INPUT_DIR=input
OUTPUT_DIR=results
```

模型選擇完全由 `.env` 控制：

- `KEYWORD_MODEL`：Gemini keyword selection model ID。
- `ASR_MODELS`：一個或多個以逗號分隔的 ASR model ID。
- 支援的 ASR model ID：`qwen3-asr-flash`、`qwen-audio-3.0-asr-flash`。
- `INPUT_DIR`、`OUTPUT_DIR` 可以是 project-relative 或 absolute path。

`.env` 已被 Git ignore。請勿把 API keys commit 入 repository。

### Optional network settings

如公司網絡需要自訂 endpoint 或 proxy，可以在 `.env` 加入：

```dotenv
DASHSCOPE_HTTP_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_NATIVE_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
DASHSCOPE_PROXY=http://127.0.0.1:10808
```

## 4. 放入 appointment notes 同錄音

每次 consultation 需要兩個同名檔案，直接放在 `input/`，不要放在子目錄。

最簡單格式是 plain-text notes：

```text
input/
├── consultation-001.appointment.txt
├── consultation-001.wav
├── consultation-002.appointment.txt
└── consultation-002.m4a
```

`consultation-001.appointment.txt` 可以直接包含醫生 notes，例如：

```text
Diagnosis: URTI
Sore throat and mild fever for two days. No shortness of breath.
PARACETAMOL 500MG prescribed. SL x 1/7.
```

亦支援現有醫療系統匯出的 JSON：

```text
input/consultation-001.record_appointment.json
input/consultation-001.wav
```

另外支援 `<id>.appointment.json`。JSON 入面的文字欄位及 HTML 內容會自動轉成可搜尋的純文字。

錄音格式支援：WAV、MP3、M4A、FLAC、AAC、OGG。

重要規則：

- appointment note 同錄音的 `<consultation-id>` 必須完全相同。
- 同一個 ID 只可以有一份 appointment note 及一份錄音。
- 空檔、欠配對或多重配對會立即報錯，避免靜默漏評。

## 5. 執行

```bash
python3 real_consultation_keyword_evaluate.py
```

首次執行會呼叫 Gemini 及 DashScope。成功後會顯示 report 的完整 path。

相同輸入再次執行時，程式會檢查 SHA-256 hash 並重用已產生的 keywords 及 transcripts，不會重複呼叫 API。appointment notes、錄音、model 或 keyword prompt 有改動時，相關 cache 會自動失效。

## 6. 輸出

預設輸出到 `results/`：

```text
results/
├── ASR-evaluation-report.md       # 人類可讀總結
├── summary.csv                    # 整體、逐 consultation、逐類別 recall
├── file_metrics.csv               # 每個 transcript 的分數
├── keyword_list.csv               # Keyword 及 accepted forms
├── keyword_results.csv            # 每個 keyword 的命中詳情
├── generated_keywords.json        # Gemini keyword cache 及來源證據
├── asr_manifest.json              # ASR cache、model 及 audio hash
├── run_manifest.json              # 本次成功執行的設定與輸入 hashes
└── transcripts/
    └── <asr-model>/
        └── <consultation-id>.txt  # ASR TXT 結果
```

CSV 使用 UTF-8 BOM，可直接用 Excel 開啟。主要輸出會先寫入 temporary file，再以 atomic replace 發佈，避免中斷執行留下半份檔案。

## 7. 評分方式及限制

- 每個 keyword group 代表一個獨立臨床概念；任一 accepted form 命中便計一次。
- Matching 會正規化 Unicode、英文字母大小寫、空白及標點。
- 指標是 appointment-grounded keyword recall，不是 CER/WER。
- 額外或錯誤的 ASR 內容不會扣分，因此 report 不代表完整臨床安全驗證。
- Appointment notes 可能包含病人資料。錄音會送到 DashScope，notes 會送到 Gemini；使用前必須確認機構的私隱、同意及資料保存要求。

## 8. 測試

```bash
python3 -m unittest test_real_consultation_keyword_evaluate.py -v
```

## 9. 進階維護命令

一般用戶不需要以下 parameters。維護時可以強制重新產生 cache：

```bash
python3 real_consultation_keyword_evaluate.py --refresh-keywords
python3 real_consultation_keyword_evaluate.py --refresh-asr
```

亦可用 `--model-output [LABEL=]PATH` 離線評估現有 TXT transcripts；完整選項可用：

```bash
python3 real_consultation_keyword_evaluate.py --help
```
