# ModelPlayGround

[English](#english) | [繁體中文](#繁體中文)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![uv](https://img.shields.io/badge/uv-managed-de5fe9)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688)
![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

<p align="center">
  <img src="docs/demo.gif" alt="ModelPlayGround Demo" width="880">
</p>

---

<a name="english"></a>
## English

**ModelPlayGround** is a pre-deployment evaluation tool for LLM features. Given a one-sentence description of the intended behaviour, it generates an adversarial test set, runs it against your model, reports per-criterion failures, proposes prompt revisions, and re-runs the full suite to detect regressions. The resulting test set and system prompt export as a provider-agnostic package.

LLM features are commonly shipped after ad-hoc manual testing, with no recorded acceptance criteria and no way to detect regressions when the prompt or the model changes. ModelPlayGround derives those criteria from the task description and keeps them runnable.

> [!NOTE]
> Runs against any OpenAI-compatible endpoint — Azure OpenAI, OpenAI, Groq, OpenRouter, Ollama. Three environment variables are required.

### Features

- **Failure-mode analysis** — classifies the task against known LLM failure modes (hallucination, over-commitment, unhandled missing input, format drift, conflicting-input blindness, dropped sub-requests, privacy leakage) and cites the phrase in the task description that triggered each classification.
- **Adversarial test generation** — produces five test cases graded from routine to adversarial. The final case targets instruction-following under pressure: it bundles multiple requests, omits required fields, and explicitly instructs the model to guess rather than ask.
- **Rubric decomposition** — the judge decomposes each pass condition into independently verifiable atomic criteria and cites supporting evidence per criterion before returning `pass = all(criteria)`. Failures identify the specific unmet requirement rather than an aggregate score.
- **Self-consistency judging** — each verdict is sampled three times and resolved by majority vote. Disagreement is reported as a vote tally rather than collapsed into a binary result; ties resolve to fail.
- **Regression-aware patching** — failures are converted into numbered prompt rules, editable before application. Applying a patch re-runs the full suite rather than only the failing cases, so a rule that fixes one case while breaking another is surfaced.
- **Provider-agnostic export** — test cases, rubric, and system prompt export as a standalone package with a promptfoo configuration. Retargeting to another provider requires changing only the `providers` field.

### Screenshots

Failure-mode analysis. Each classification cites the phrase that triggered it, alongside unresolved questions that affect the acceptance criteria:

![Failure-mode analysis](docs/risk.png)

Test cases with per-criterion verdicts. The `2/3` badge on the fourth case marks a verdict where the judge disagreed with itself across samples:

![Test cases and verdicts](docs/cases.png)

Summary. Risks are derived from the observed failures rather than a static checklist:

![Summary](docs/verdict.png)

### Export format

| File | Purpose |
|---|---|
| `SKILL.md` | Agent skill definition with YAML frontmatter |
| `system_prompt.md` | The revised system prompt |
| `evals/cases.jsonl` | Test cases and pass conditions, provider-independent |
| `evals/rubric.md` | Human-readable acceptance criteria |
| `evals/prompt.json` | Chat template referenced by the promptfoo config |
| `promptfooconfig.yaml` | Runnable via `npx promptfoo eval` |
| `run.py` | Minimal example; `--selftest` replays the full test set |

The rules section of the generated `SKILL.md` is derived from the test cases that failed, so it documents constraints specific to the task:

> - Never infer, invent, or rewrite a due date for a task — for example "tomorrow morning" or "this Friday".
> - Never treat someone as an owner based on context. Speaking up in the meeting or belonging to the relevant department is not an assignment.

### Architecture

| File | Responsibility |
|---|---|
| `app.py` | FastAPI endpoints and SSE streaming. Client-side state is posted back with each request, so the server is stateless. |
| `core/llm.py` | OpenAI-compatible client. Records time-to-first-token; drops unsupported parameters and retries on `400`, so the same code path serves Ollama and Azure OpenAI. |
| `core/prompts.py` | The seven prompts, centralised for tuning. |
| `core/pipeline.py` | `analyze` / `evaluate` / `suggest_patch` / `verdict`, separated so the patch-and-re-run loop can invoke the latter stages independently. |
| `core/judge.py` | Rubric decomposition, majority-vote tallying, bias measurement. |
| `core/calibration.py` | Cohen's kappa, bootstrap confidence interval, false-positive rate. |
| `core/store.py` | Run persistence and sampling for calibration. |
| `core/pack.py` | Export package construction. |
| `calibrate.py` | CLI for labelling samples and reporting inter-rater agreement. |
| `static/index.html` | Single-page frontend, no build step. |

### Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/breezy89757/ModelPlayGround.git
cd ModelPlayGround
uv sync
cp .env.example .env
```

Three values are required in `.env`:

```bash
LLM_BASE_URL=https://api.openai.com/v1    # or an Azure / Groq / OpenRouter / Ollama endpoint
LLM_API_KEY=sk-...
MODEL=gpt-4o-mini                          # deployment name on Azure
```

```bash
uv run uvicorn app:app --port 8000
```

Open <http://localhost:8000>. A complete run takes approximately 90 seconds and 10 API calls.

The interface is in Traditional Chinese. Generated prompts and test cases follow the language of the task description.

### Environment variables

Only the first three are required. Each role falls back to `MODEL`, so a single model is sufficient; separating them allows a faster model where latency is user-visible and a stronger one where output quality determines the result.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLM_BASE_URL` | OpenAI-compatible endpoint | *(required)* |
| `LLM_API_KEY` | API key for that endpoint | *(required)* |
| `MODEL` | Model under test | *(required)* |
| `BRAIN_MODEL` / `BRAIN_EFFORT` | Failure-mode analysis and test generation; latency-sensitive | `MODEL` |
| `AUTHOR_MODEL` / `AUTHOR_EFFORT` | System prompt authoring and patch generation; quality-sensitive | `MODEL` |
| `JUDGE_MODEL` | Judge model; determines verdict reliability | `AUTHOR_MODEL` |
| `JUDGE_VOTES` | Samples per verdict for majority vote. `1` disables self-consistency | `3` |
| `SKILL_MODEL` / `SKILL_EFFORT` | `SKILL.md` generation; structured rewriting | `MODEL` |
| `TARGET_EFFORT` | Reasoning effort for the model under test; ignored where unsupported | *(unset)* |

### Judge reliability

LLM-as-a-judge introduces variance that is rarely quantified. Measured on this project: re-sampling the same verdict five times over identical inputs flipped approximately one case in five.

Two mitigations are implemented, both surfaced in the interface rather than applied silently:

- **Rubric decomposition.** The judge cannot return a holistic verdict. It must enumerate atomic criteria and cite evidence for each. The per-criterion output shown in the screenshots is a direct consequence.
- **Self-consistency.** Three samples per verdict, resolved by majority. Splits are displayed as `2/3` rather than presented as settled.

`calibrate.py` samples stored runs, withholds the judge's verdict during labelling, and reports agreement:

```bash
uv run calibrate.py label -n 20
uv run calibrate.py report
```

The report uses Cohen's kappa rather than raw agreement. On a set where 90% of cases should pass, a judge that always returns `pass` achieves 90% agreement while carrying no discriminative information; its kappa is 0. False positives and false negatives are reported separately, since only false positives result in shipping a defective prompt.

> [!IMPORTANT]
> No kappa figure is published for this repository. The labelled set is not yet large enough for the estimate to be meaningful.

### Implementation notes

- **Test cases must not require tool calls.** The model under test is a plain chat model with no tool access. Pass conditions phrased as "must first call the order API" fail universally and render the verdict uninformative. External data is embedded directly in the test input instead.
- **Patch rules require short numbered imperatives.** With identical failures and an identical model, a single dense paragraph of conditions fixed 0 of 3 cases; the same content restructured as `1. Never … / 2. If … then …` fixed 3 of 3. Formatting was the only difference.

### Tests

```bash
uv run pytest      # 45 passed
```

Coverage concentrates on the statistical functions and the export package. An incorrect agreement statistic is more misleading than none, since it presents as rigorous.

### Roadmap

- Publish a kappa estimate once the labelled set is sufficient
- Item discrimination — cases passed or failed by every model carry no information but currently contribute to the score
- Score history across patch iterations (`runs/` already persists the data)
- MCP and tool-calling evaluation: scoring call trajectories rather than text

---

<a name="繁體中文"></a>
## 繁體中文

**ModelPlayGround** 是 LLM 功能的上線前評估工具。給定一句話描述預期行為，它會產生一組對抗性測試集、對你的模型實際執行、逐條回報未滿足的條件、提出 prompt 修訂，並重跑整組測試以偵測退化。產出的測試集與 system prompt 會匯出成不綁定供應商的套件。

LLM 功能通常在臨時性的人工測試後就上線，既沒有留下驗收標準，也無法在 prompt 或模型變更時偵測退化。ModelPlayGround 從任務描述推導出這些標準，並讓它們保持可重複執行。

> [!NOTE]
> 可搭配任何 OpenAI 相容端點——Azure OpenAI、OpenAI、Groq、OpenRouter、Ollama。只需三個環境變數。

### 功能

- **失效模式分析** — 依已知的 LLM 失效模式（幻覺、逾越授權承諾、未處理的資訊缺漏、格式偏移、對矛盾輸入無感、遺漏子請求、個資外洩）分類任務，並標示任務描述中觸發該分類的原句。
- **對抗性測試生成** — 產生五個由常規到對抗的測試案例。最後一案針對受壓下的指令遵循能力：同時夾帶多個請求、刻意省略必要欄位，並明確要求模型用猜的而非追問。
- **評分標準分解** — 檢核者須先將通過條件拆解成可獨立驗證的原子條件，逐條引用回答中的依據，再輸出 `pass = all(criteria)`。失敗時指出的是未滿足的具體條件，而非一個總分。
- **自我一致性檢核** — 每個判定重複取樣三次並以多數決決定。意見分歧時回報票數，而非壓縮成二元結果；平手時判定為失敗。
- **具退化偵測的修補** — 失敗案例會轉換成編號的 prompt 規則，套用前可編輯。套用修補後重跑整組測試而非僅失敗案例，因此「修好一題卻弄壞另一題」的規則會被揭露。
- **不綁供應商的匯出** — 測試案例、評分標準與 system prompt 會匯出成獨立套件，內含 promptfoo 設定。要改用其他供應商，只需修改 `providers` 欄位。

### 畫面

失效模式分析。每項分類都標示觸發它的原句，並列出會影響驗收標準、但尚未釐清的問題：

![失效模式分析](docs/risk.png)

測試案例與逐條判定。第四案右上角的 `2/3` 表示檢核者在多次取樣間出現分歧：

![測試案例與判定](docs/cases.png)

總結。風險由實際觀察到的失敗推導而來，而非固定的檢查清單：

![總結](docs/verdict.png)

### 匯出內容

| 檔案 | 用途 |
|---|---|
| `SKILL.md` | 含 YAML frontmatter 的 Agent skill 定義 |
| `system_prompt.md` | 修訂後的 system prompt |
| `evals/cases.jsonl` | 測試案例與通過條件，不綁定供應商 |
| `evals/rubric.md` | 人類可讀的驗收標準 |
| `evals/prompt.json` | promptfoo 設定引用的對話範本 |
| `promptfooconfig.yaml` | 可直接以 `npx promptfoo eval` 執行 |
| `run.py` | 最小可執行範例；`--selftest` 重播整組測試 |

`SKILL.md` 的規則段落由實際失敗的測試案例反推而來，因此記載的是該任務特有的限制：

> - 不得為任何待辦推定、編造或改寫截止日期，例如「明天上午」或「本週五」。
> - 不得依上下文推測而將某人視為負責人。在會議中發言或隸屬相關部門，都不構成指派。

### 架構

| 檔案 | 職責 |
|---|---|
| `app.py` | FastAPI 端點與 SSE 串流。狀態由前端持有並隨每次請求回傳，伺服器無狀態。 |
| `core/llm.py` | OpenAI 相容客戶端。記錄首字延遲；遇到 `400` 會移除不支援的參數重試，因此同一段程式碼可同時服務 Ollama 與 Azure OpenAI。 |
| `core/prompts.py` | 七段提示詞，集中管理以便調校。 |
| `core/pipeline.py` | `analyze` / `evaluate` / `suggest_patch` / `verdict`，拆分後修補重跑迴圈可單獨呼叫後半段。 |
| `core/judge.py` | 評分標準分解、多數決併票、偏誤量測。 |
| `core/calibration.py` | Cohen's kappa、bootstrap 信賴區間、偽陽性率。 |
| `core/store.py` | 執行紀錄持久化與校準抽樣。 |
| `core/pack.py` | 匯出套件建置。 |
| `calibrate.py` | 標註樣本與回報評分者間一致度的 CLI。 |
| `static/index.html` | 單頁前端，無建置步驟。 |

### 安裝與執行

需要 Python 3.12+ 與 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/breezy89757/ModelPlayGround.git
cd ModelPlayGround
uv sync
cp .env.example .env
```

`.env` 需填入三個值：

```bash
LLM_BASE_URL=https://api.openai.com/v1    # 或你的 Azure / Groq / OpenRouter / Ollama 端點
LLM_API_KEY=sk-...
MODEL=gpt-4o-mini                          # Azure 上填 deployment 名稱
```

```bash
uv run uvicorn app:app --port 8000
```

開啟 <http://localhost:8000>。完整一輪約 90 秒、10 次 API 呼叫。

### 環境變數

只有前三項為必填。其餘角色未設定時一律退回 `MODEL`，因此單一模型即可運作；分開設定可在使用者感知得到延遲之處用較快的模型，在輸出品質決定結果之處用較強的模型。

| 變數 | 說明 | 預設 |
| :--- | :--- | :--- |
| `LLM_BASE_URL` | OpenAI 相容端點 | *(必填)* |
| `LLM_API_KEY` | 該端點的 API 金鑰 | *(必填)* |
| `MODEL` | 受測模型 | *(必填)* |
| `BRAIN_MODEL` / `BRAIN_EFFORT` | 失效模式分析與測試生成；延遲敏感 | `MODEL` |
| `AUTHOR_MODEL` / `AUTHOR_EFFORT` | system prompt 撰寫與修補生成；品質敏感 | `MODEL` |
| `JUDGE_MODEL` | 檢核模型；決定判定的可信度 | `AUTHOR_MODEL` |
| `JUDGE_VOTES` | 每次判定的取樣次數。設為 `1` 即停用自我一致性 | `3` |
| `SKILL_MODEL` / `SKILL_EFFORT` | `SKILL.md` 生成；結構化改寫 | `MODEL` |
| `TARGET_EFFORT` | 受測模型的推理力度；端點不支援時自動忽略 | *(未設定)* |

### 檢核可信度

LLM-as-a-judge 會引入鮮少被量化的變異。本專案實測：對相同輸入重複取樣同一判定五次，約有五分之一的案例會翻盤。

已實作兩項對策，且都呈現在介面上而非私下套用：

- **評分標準分解。** 檢核者不得給出整體式判定，必須列舉原子條件並逐條引用依據。畫面上的逐條輸出即為此設計的直接結果。
- **自我一致性。** 每次判定取樣三次並以多數決決定。分歧時顯示為 `2/3`，而非呈現為已定論。

`calibrate.py` 會從已儲存的執行紀錄抽樣，在標註過程中隱藏檢核者的判定，並回報一致度：

```bash
uv run calibrate.py label -n 20
uv run calibrate.py report
```

報表採用 Cohen's kappa 而非原始一致率。在九成案例應通過的集合上，一個永遠回答 `pass` 的檢核者可達成 90% 一致率，卻不具任何鑑別力——其 kappa 恰為 0。偽陽性與偽陰性分開回報，因為只有偽陽性會導致有缺陷的 prompt 被上線。

> [!IMPORTANT]
> 本專案尚未公布 kappa 數值。已標註的樣本數尚不足以讓該估計具有意義。

### 實作筆記

- **測試案例不得要求呼叫工具。** 受測模型是純對話模型，沒有工具存取權。將通過條件寫成「必須先呼叫訂單 API」會導致全數失敗，判定也失去資訊量。外部資料改為直接內嵌於測試輸入中。
- **修補規則須為簡短的編號祈使句。** 在失敗案例與模型皆相同的條件下，單一段密集陳述條件的長文修好 0 / 3 案；同樣內容改寫為 `1. 不得…／2. 若…則…` 後修好 3 / 3。唯一的差異在於排版。

### 測試

```bash
uv run pytest      # 45 passed
```

測試集中在統計函式與匯出套件。錯誤的一致度統計比沒有統計更容易誤導，因為它看起來很嚴謹。

### 後續規劃

- 標註樣本累積足夠後公布 kappa 估計值
- 案例鑑別度——所有模型都通過或都失敗的案例不具資訊量，目前仍計入分數
- 跨修補版本的分數歷程（`runs/` 已保存所需資料）
- MCP 與 tool-calling 評估：對呼叫軌跡而非文字評分

---

## Contributing

Contributions are welcome. Please open an issue to discuss your proposal before submitting a pull request.

## License

[MIT](LICENSE)
