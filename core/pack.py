"""把一次驗收的結果打包成可以直接拿去用的資料夾（zip）。

這是整個工具真正的交付物：測試集與 system prompt 不綁任何 provider，
拿去接 Claude / Gemini / 地端模型都成立。
"""
import io
import os
import json
import zipfile
import datetime

import yaml

from . import llm, prompts


async def skill_doc(state: dict) -> str:
    """用 LLM 寫 SKILL.md 的內容區塊（給另一個 agent 讀的說明）。"""
    req = state.get("requirements", {})
    cases = state.get("cases", [])
    stats = state.get("stats", {})
    body = json.dumps(
        {
            "requirements": req,
            "system_prompt": state.get("system_prompt", ""),
            "cases": [{"id": c["id"], "title": c["title"], "check": c["check"]} for c in cases],
            "failed": stats.get("failed", []),
        },
        ensure_ascii=False, indent=2,
    )
    # 刻意用便宜快的那顆：這是結構化改寫，不是難推理。
    # 實測小模型比大模型快 3 倍，而且「必須遵守」那節寫得更具體。
    model = (os.environ.get("SKILL_MODEL") or os.environ.get("BRAIN_MODEL")
             or os.environ.get("MODEL") or "gpt-4o-mini")
    r = await llm.complete(
        model,
        [{"role": "system", "content": prompts.SKILL_DOC},
         {"role": "user", "content": body}],
        effort=os.environ.get("SKILL_EFFORT") or None,
    )
    return r.text.strip()


def _slug(state: dict) -> str:
    s = (state.get("requirements", {}) or {}).get("task_name") or "ai-task"
    keep = "abcdefghijklmnopqrstuvwxyz0123456789-"
    s = "".join(ch if ch in keep else "-" for ch in s.lower()).strip("-")
    return s or "ai-task"


def build(state: dict, doc: str) -> tuple[str, bytes]:
    """回傳 (檔名, zip bytes)。"""
    slug = _slug(state)
    req = state.get("requirements", {}) or {}
    cases = state.get("cases", []) or []
    stats = state.get("stats", {}) or {}
    verdict = state.get("verdict", {}) or {}
    sysp = state.get("system_prompt", "")
    model = stats.get("model", "")
    today = datetime.date.today().isoformat()

    # ── SKILL.md ──
    desc = (req.get("task_summary") or slug).replace("\n", " ")[:180]
    skill = f"---\nname: {slug}\ndescription: {desc}\n---\n\n{doc}\n"

    # ── evals/cases.jsonl ──
    jsonl = "\n".join(
        json.dumps({
            "id": c.get("id"), "title": c.get("title"),
            "difficulty": c.get("difficulty"),
            "input": c.get("user_input"), "check": c.get("check"),
        }, ensure_ascii=False)
        for c in cases
    )

    # ── evals/rubric.md ──
    rubric = f"# 驗收標準\n\n共 {len(cases)} 題。每題只看 `check` 說的那件事，漏掉任一要求即為未通過。\n\n"
    for c in cases:
        rubric += (f"## {c.get('id')} · {c.get('title')} `{c.get('difficulty')}`\n\n"
                   f"**輸入**\n\n```\n{c.get('user_input','')}\n```\n\n"
                   f"**通過條件**：{c.get('check','')}\n\n")

    # ── promptfoo ──
    prompt_json = json.dumps(
        [{"role": "system", "content": sysp},
         {"role": "user", "content": "{{input}}"}],
        ensure_ascii=False, indent=2,
    )
    pf = {
        "description": f"{slug} 驗收測試（由 ModelPlayGround 產生 {today}）",
        "prompts": ["file://evals/prompt.json"],
        # 換 provider 只要改這裡，測試集本身不用動
        "providers": [f"openai:chat:{model}"],
        "tests": [
            {"vars": {"input": c.get("user_input", "")},
             "assert": [{"type": "llm-rubric", "value": c.get("check", "")}]}
            for c in cases
        ],
    }
    pf_yaml = yaml.safe_dump(pf, allow_unicode=True, sort_keys=False, width=100)

    # ── run.py ──
    runpy = f'''"""最小可跑範例。已填好模型與 system prompt。

任何 OpenAI 相容端點都能跑：

    pip install openai
    set LLM_BASE_URL=https://api.openai.com/v1      # 或你的 Azure / Ollama / Groq 端點
    set LLM_API_KEY=...
    python run.py "你的問題"
    python run.py --selftest                        # 跑全部驗收案例
"""
import os, sys, json, pathlib
from openai import OpenAI

HERE = pathlib.Path(__file__).parent
SYSTEM = (HERE / "system_prompt.md").read_text(encoding="utf-8")
MODEL = os.environ.get("MODEL") or {model!r}

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
    api_key=os.environ["LLM_API_KEY"],
)


def ask(text: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{{"role": "system", "content": SYSTEM}},
                  {{"role": "user", "content": text}}],
    )
    return r.choices[0].message.content


def selftest() -> None:
    """跑一次驗收案例，人工檢查輸出是否符合 check。"""
    for line in (HERE / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        print(f"\\n{{'='*70}}\\n[{{c['id']}}] {{c['title']}}\\n通過條件: {{c['check']}}\\n{{'-'*70}}")
        print(ask(c["input"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        print(ask(sys.argv[1] if len(sys.argv) > 1 else "你好"))
'''

    # ── README.md ──
    risks = "\n".join(f"- {r}" for r in verdict.get("risks", [])) or "- （無）"
    failed = "\n".join(f"- `{f['id']}` — {f['reason']}" for f in stats.get("failed", [])) or "- 全數通過"
    readme = f"""# {slug}

> {desc}

由 [ModelPlayGround](https://github.com/breezy89757) 於 {today} 產生。

## 驗收結果

| | |
|---|---|
| 受測模型 | `{model}` |
| 通過 | **{stats.get('passed', 0)} / {stats.get('total', 0)}** |
| 結論 | {verdict.get('headline', '')} |

{verdict.get('evidence', '')}

### 未通過的案例

{failed}

### 直接上線的風險

{risks}

## 這包裡面有什麼

| 檔案 | 用途 |
|---|---|
| `SKILL.md` | Agent skill 定義，可直接放進 agent 的 skills 目錄 |
| `system_prompt.md` | 經過驗收的 system prompt |
| `evals/cases.jsonl` | {len(cases)} 個驗收案例，**與 provider 無關** |
| `evals/rubric.md` | 人類可讀的驗收標準 |
| `promptfooconfig.yaml` | `npx promptfoo eval` 直接跑 |
| `run.py` | 最小可跑範例，`python run.py --selftest` 跑全部案例 |

## 換模型

`evals/cases.jsonl` 不綁任何 provider。要在 Claude / Gemini / 地端上驗收同一份標準，
只需改 `promptfooconfig.yaml` 的 `providers`：

```yaml
providers:
  - id: anthropic:messages:claude-sonnet-5
  - id: google:gemini-3.8-flash
  - id: ollama:chat:qwen3
```

## 之後怎麼用

出新模型、或改了 system prompt 之後，重跑一次：

```bash
npx promptfoo eval
```

通過數掉了就代表退步了。這份測試集是你的資產，別弄丟。
"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{slug}/SKILL.md", skill)
        z.writestr(f"{slug}/system_prompt.md", sysp + "\n")
        z.writestr(f"{slug}/README.md", readme)
        z.writestr(f"{slug}/evals/cases.jsonl", jsonl + "\n")
        z.writestr(f"{slug}/evals/rubric.md", rubric)
        z.writestr(f"{slug}/evals/prompt.json", prompt_json)
        z.writestr(f"{slug}/promptfooconfig.yaml", pf_yaml)
        z.writestr(f"{slug}/run.py", runpy)
    return f"{slug}.zip", buf.getvalue()
