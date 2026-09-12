"""主流程。

拆成可獨立呼叫的階段，因為「修補 → 重跑」需要單獨執行後半段：

    analyze()        一句話 -> 需求 / system prompt / 出包情境
    evaluate()       跑情境 + 評分
    suggest_patch()  看失敗案例 -> 產出可附加到 system prompt 的補充指令
    verdict()        總結

狀態不存在伺服器端 —— 由前端持有並回傳，這樣之後上雲不用管 session。
emit 是同步函式（put_nowait 進 queue），可以直接在 streaming callback 裡呼叫。
"""
import os
import json
import asyncio

from . import llm, prompts, store
from . import judge as judgelib

SEM = asyncio.Semaphore(6)  # 同時打端點的上限，避免被限流

# 受測模型沒有真的工具，講清楚，避免它「假裝查了」或被誤判
HARNESS_NOTE = (
    "\n\n---\n[測試環境說明] 你目前沒有任何可呼叫的工具或 API。"
    "外部查詢結果會由系統直接提供在使用者訊息中。"
    "若需要的資料沒有被提供，請明確說出你需要什麼，絕對不要編造。"
)


def target_model() -> str:
    """受測模型。"""
    return os.environ.get("MODEL") or os.environ.get("MODEL_CHEAP", "gpt-4o-mini")


def _role(name: str) -> tuple[str, str | None]:
    """(模型, effort)。沒設就退回受測模型，effort 空字串視為不帶這個參數。"""
    model = os.environ.get(f"{name}_MODEL") or target_model()
    effort = os.environ.get(f"{name}_EFFORT") or None
    return model, effort


# ── 階段 1：理解與設計 ─────────────────────────────────────────

async def analyze(user_input: str, emit) -> dict:
    brain, brain_effort = _role("BRAIN")
    author, author_effort = _role("AUTHOR")

    emit("stage", {"label": "正在理解你要做的事…"})
    req = await llm.complete_json(brain, prompts.EXTRACT, user_input, effort=brain_effort)
    emit("requirements", req)

    emit("stage", {"label": "正在設計出包情境…"})
    req_text = json.dumps(req, ensure_ascii=False, indent=2)

    # 兩件事同時做，但情境先回來就先畫，不要讓使用者盯著空白等 system prompt
    sys_task = asyncio.create_task(
        llm.complete(
            author,
            [{"role": "system", "content": prompts.AUTHOR_SYSTEM_PROMPT},
             {"role": "user", "content": req_text}],
            effort=author_effort,
        )
    )
    cases_json = await llm.complete_json(brain, prompts.GEN_CASES, req_text, effort=brain_effort)
    cases = cases_json.get("cases", [])[:5]
    emit("cases", {"cases": cases})

    emit("stage", {"label": "正在寫 system prompt…"})
    system_prompt = (await sys_task).text.strip()
    emit("system_prompt", {"text": system_prompt})

    return {"requirements": req, "cases": cases, "system_prompt": system_prompt}


# ── 階段 2：實測與評分 ─────────────────────────────────────────

async def evaluate(system_prompt: str, cases: list, emit, run: int = 1) -> dict:
    """跑每個情境並評分。回傳 {case_id: {...}}。"""
    model = target_model()
    effort = os.environ.get("TARGET_EFFORT") or None
    run_system = system_prompt + HARNESS_NOTE
    out: dict[str, dict] = {}

    emit("stage", {"label": f"實測中（第 {run} 輪）…"})

    async def one(case: dict):
        cid = case["id"]
        async with SEM:
            emit("cell_start", {"case": cid, "run": run, "model": model})
            r = await llm.complete(
                model,
                [{"role": "system", "content": run_system},
                 {"role": "user", "content": case["user_input"]}],
                effort=effort,
                max_tokens=900,
                on_delta=lambda p: emit("delta", {"case": cid, "run": run, "text": p}),
            )
        out[cid] = {
            "text": r.text, "ttft": round(r.ttft, 2), "total": round(r.total, 2),
            "in_tokens": r.in_tokens, "out_tokens": r.out_tokens, "error": r.error,
        }
        emit("cell_done", {"case": cid, "run": run,
                           **{k: v for k, v in out[cid].items() if k != "text"}})

    await asyncio.gather(*[one(c) for c in cases])

    emit("stage", {"label": "檢核中…"})

    async def do_judge(case: dict):
        cell = out[case["id"]]
        if cell["error"]:
            v = judgelib.Verdict(passed=False, confidence=0.0, score=0.0,
                                 reason="呼叫失敗", contested=False)
        else:
            v = await judgelib.judge(case["check"], cell["text"], sem=SEM)
        cell["judge"] = v.dict()
        emit("judged", {"case": case["id"], "run": run, **v.dict()})

    await asyncio.gather(*[do_judge(c) for c in cases])
    return out


# ── 階段 3：修補 ──────────────────────────────────────────────

async def suggest_patch(system_prompt: str, failures: list) -> dict:
    """failures: [{case_id, user_input, check, output, reason}]"""
    author, author_effort = _role("AUTHOR")
    body = "【目前的 system prompt】\n" + system_prompt + "\n\n【失敗案例】\n"
    for f in failures:
        body += (
            f"\n--- {f['case_id']} ---\n"
            f"輸入：{f['user_input'][:400]}\n"
            f"檢查標準：{f['check']}\n"
            f"模型實際回答：{(f.get('output') or '')[:600]}\n"
            f"檢核判定未通過的理由：{f['reason']}\n"
        )
    return await llm.complete_json(author, prompts.PATCH, body, effort=author_effort)


# ── 階段 4：總結 ──────────────────────────────────────────────

async def verdict(cases: list, results: dict, emit) -> dict:
    author, author_effort = _role("AUTHOR")
    passed = sum(1 for c in results.values() if c.get("judge", {}).get("passed"))
    stats = {
        "model": target_model(),
        "passed": passed,
        "total": len(results),
        "failed": [
            {"id": cid, "reason": c.get("judge", {}).get("reason", ""),
             "confidence": c.get("judge", {}).get("confidence")}
            for cid, c in results.items() if not c.get("judge", {}).get("passed")
        ],
        # 票數不一致的情境：這些結論不該被當成定論
        "contested": [cid for cid, c in results.items()
                      if c.get("judge", {}).get("contested")],
    }
    emit("stage", {"label": "整理結論…"})
    try:
        v = await llm.complete_json(
            author, prompts.VERDICT,
            json.dumps({"cases": cases, "stats": stats}, ensure_ascii=False, default=str),
            effort=author_effort,
        )
    except Exception as e:
        v = {"status": "needs_work", "headline": f"結論生成失敗: {e}"[:60],
             "evidence": "", "risks": []}
    payload = {"stats": stats, "verdict": v}
    emit("verdict", payload)
    return payload


def persist(state: dict, emit=None) -> str:
    """存檔。分數曲線與之後的抽樣都要靠它。"""
    rid = store.save(state)
    if emit:
        emit("saved", {"run_id": rid,
                       "prompt_hash": store.prompt_hash(state.get("system_prompt", ""))})
    return rid


# ── 完整的第一輪 ──────────────────────────────────────────────

async def run(user_input: str, emit) -> dict:
    a = await analyze(user_input, emit)
    emit("model", {"model": target_model()})
    results = await evaluate(a["system_prompt"], a["cases"], emit, run=1)
    v = await verdict(a["cases"], results, emit)
    state = {**a, "results": results, **v, "run": 1}
    persist(state, emit)
    emit("done", {})
    return state
