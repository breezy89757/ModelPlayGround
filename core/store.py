"""把每次 run 存成檔案。

沒有這層，「重跑後分數變好了」這句話是不可查證的 —— 你無法排除那只是隨機波動。
存下 prompt 的雜湊、模型、投票數與完整輸出，之後才能 diff、才能抽樣做校準。
"""
import json
import time
import hashlib
import pathlib
import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def save(state: dict) -> str:
    """回傳 run id。"""
    RUNS.mkdir(exist_ok=True)
    slug = (state.get("requirements", {}) or {}).get("task_name") or "run"
    rid = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{slug[:40]}"
    payload = {
        "id": rid,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "unix": int(time.time()),
        "prompt_hash": prompt_hash(state.get("system_prompt", "")),
        **state,
    }
    (RUNS / f"{rid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rid


def load(rid: str) -> dict:
    return json.loads((RUNS / f"{rid}.json").read_text(encoding="utf-8"))


def listing() -> list[dict]:
    if not RUNS.exists():
        return []
    out = []
    for p in sorted(RUNS.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        st = d.get("stats", {}) or {}
        out.append({
            "id": d.get("id", p.stem),
            "saved_at": d.get("saved_at"),
            "task": (d.get("requirements", {}) or {}).get("task_name"),
            "prompt_hash": d.get("prompt_hash"),
            "run": d.get("run"),
            "model": st.get("model"),
            "passed": st.get("passed"),
            "total": st.get("total"),
        })
    return out


def samples(limit: int | None = None) -> list[dict]:
    """把所有 run 攤平成可標註的單筆：(檢查標準, 回答, 評審判定)。"""
    out = []
    for meta in listing():
        try:
            d = load(meta["id"])
        except Exception:
            continue
        cases = {c["id"]: c for c in d.get("cases", [])}
        model = (d.get("stats") or {}).get("model", "?")
        for cid, cell in (d.get("results") or {}).items():
            j = cell.get("judge") or {}
            if not cases.get(cid) or cell.get("error"):
                continue
            out.append({
                "key": f"{meta['id']}::{cid}",
                "run_id": meta["id"], "model": model, "case_id": cid,
                "title": cases[cid].get("title"),
                "input": cases[cid].get("user_input"),
                "check": cases[cid].get("check"),
                "output": cell.get("text", ""),
                "judge_pass": bool(j.get("passed", j.get("pass"))),
                "confidence": j.get("confidence"),
                "contested": j.get("contested"),
                "reason": j.get("reason"),
            })
    return out[:limit] if limit else out
