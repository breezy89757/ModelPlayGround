"""評審層。

整個工具在賣「證據」，所以這一層是產品可信度的地基。實測過的問題：
單次 holistic 判定在重複呼叫下有約 20% 的機率翻盤（同一份回答、同一個標準）。

對策有二：
  1. 分解式評審 —— 先把檢查標準拆成原子條件，逐條檢核（見 prompts.JUDGE）
  2. n 次投票 —— 取多數決，並回報信心度；意見分歧的題目標記為 contested，
     不要假裝它是個確定的結論

回報 confidence 而不是二元判定，是刻意的。使用者看到「5 次中 4 次不通過」
比看到一個紅章誠實得多。
"""
import os
import asyncio
import statistics
from dataclasses import dataclass, field, asdict

from . import llm, prompts


def votes_n() -> int:
    return max(1, int(os.environ.get("JUDGE_VOTES", "3")))


def judge_model() -> str:
    # 一路退回 MODEL：只設了 MODEL 的人也要能跑
    return (os.environ.get("JUDGE_MODEL") or os.environ.get("AUTHOR_MODEL")
            or os.environ.get("MODEL") or "gpt-4o-mini")


@dataclass
class Verdict:
    passed: bool
    confidence: float          # 多數決的票數比例 0..1
    score: float               # 分數平均
    reason: str
    contested: bool            # 票數不一致 -> 這個結論不可盡信
    votes: list = field(default_factory=list)        # [bool]
    scores: list = field(default_factory=list)       # [int]
    reasons: list = field(default_factory=list)      # [str]
    criteria: list = field(default_factory=list)     # 多數方的原子條件檢核
    error: str | None = None

    def dict(self) -> dict:
        return asdict(self)


def _tally(results: list[dict], n: int) -> Verdict:
    """把 n 次判定併成一個結論。"""
    votes = [bool(r.get("pass")) for r in results]
    scores = [r.get("score") for r in results if isinstance(r.get("score"), (int, float))]
    reasons = [str(r.get("reason", "")) for r in results]

    n_pass = sum(votes)
    passed = n_pass * 2 > len(votes)          # 嚴格多數；平手時判不通過（保守）
    majority = n_pass if passed else len(votes) - n_pass
    confidence = majority / len(votes) if votes else 0.0

    # 理由與原子條件都取自多數方，不要拿少數方的說法去解釋多數方的結論
    side = [r for r, v in zip(results, votes) if v == passed]
    pick = side[0] if side else (results[0] if results else {})

    return Verdict(
        passed=passed,
        confidence=round(confidence, 3),
        score=round(statistics.fmean(scores), 2) if scores else 0.0,
        reason=str(pick.get("reason", "")),
        contested=len(set(votes)) > 1,
        votes=votes, scores=scores, reasons=reasons,
        criteria=pick.get("criteria", []) or [],
    )


async def judge(check: str, output: str, n: int | None = None,
                sem: asyncio.Semaphore | None = None) -> Verdict:
    """對一份回答重複判定 n 次，回傳含信心度的結論。"""
    n = n or votes_n()
    model = judge_model()
    body = f"【檢查標準】\n{check}\n\n【模型回答】\n{output}"

    async def once():
        if sem:
            async with sem:
                return await llm.complete_json(model, prompts.JUDGE, body, effort="low")
        return await llm.complete_json(model, prompts.JUDGE, body, effort="low")

    got = await asyncio.gather(*[once() for _ in range(n)], return_exceptions=True)
    ok = [r for r in got if isinstance(r, dict)]
    if not ok:
        first = next((r for r in got if isinstance(r, BaseException)), None)
        return Verdict(passed=False, confidence=0.0, score=0.0,
                       reason="評分失敗", contested=False,
                       error=f"{type(first).__name__}: {first}"[:200] if first else "unknown")

    v = _tally(ok, n)
    # 回傳的 criteria 可能與 pass 自相矛盾（模型偶爾會），以 criteria 為準並記下來
    if v.criteria and all("met" in c for c in v.criteria):
        derived = all(bool(c.get("met")) for c in v.criteria)
        if derived != v.passed:
            v.reason = (v.reason or "") + f"（原始判定與逐條檢核不一致，採逐條結果）"
            v.passed = derived
    return v


# ── 偏誤量測 ────────────────────────────────────────────────
# 不做「自動校正」，只量測並回報。看不見的校正比看得見的偏誤更危險。

def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman 等級相關。用來看評分是不是被回答長度帶著走。"""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def rank(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:                      # 平手取平均名次
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def bias_report(cells: list[dict]) -> dict:
    """cells: [{out_tokens, judge:{score, votes, contested}}]

    length_bias 明顯為正，代表評審偏好長答案 —— 那分數就不只在量正確性。
    """
    scored = [(c.get("out_tokens", 0), c["judge"]["score"])
              for c in cells if c.get("judge") and c["judge"].get("score") is not None]
    contested = [c for c in cells if c.get("judge", {}).get("contested")]
    return {
        "n": len(cells),
        "length_bias": _spearman([a for a, _ in scored], [b for _, b in scored]),
        "contested": len(contested),
        "self_consistency": round(1 - len(contested) / len(cells), 3) if cells else None,
        "judge_model": judge_model(),
        "votes": votes_n(),
    }
