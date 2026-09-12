"""評審校準。

「你怎麼知道你的評審是對的？」—— 這一層就是答案。

做法：從實際跑過的 run 抽樣，由人工標註通過／不通過，再跟評審的判定比對。
輸出 Cohen's kappa 與偽陽性率。

為什麼看 kappa 而不是只看一致率：如果 90% 的題目都通過，一個「永遠說通過」的
評審也能拿到 90% 一致率，但它毫無資訊量。kappa 會把「碰巧猜中」的部分扣掉。

偽陽性（評審說過、人工說沒過）是最危險的一種錯，因為它讓使用者以為可以上線。
它跟偽陰性要分開看，不能混在一個總數裡。
"""
import json
import random
import pathlib
import statistics
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "calibration" / "labels.jsonl"


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """兩個標註者在二元判定上的一致度，扣除隨機一致的部分。

    1.0 完全一致；0 等同隨機；負值代表比隨機還糟。
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n

    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)      # 隨機一致的期望值
    if pe >= 1.0:                            # 兩邊都是單一類別，kappa 無定義
        return None
    return round((po - pe) / (1 - pe), 3)


def _bootstrap_ci(a: list[bool], b: list[bool], iters: int = 2000,
                  seed: int = 0) -> tuple[float, float] | None:
    """kappa 的 95% 信賴區間。樣本少的時候，點估計會騙人。"""
    n = len(a)
    if n < 10:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohens_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            vals.append(k)
    if len(vals) < iters * 0.5:
        return None
    vals.sort()
    return (round(vals[int(len(vals) * 0.025)], 3),
            round(vals[int(len(vals) * 0.975)], 3))


def interpret(k: float | None) -> str:
    if k is None:
        return "無法計算"
    if k < 0.20:
        return "幾乎等同隨機 —— 這個評審不能用"
    if k < 0.40:
        return "一致度偏低 —— 判決不可盡信"
    if k < 0.60:
        return "中等 —— 可用於方向性判斷，不宜當驗收門檻"
    if k < 0.80:
        return "良好 —— 可以當驗收依據"
    return "很高 —— 評審與人工判斷高度吻合"


@dataclass
class Report:
    n: int
    agreement: float
    kappa: float | None
    kappa_ci: tuple | None
    false_positive: int        # 評審說過、人工說沒過 <- 最危險
    false_negative: int
    fp_rate: float
    fn_rate: float
    human_pass_rate: float
    judge_pass_rate: float
    mean_confidence: float | None
    contested_accuracy: float | None    # 評審自己猶豫時，它對的機率
    confident_accuracy: float | None    # 評審有把握時，它對的機率

    def dict(self) -> dict:
        return self.__dict__ | {"verdict": interpret(self.kappa)}

    def text(self) -> str:
        ci = f"  95% CI [{self.kappa_ci[0]}, {self.kappa_ci[1]}]" if self.kappa_ci else ""
        lines = [
            f"校準樣本 n = {self.n}",
            f"原始一致率     {self.agreement:.1%}",
            f"Cohen's kappa  {self.kappa}{ci}",
            f"               {interpret(self.kappa)}",
            "",
            f"偽陽性（說過但其實沒過）{self.false_positive:>3} 筆  {self.fp_rate:.1%}  <- 最危險",
            f"偽陰性（說沒過但其實過）{self.false_negative:>3} 筆  {self.fn_rate:.1%}",
            "",
            f"人工通過率 {self.human_pass_rate:.1%}   評審通過率 {self.judge_pass_rate:.1%}",
        ]
        if self.confident_accuracy is not None:
            lines += [
                "",
                f"評審有把握時的正確率  {self.confident_accuracy:.1%}",
                f"評審猶豫時的正確率    "
                + (f"{self.contested_accuracy:.1%}" if self.contested_accuracy is not None else "—"),
            ]
        return "\n".join(lines)


def report(records: list[dict]) -> Report | None:
    """records: [{judge_pass, human_pass, confidence?, contested?}]"""
    rs = [r for r in records if r.get("human_pass") is not None]
    if not rs:
        return None
    j = [bool(r["judge_pass"]) for r in rs]
    h = [bool(r["human_pass"]) for r in rs]
    n = len(rs)

    fp = sum(1 for x, y in zip(j, h) if x and not y)
    fn = sum(1 for x, y in zip(j, h) if not x and y)
    confs = [r["confidence"] for r in rs if isinstance(r.get("confidence"), (int, float))]

    def acc(subset):
        return (statistics.fmean([1.0 if r["judge_pass"] == r["human_pass"] else 0.0
                                  for r in subset])
                if subset else None)

    return Report(
        n=n,
        agreement=sum(1 for x, y in zip(j, h) if x == y) / n,
        kappa=cohens_kappa(j, h),
        kappa_ci=_bootstrap_ci(j, h),
        false_positive=fp, false_negative=fn,
        fp_rate=fp / n, fn_rate=fn / n,
        human_pass_rate=sum(h) / n, judge_pass_rate=sum(j) / n,
        mean_confidence=round(statistics.fmean(confs), 3) if confs else None,
        contested_accuracy=acc([r for r in rs if r.get("contested")]),
        confident_accuracy=acc([r for r in rs if not r.get("contested")]),
    )


# ── 標註集的讀寫 ─────────────────────────────────────────────

def load_labels() -> list[dict]:
    if not LABELS.exists():
        return []
    out = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def save_label(rec: dict) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
