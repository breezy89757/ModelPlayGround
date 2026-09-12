"""評審校準工具。

    python calibrate.py label [-n 20]   從存下的 run 抽樣，人工標註
    python calibrate.py report          算 Cohen's kappa 與偽陽性率
    python calibrate.py runs            列出所有 run

標註時**先不顯示評審的判定**，看完回答才揭曉 —— 否則你會被它帶著走，
標出來的東西沒有校準價值。
"""
import sys
import random
import argparse

# Windows 主控台預設 cp950，中文會變亂碼
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from dotenv import load_dotenv

load_dotenv()

from core import store, calibration

W = 78


def _wrap(text: str, indent: str = "  ", limit: int = 1800) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + f"\n{indent}…（截斷，原長 {len(text)} 字）"
    return "\n".join(indent + ln for ln in text.splitlines())


def cmd_label(args):
    done = {r["key"] for r in calibration.load_labels()}
    pool = [s for s in store.samples() if s["key"] not in done]
    if not pool:
        print("沒有可標註的樣本了。先跑幾次驗收，或 report 看現有結果。")
        return

    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.n]
    print(f"待標註 {len(pool)} 筆（已標 {len(done)} 筆）")
    print("每筆請回答：y = 通過, n = 不通過, s = 跳過, q = 結束\n")

    for i, s in enumerate(pool, 1):
        print("=" * W)
        print(f"[{i}/{len(pool)}] {s['title']}   ({s['model']} · {s['case_id']})")
        print("-" * W)
        print("【輸入】")
        print(_wrap(s["input"], limit=900))
        print("\n【通過條件】")
        print(_wrap(s["check"]))
        print("\n【模型回答】")
        print(_wrap(s["output"]))
        print("-" * W)

        while True:
            a = input("你的判定 [y/n/s/q] > ").strip().lower()
            if a in ("y", "n", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue

        human = a == "y"
        calibration.save_label({
            "key": s["key"], "run_id": s["run_id"], "model": s["model"],
            "case_id": s["case_id"], "human_pass": human,
            "judge_pass": s["judge_pass"], "confidence": s.get("confidence"),
            "contested": s.get("contested"),
        })
        mark = "一致" if human == s["judge_pass"] else "★ 不一致"
        print(f"  -> 評審判定 {'通過' if s['judge_pass'] else '不通過'}"
              f"（信心 {s.get('confidence')}）  {mark}\n")

    cmd_report(args)


def cmd_report(args):
    labels = calibration.load_labels()
    rep = calibration.report(labels)
    print()
    print("=" * W)
    if not rep:
        print("還沒有標註資料。先跑 `python calibrate.py label`。")
        return
    print(rep.text())
    print("=" * W)
    if rep.n < 30:
        print(f"\n⚠️ 樣本只有 {rep.n} 筆，kappa 的信賴區間會很寬。至少標到 80 筆再引用這個數字。")


def cmd_runs(args):
    rows = store.listing()
    if not rows:
        print("還沒有存下任何 run。")
        return
    print(f"{'run id':<34} {'模型':<16} {'分數':<7} {'prompt':<14} 時間")
    for r in rows[: args.n]:
        sc = f"{r['passed']}/{r['total']}" if r.get("passed") is not None else "—"
        print(f"{r['id']:<34} {(r.get('model') or '—'):<16} {sc:<7} "
              f"{(r.get('prompt_hash') or '—'):<14} {r.get('saved_at') or ''}")
    print(f"\n共 {len(rows)} 筆，可標註樣本 {len(store.samples())} 筆")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("label", help="人工標註樣本")
    a.add_argument("-n", type=int, default=20)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_label)

    b = sub.add_parser("report", help="算 kappa")
    b.set_defaults(func=cmd_report)

    c = sub.add_parser("runs", help="列出 run")
    c.add_argument("-n", type=int, default=20)
    c.set_defaults(func=cmd_runs)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
