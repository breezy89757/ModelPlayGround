"""投票併票與偏誤量測的測試（不打 LLM）。"""
import pytest

from core.judge import _tally, _spearman, bias_report, Verdict


def v(passed, score, reason="r", criteria=None):
    d = {"pass": passed, "score": score, "reason": reason}
    if criteria is not None:
        d["criteria"] = criteria
    return d


def test_unanimous_pass():
    r = _tally([v(True, 5), v(True, 5), v(True, 4)], 3)
    assert r.passed is True
    assert r.confidence == 1.0
    assert r.contested is False
    assert r.score == pytest.approx(4.67, abs=0.01)


def test_majority_wins_and_flags_contested():
    r = _tally([v(True, 5), v(False, 2), v(False, 3)], 3)
    assert r.passed is False
    assert r.confidence == pytest.approx(2 / 3, abs=0.01)
    assert r.contested is True


def test_tie_fails_closed():
    """平手時判不通過。驗收工具寧可誤殺，不可放行。"""
    r = _tally([v(True, 5), v(False, 1)], 2)
    assert r.passed is False
    assert r.confidence == 0.5


def test_reason_comes_from_the_majority_side():
    r = _tally([v(False, 1, "漏了負責人"), v(False, 2, "沒寫期限"), v(True, 5, "很好")], 3)
    assert r.passed is False
    assert r.reason in ("漏了負責人", "沒寫期限")
    assert r.reason != "很好"


def test_criteria_override_contradictory_pass():
    """模型偶爾會逐條說 met=false 卻給 pass=true。以逐條為準。"""
    c = [{"id": 1, "text": "x", "met": True}, {"id": 2, "text": "y", "met": False}]
    r = _tally([v(True, 5, "看起來不錯", c)] * 3, 3)
    assert r.passed is True          # _tally 只做併票
    # 覆寫邏輯在 judge() 裡，這裡驗證資料有被保留下來供覆寫
    assert r.criteria == c
    assert not all(x["met"] for x in r.criteria)


def test_single_vote_still_works():
    r = _tally([v(True, 4)], 1)
    assert r.passed is True and r.confidence == 1.0 and r.contested is False


def test_missing_scores_do_not_crash():
    r = _tally([{"pass": True, "reason": "x"}, {"pass": True, "reason": "y"}], 2)
    assert r.passed is True and r.score == 0.0


def test_spearman_detects_monotonic_relation():
    assert _spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0
    assert _spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == -1.0


def test_spearman_handles_ties_and_short_input():
    assert _spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None   # 零變異
    assert _spearman([1, 2], [1, 2]) is None               # n < 3


def test_bias_report_flags_length_preference():
    """長答案分數就高 -> 評分不只在量正確性。"""
    cells = [{"out_tokens": t,
              "judge": {"score": s, "contested": False}}
             for t, s in [(50, 1), (100, 2), (200, 3), (400, 4), (800, 5)]]
    rep = bias_report(cells)
    assert rep["length_bias"] == 1.0
    assert rep["self_consistency"] == 1.0
    assert rep["contested"] == 0


def test_bias_report_counts_contested():
    cells = [{"out_tokens": 100, "judge": {"score": 3, "contested": True}},
             {"out_tokens": 100, "judge": {"score": 3, "contested": False}}]
    rep = bias_report(cells)
    assert rep["contested"] == 1
    assert rep["self_consistency"] == 0.5


def test_verdict_is_json_serialisable():
    import json
    d = Verdict(True, 1.0, 5.0, "ok", False).dict()
    assert json.loads(json.dumps(d))["passed"] is True
