"""校準指標的測試。

這些函式決定「我們憑什麼說評審可信」，所以邊界必須釘死 ——
一個算錯的 kappa 比沒有 kappa 更糟，因為它看起來很科學。
"""
import pytest

from core.calibration import cohens_kappa, report, interpret


def test_perfect_agreement():
    a = [True, False, True, False, True]
    assert cohens_kappa(a, a) == 1.0


def test_total_disagreement_is_negative():
    a = [True, False, True, False]
    b = [False, True, False, True]
    assert cohens_kappa(a, b) < 0


def test_chance_agreement_is_near_zero():
    # 兩邊各半、彼此獨立：一致率 50%，但 kappa 應該接近 0
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_high_agreement_rate_can_still_be_worthless():
    """這正是只看一致率會被騙的情形。

    人工說 9 過 1 不過；一個「永遠說通過」的評審一致率高達 90%，
    但它沒有任何辨別能力，kappa 必須反映這件事。
    """
    human = [True] * 9 + [False]
    always_pass = [True] * 10
    agreement = sum(x == y for x, y in zip(human, always_pass)) / 10
    assert agreement == 0.9

    # kappa 正好是 0：一致率再高，只要沒有超出隨機的辨別力，就等同於沒有資訊。
    # 這就是為什麼 README 引用的是 kappa 而不是一致率。
    assert cohens_kappa(always_pass, human) == 0.0


def test_single_class_returns_none():
    assert cohens_kappa([True] * 5, [True] * 5) is None


def test_length_mismatch_and_empty():
    assert cohens_kappa([], []) is None
    assert cohens_kappa([True], [True, False]) is None


def test_report_separates_false_positive_from_false_negative():
    recs = [
        {"judge_pass": True, "human_pass": False},   # 偽陽性：以為可以上線
        {"judge_pass": True, "human_pass": False},
        {"judge_pass": False, "human_pass": True},   # 偽陰性
        {"judge_pass": True, "human_pass": True},
        {"judge_pass": False, "human_pass": False},
    ]
    r = report(recs)
    assert r.n == 5
    assert r.false_positive == 2
    assert r.false_negative == 1
    assert r.fp_rate == pytest.approx(0.4)
    assert r.agreement == pytest.approx(0.4)


def test_report_ignores_unlabelled():
    recs = [
        {"judge_pass": True, "human_pass": None},
        {"judge_pass": True, "human_pass": True},
    ]
    assert report(recs).n == 1


def test_report_none_when_nothing_labelled():
    assert report([{"judge_pass": True, "human_pass": None}]) is None
    assert report([]) is None


def test_contested_vs_confident_accuracy():
    recs = [
        {"judge_pass": True, "human_pass": True, "contested": False},
        {"judge_pass": True, "human_pass": True, "contested": False},
        {"judge_pass": True, "human_pass": False, "contested": True},
        {"judge_pass": False, "human_pass": True, "contested": True},
    ]
    r = report(recs)
    assert r.confident_accuracy == pytest.approx(1.0)
    assert r.contested_accuracy == pytest.approx(0.0)


def test_interpret_covers_the_range():
    assert "隨機" in interpret(0.1)
    assert interpret(None) == "無法計算"
    assert "很高" in interpret(0.9)


def test_bootstrap_ci_needs_enough_samples():
    r = report([{"judge_pass": True, "human_pass": True},
                {"judge_pass": False, "human_pass": False}])
    assert r.kappa_ci is None          # n < 10 不給區間

    recs = ([{"judge_pass": True, "human_pass": True}] * 15 +
            [{"judge_pass": False, "human_pass": False}] * 15 +
            [{"judge_pass": True, "human_pass": False}] * 5)
    r = report(recs)
    assert r.kappa_ci is not None
    lo, hi = r.kappa_ci
    assert lo <= r.kappa <= hi
