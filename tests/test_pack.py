"""下載包的測試。這是交付物，壞掉使用者會直接拿到壞檔案。"""
import io
import json
import zipfile

import pytest
import yaml

from core.pack import build, _slug
from core.llm import _parse_json


STATE = {
    "requirements": {
        "task_name": "meeting-summary",
        "task_summary": "把逐字稿整理成摘要與待辦。",
    },
    "cases": [
        {"id": "c1", "title": "例行摘要", "difficulty": "easy",
         "user_input": "整理這份逐字稿", "check": "必須標示會議重點區塊。"},
        {"id": "c2", "title": "缺漏處理", "difficulty": "edge",
         "user_input": "沒有期限的待辦", "check": "不得自行推定截止日期。"},
    ],
    "system_prompt": "你是會議記錄助理。",
    "stats": {"model": "gpt-4o-mini", "passed": 1, "total": 2,
              "failed": [{"id": "c2", "reason": "擅自補期限"}]},
    "verdict": {"status": "risky", "headline": "草稿可用", "evidence": "失敗集中在臆測。",
                "risks": ["可能編造截止日"]},
}


@pytest.fixture
def z():
    _, blob = build(STATE, "# 會議摘要\n\n說明。")
    return zipfile.ZipFile(io.BytesIO(blob))


def test_filename_follows_task_name():
    name, _ = build(STATE, "x")
    assert name == "meeting-summary.zip"


def test_contains_every_promised_file(z):
    got = set(z.namelist())
    for f in ["SKILL.md", "system_prompt.md", "README.md", "evals/cases.jsonl",
              "evals/rubric.md", "evals/prompt.json", "promptfooconfig.yaml", "run.py"]:
        assert f"meeting-summary/{f}" in got


def test_skill_frontmatter_is_parseable(z):
    txt = z.read("meeting-summary/SKILL.md").decode("utf-8")
    assert txt.startswith("---\n")
    fm = yaml.safe_load(txt.split("---")[1])
    assert fm["name"] == "meeting-summary"
    assert fm["description"]


def test_description_is_single_line(z):
    """description 跨行會讓 frontmatter 解析出錯。"""
    fm_raw = z.read("meeting-summary/SKILL.md").decode("utf-8").split("---")[1]
    assert len([l for l in fm_raw.strip().splitlines() if l.strip()]) == 2


def test_cases_jsonl_one_object_per_line(z):
    lines = z.read("meeting-summary/evals/cases.jsonl").decode("utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert {"id", "title", "difficulty", "input", "check"} <= obj.keys()


def test_promptfoo_config_is_valid_yaml_with_one_test_per_case(z):
    cfg = yaml.safe_load(z.read("meeting-summary/promptfooconfig.yaml"))
    assert len(cfg["tests"]) == 2
    assert cfg["prompts"] == ["file://evals/prompt.json"]
    assert cfg["tests"][0]["assert"][0]["type"] == "llm-rubric"
    assert cfg["tests"][1]["assert"][0]["value"] == "不得自行推定截止日期。"


def test_prompt_json_carries_system_and_placeholder(z):
    msgs = json.loads(z.read("meeting-summary/evals/prompt.json"))
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "你是會議記錄助理。"
    assert "{{input}}" in msgs[1]["content"]


def test_run_py_is_valid_python(z):
    import ast
    ast.parse(z.read("meeting-summary/run.py").decode("utf-8"))


def test_readme_reports_the_failure(z):
    txt = z.read("meeting-summary/README.md").decode("utf-8")
    assert "1 / 2" in txt
    assert "擅自補期限" in txt
    assert "可能編造截止日" in txt


def test_empty_state_does_not_crash():
    name, blob = build({}, "")
    assert name == "ai-task.zip"
    assert zipfile.ZipFile(io.BytesIO(blob)).namelist()


@pytest.mark.parametrize("raw", ["Meeting Summary", "客服/機器人", "A_B.C", "  spaced  "])
def test_slug_is_always_filesystem_safe(raw):
    got = _slug({"requirements": {"task_name": raw}})
    assert got == got.lower()
    assert all(c.isalnum() or c == "-" for c in got)
    assert not got.startswith("-") and not got.endswith("-")


def test_slug_falls_back_when_unusable():
    assert _slug({}) == "ai-task"
    assert _slug({"requirements": {"task_name": "///"}}) == "ai-task"


# ── llm._parse_json 的邊界，模型常常不照規矩回 ──

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    '這是你要的結果：\n{"a": 1}\n希望有幫助',
    '  \n{"a": 1}\n  ',
])
def test_parse_json_tolerates_model_sloppiness(raw):
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_raises_on_garbage():
    with pytest.raises(Exception):
        _parse_json("完全不是 JSON")
