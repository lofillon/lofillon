from __future__ import annotations

from impactlens.evaluation.goldset import get_path


def test_get_path_dot_and_index() -> None:
    obj = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert get_path(obj, "a.b[0].c") == 1
    assert get_path(obj, "a.b[1].c") == 2
    assert get_path(obj, "a.b[2].c") is None

