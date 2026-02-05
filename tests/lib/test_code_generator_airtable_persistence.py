"""Unit tests for helpers inside lib.code_generator_airtable_persistence."""

from lib import code_generator_airtable_persistence as persistence


def test_sanitize_repo_name_lowercases_and_strips_invalid_chars() -> None:
    assert persistence._sanitize_repo_name("My Awesome Repo!!!") == "my-awesome-repo"


def test_sanitize_repo_name_returns_none_for_empty_input() -> None:
    assert persistence._sanitize_repo_name("   !!!   ") is None


def test_normalize_language_prefers_explicit_value() -> None:
    record = {"files.path": "apps/main.py"}
    assert persistence._normalize_language("Python", record) == "Python"


def test_normalize_language_falls_back_to_extension() -> None:
    record = {"files.path": "services/data_collector.gs"}
    assert persistence._normalize_language(None, record) == "Apps Script"


def test_compute_lines_of_code_ignores_blank_lines() -> None:
    source = """
    def foo():
        pass


    def bar():
            return 42
    """
    assert persistence._compute_lines_of_code(source) == 4
