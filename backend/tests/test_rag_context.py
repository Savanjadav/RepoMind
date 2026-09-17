from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from app.rag_context import ContextEvidence, FormattedContext, format_context


def _evidence(
    *,
    repository_name: str = "atlas-service",
    path: str = "src/auth/service.py",
    symbol_name: str | None = "authenticate_user",
    start_line: int = 10,
    end_line: int = 12,
    content: str = "def authenticate_user():\n    return True",
) -> ContextEvidence:
    return ContextEvidence(
        repository_name=repository_name,
        path=path,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
    )


def _assert_bounded(result: FormattedContext) -> None:
    assert result.used_characters == len(result.text)
    assert result.used_characters <= result.max_characters


def _truncated_block(evidence: ContextEvidence, *, shown: int, index: int = 1) -> str:
    line_metadata = (
        f"Line: {evidence.start_line}"
        if evidence.start_line == evidence.end_line
        else f"Lines: {evidence.start_line}-{evidence.end_line}"
    )
    symbol = evidence.symbol_name if evidence.symbol_name is not None else "<none>"
    return "\n".join(
        (
            f"[Evidence {index}]",
            f"Repository: {evidence.repository_name}",
            f"Path: {evidence.path}",
            f"Symbol: {symbol}",
            line_metadata,
            f"Content-Characters: {shown}/{len(evidence.content)}",
            "Content:",
            evidence.content[:shown],
            "[TRUNCATED]",
            f"[End Evidence {index}]",
        )
    )


def test_context_evidence_is_frozen_and_slotted() -> None:
    evidence = _evidence()
    assert not hasattr(evidence, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(evidence, "path", "changed.py")


def test_formatted_context_is_frozen_and_slotted() -> None:
    result = format_context([], max_characters=1)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(result, "text", "changed")


def test_valid_evidence_preserves_values_exactly() -> None:
    evidence = _evidence(
        repository_name="  Atlas  ",
        path=" src/auth.py ",
        symbol_name=" auth ",
        content="\tCafé ✓\n",
    )
    assert evidence.repository_name == "  Atlas  "
    assert evidence.path == " src/auth.py "
    assert evidence.symbol_name == " auth "
    assert evidence.content == "\tCafé ✓\n"


@pytest.mark.parametrize("value", [None, 1, True, [], object()])
def test_repository_name_requires_actual_string(value: object) -> None:
    with pytest.raises(ValueError):
        _evidence(repository_name=cast(str, value))


@pytest.mark.parametrize("value", ["", " ", "\u2003"])
def test_repository_name_requires_non_whitespace(value: str) -> None:
    with pytest.raises(ValueError):
        _evidence(repository_name=value)


@pytest.mark.parametrize("control", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"])
def test_repository_name_rejects_ascii_controls(control: str) -> None:
    with pytest.raises(ValueError):
        _evidence(repository_name=f"atlas{control}service")


@pytest.mark.parametrize("value", [None, 1, True, [], object()])
def test_path_requires_actual_string(value: object) -> None:
    with pytest.raises(ValueError):
        _evidence(path=cast(str, value))


@pytest.mark.parametrize("value", ["", " ", "\u2003"])
def test_path_requires_non_whitespace(value: str) -> None:
    with pytest.raises(ValueError):
        _evidence(path=value)


@pytest.mark.parametrize("control", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"])
def test_path_rejects_ascii_controls(control: str) -> None:
    with pytest.raises(ValueError):
        _evidence(path=f"src/auth{control}/service.py")


def test_symbol_name_accepts_none_and_preserves_valid_string() -> None:
    assert _evidence(symbol_name=None).symbol_name is None
    assert _evidence(symbol_name="  authenticate  ").symbol_name == "  authenticate  "


@pytest.mark.parametrize("value", [1, True, [], object()])
def test_symbol_name_requires_actual_string_or_none(value: object) -> None:
    with pytest.raises(ValueError):
        _evidence(symbol_name=cast(str, value))


@pytest.mark.parametrize("value", ["", " ", "\u2003"])
def test_symbol_name_requires_non_whitespace(value: str) -> None:
    with pytest.raises(ValueError):
        _evidence(symbol_name=value)


@pytest.mark.parametrize("control", ["\x00", "\t", "\n", "\r", "\x1f", "\x7f"])
def test_symbol_name_rejects_ascii_controls(control: str) -> None:
    with pytest.raises(ValueError):
        _evidence(symbol_name=f"authenticate{control}user")


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_start_line_requires_actual_integer(value: object) -> None:
    with pytest.raises(ValueError):
        _evidence(start_line=cast(int, value))


@pytest.mark.parametrize("value", [True, False, 12.0, "12", None])
def test_end_line_requires_actual_integer(value: object) -> None:
    with pytest.raises(ValueError):
        _evidence(end_line=cast(int, value))


@pytest.mark.parametrize("start_line", [0, -1])
def test_start_line_must_be_positive(start_line: int) -> None:
    with pytest.raises(ValueError):
        _evidence(start_line=start_line)


def test_end_line_must_not_precede_start_line() -> None:
    with pytest.raises(ValueError):
        _evidence(start_line=10, end_line=9)


@pytest.mark.parametrize("content", [None, 1, True, [], object()])
def test_content_requires_actual_string(content: object) -> None:
    with pytest.raises(ValueError):
        _evidence(content=cast(str, content))


def test_empty_content_is_valid() -> None:
    assert _evidence(content="").content == ""


def test_one_evidence_block_has_exact_human_inspectable_format() -> None:
    evidence = _evidence(content="return True", start_line=10, end_line=10)
    result = format_context([evidence], max_characters=1_000)
    assert result.text == "\n".join(
        (
            "[Evidence 1]",
            "Repository: atlas-service",
            "Path: src/auth/service.py",
            "Symbol: authenticate_user",
            "Line: 10",
            "Content-Characters: 11/11",
            "Content:",
            "return True",
            "[End Evidence 1]",
        )
    )
    assert result.included_evidence_count == 1
    assert result.truncated is False
    _assert_bounded(result)


def test_symbol_none_and_line_range_are_explicit() -> None:
    result = format_context(
        [_evidence(symbol_name=None, start_line=3, end_line=8)],
        max_characters=1_000,
    )
    assert "Symbol: <none>" in result.text
    assert "Lines: 3-8" in result.text
    assert "Line: 3\n" not in result.text


@pytest.mark.parametrize(
    "content",
    [
        "\tfirst\nsecond\n",
        "Café ✓ 東京",
        "[Evidence 2]\n[TRUNCATED]\n[End Evidence 1]",
        "",
        "control-like source: \x00\x1f\x7f",
    ],
)
def test_source_content_is_preserved_exactly(content: str) -> None:
    evidence = _evidence(content=content)
    result = format_context([evidence], max_characters=10_000)
    prefix = result.text.index("Content:\n") + len("Content:\n")
    suffix = result.text.rindex("\n[End Evidence 1]")
    assert result.text[prefix:suffix] == content
    assert f"Content-Characters: {len(content)}/{len(content)}" in result.text
    _assert_bounded(result)


def test_multiple_blocks_preserve_order_exact_separator_and_duplicates() -> None:
    first = _evidence(path="first.py", content="first")
    duplicate = _evidence(path="first.py", content="first")
    third = _evidence(path="third.py", content="third")
    result = format_context([first, duplicate, third], max_characters=10_000)
    assert result.text.index("[Evidence 1]") < result.text.index("[Evidence 2]")
    assert result.text.index("[Evidence 2]") < result.text.index("[Evidence 3]")
    assert result.text.count("Path: first.py") == 2
    assert "[End Evidence 1]\n\n[Evidence 2]" in result.text
    assert "[End Evidence 2]\n\n[Evidence 3]" in result.text
    assert result.included_evidence_count == 3
    assert result.truncated is False
    _assert_bounded(result)


def test_empty_evidence_returns_empty_complete_context() -> None:
    result = format_context([], max_characters=7)
    assert result == FormattedContext("", 0, False, 0, 7)
    _assert_bounded(result)


@pytest.mark.parametrize("budget", [True, False, 1.0, "1", None])
def test_max_characters_requires_actual_integer(budget: object) -> None:
    with pytest.raises(ValueError):
        format_context([], max_characters=cast(int, budget))


@pytest.mark.parametrize("budget", [0, -1])
def test_max_characters_must_be_positive(budget: int) -> None:
    with pytest.raises(ValueError):
        format_context([], max_characters=budget)


def test_exact_fit_includes_complete_block() -> None:
    evidence = _evidence(content="x" * 200)
    complete = format_context([evidence], max_characters=10_000)
    exact = format_context([evidence], max_characters=len(complete.text))
    assert exact.text == complete.text
    assert exact.truncated is False
    _assert_bounded(exact)


def test_one_character_below_complete_block_truncates_within_budget() -> None:
    evidence = _evidence(content="x" * 200)
    complete = format_context([evidence], max_characters=10_000)
    result = format_context([evidence], max_characters=len(complete.text) - 1)
    assert result.included_evidence_count == 1
    assert result.truncated is True
    assert "[TRUNCATED]" in result.text
    assert result.text != complete.text
    _assert_bounded(result)


def test_zero_character_truncated_block_and_marker_count_toward_budget() -> None:
    evidence = _evidence(content="x" * 200)
    zero_block = _truncated_block(evidence, shown=0)
    exact = format_context([evidence], max_characters=len(zero_block))
    too_small = format_context([evidence], max_characters=len(zero_block) - 1)
    assert exact.text == zero_block
    assert exact.included_evidence_count == 1
    assert exact.truncated is True
    assert "Content-Characters: 0/200" in exact.text
    assert too_small.text == ""
    assert too_small.included_evidence_count == 0
    assert too_small.truncated is True
    _assert_bounded(exact)
    _assert_bounded(too_small)


def test_metadata_overhead_can_exhaust_budget_without_partial_output() -> None:
    evidence = _evidence(repository_name="r" * 200, content="x")
    result = format_context([evidence], max_characters=20)
    assert result == FormattedContext("", 0, True, 0, 20)
    _assert_bounded(result)


def test_separator_counts_toward_budget() -> None:
    first = _evidence(path="first.py", content="a")
    second = _evidence(path="second.py", content="b" * 200)
    first_text = format_context([first], max_characters=10_000).text
    second_zero = _truncated_block(second, shown=0, index=2)
    exact_budget = len(first_text) + len("\n\n") + len(second_zero)
    exact = format_context([first, second], max_characters=exact_budget)
    under = format_context([first, second], max_characters=exact_budget - 1)
    assert exact.included_evidence_count == 2
    assert "Content-Characters: 0/200" in exact.text
    assert under.included_evidence_count == 1
    assert under.text == first_text
    assert under.truncated is True
    _assert_bounded(exact)
    _assert_bounded(under)


def test_full_higher_ranked_blocks_precede_final_truncated_block() -> None:
    first = _evidence(path="first.py", content="first")
    second = _evidence(path="second.py", content="x" * 200)
    first_text = format_context([first], max_characters=10_000).text
    second_zero = _truncated_block(second, shown=0, index=2)
    budget = len(first_text) + 2 + len(second_zero) + 25
    result = format_context([first, second], max_characters=budget)
    assert result.text.startswith(first_text + "\n\n[Evidence 2]")
    assert result.included_evidence_count == 2
    assert result.truncated is True
    assert "[TRUNCATED]" in result.text
    _assert_bounded(result)


def test_oversized_first_item_is_not_skipped_for_smaller_later_item() -> None:
    first = _evidence(repository_name="r" * 500, content="x")
    second = _evidence(repository_name="r", path="p", symbol_name=None, content="y")
    second_complete = format_context([second], max_characters=10_000)
    result = format_context([first, second], max_characters=len(second_complete.text))
    assert result.text == ""
    assert result.included_evidence_count == 0
    assert result.truncated is True
    _assert_bounded(result)


def test_omitted_later_evidence_sets_truncated() -> None:
    first = _evidence(path="first.py", content="first")
    second = _evidence(path="second.py", content="second")
    first_complete = format_context([first], max_characters=10_000)
    result = format_context([first, second], max_characters=len(first_complete.text))
    assert result.text == first_complete.text
    assert result.included_evidence_count == 1
    assert result.truncated is True
    _assert_bounded(result)


def test_truncation_preserves_prefix_and_original_line_range() -> None:
    evidence = _evidence(start_line=40, end_line=75, content="0123456789" * 20)
    expected = _truncated_block(evidence, shown=17)
    result = format_context([evidence], max_characters=len(expected))
    assert result.text == expected
    assert "Content-Characters: 17/200" in result.text
    assert "Lines: 40-75" in result.text
    assert "01234567890123456\n[TRUNCATED]" in result.text
    _assert_bounded(result)


@pytest.mark.parametrize("shown", [9, 10, 99, 100])
def test_truncation_is_exact_across_shown_digit_boundaries(shown: int) -> None:
    evidence = _evidence(content="x" * 200)
    expected = _truncated_block(evidence, shown=shown)
    result = format_context([evidence], max_characters=len(expected))
    assert result.text == expected
    assert f"Content-Characters: {shown}/200" in result.text
    _assert_bounded(result)


def test_repeated_calls_are_deterministic_and_do_not_mutate_input() -> None:
    evidence = [_evidence(path="one.py"), _evidence(path="two.py")]
    original = list(evidence)
    first = format_context(evidence, max_characters=300)
    second = format_context(evidence, max_characters=300)
    assert first == second
    assert evidence == original
    _assert_bounded(first)


def test_invalid_evidence_member_is_rejected() -> None:
    with pytest.raises(ValueError):
        format_context(cast(list[ContextEvidence], [object()]), max_characters=100)


@pytest.mark.parametrize(
    "values",
    [
        {"text": 1},
        {"included_evidence_count": True},
        {"included_evidence_count": -1},
        {"truncated": 1},
        {"used_characters": True},
        {"used_characters": 1},
        {"max_characters": 0},
    ],
)
def test_formatted_context_enforces_accounting_invariants(
    values: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "text": "",
        "included_evidence_count": 0,
        "truncated": False,
        "used_characters": 0,
        "max_characters": 1,
    }
    arguments.update(values)
    with pytest.raises(ValueError):
        FormattedContext(**arguments)  # type: ignore[arg-type]
