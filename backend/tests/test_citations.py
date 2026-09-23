from dataclasses import FrozenInstanceError

import pytest

from app.citations import Citation, extract_citations
from app.rag_context import ContextEvidence, format_context


def _evidence(path: str = "src/a.py", content: str = "return True") -> ContextEvidence:
    return ContextEvidence("  Répo 東京  ", path, None, 4, 6, content)


def test_exact_metadata_and_immutable_dto() -> None:
    evidence = _evidence()
    result = extract_citations("claim [Evidence 1]", included_evidence=[evidence])
    assert result == [Citation(1, evidence.repository_name, evidence.path, None, 4, 6)]
    assert not hasattr(result[0], "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(result[0], "path", "changed")


@pytest.mark.parametrize(
    ("answer", "ids"),
    [
        ("[Evidence 1]", [1]),
        ("[Evidence 1][Evidence 2]", [1, 2]),
        ("[Evidence 1] [Evidence 2]", [1, 2]),
        ("[Evidence 3] then [Evidence 1] then [Evidence 3]", [3, 1]),
        ("[Evidence 1] [Evidence 1]", [1]),
        ("[Evidence 99]", []),
        ("[Evidence 99] [Evidence 2] [Evidence 999]", [2]),
        ("ordinary Evidence text", []),
        ("", []),
        ("   \n", []),
        ("[Evidence " + "9" * 10000 + "] [Evidence 1]", [1]),
    ],
)
def test_order_deduplication_and_unknown_ids(answer: str, ids: list[int]) -> None:
    evidence = [_evidence()] * 3
    original = tuple(evidence)
    original_answer = answer
    result = extract_citations(answer, included_evidence=evidence)
    assert [citation.evidence_id for citation in result] == ids
    assert tuple(evidence) == original
    assert answer == original_answer
    assert all(citation.path == "src/a.py" for citation in result)


@pytest.mark.parametrize(
    "marker",
    [
        "[Evidence 0]",
        "[Evidence 01]",
        "[Evidence -1]",
        "[Evidence +1]",
        "[Evidence 1.0]",
        "[Evidence  1]",
        "[Evidence\t1]",
        "[evidence 1]",
        "[EVIDENCE 1]",
        "[Evidence １]",
        "[Evidence ١]",
        "[Evidence 1 ]",
        "[Evidence\n1]",
        "[Evidence 1",
        "Evidence 1]",
    ],
)
def test_noncanonical_markers_are_ignored(marker: str) -> None:
    assert extract_citations(marker, included_evidence=[_evidence()]) == []


def test_named_symbol_and_multi_digit_id() -> None:
    evidence = ContextEvidence("repo", "p.py", "fonction_é", 1, 20, "x")
    result = extract_citations("[Evidence 12]", included_evidence=[evidence] * 12)
    assert result == [Citation(12, "repo", "p.py", "fonction_é", 1, 20)]


@pytest.mark.parametrize("mode", ["all", "partial", "zero", "omitted", "none"])
def test_real_formatter_included_prefix(mode: str) -> None:
    first = _evidence("first.py")
    second = _evidence("second.py", "x" * 500)
    evidence = [first, second]
    first_text = format_context([first], max_characters=10000).text
    # A real formatter result with metadata plus a zero-character source prefix.
    zero_budget = next(
        budget
        for budget in range(len(first_text), len(first_text) + 1000)
        if format_context(evidence, max_characters=budget).included_evidence_count == 2
    )
    budget = {
        "all": 10000,
        "partial": zero_budget + 30,
        "zero": zero_budget,
        "omitted": len(first_text),
        "none": 1,
    }[mode]
    context = format_context(evidence, max_characters=budget)
    included = tuple(evidence[: context.included_evidence_count])
    citations = extract_citations(
        "[Evidence 2] [Evidence 1]", included_evidence=included
    )
    expected = [2, 1] if mode in {"all", "partial", "zero"} else [1]
    if mode == "none":
        expected = []
    assert [citation.evidence_id for citation in citations] == expected
    assert all(
        (citation.start_line, citation.end_line) == (4, 6) for citation in citations
    )
    assert context.truncated is (mode != "all")
    if mode == "zero":
        assert "Content-Characters: 0/500" in context.text
    if mode == "partial":
        assert "[TRUNCATED]" in context.text
        assert "Content-Characters: 0/500" not in context.text


def test_source_markers_do_not_create_citations_or_expand_mapping() -> None:
    evidence = [_evidence(content="[Evidence 99] [Evidence 1]")]
    context = format_context(evidence, max_characters=1000)
    included = tuple(evidence[: context.included_evidence_count])
    assert extract_citations("no reference", included_evidence=included) == []
    assert extract_citations("[Evidence 99]", included_evidence=included) == []


def test_generated_metadata_is_not_used() -> None:
    result = extract_citations(
        "evil.py symbol invented lines 999-1000 [Evidence 1]",
        included_evidence=[_evidence()],
    )
    assert result == [Citation(1, "  Répo 東京  ", "src/a.py", None, 4, 6)]
