"""Map canonical answer markers to supplied evidence, without integrity enforcement."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag_context import ContextEvidence

_MARKER = re.compile(r"\[Evidence ([1-9][0-9]*)\]")


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: int
    repository_name: str
    path: str
    symbol_name: str | None
    start_line: int
    end_line: int


def extract_citations(
    answer: str,
    *,
    included_evidence: Sequence[ContextEvidence],
) -> list[Citation]:
    lookup = {
        str(index): (index, evidence)
        for index, evidence in enumerate(included_evidence, start=1)
    }
    seen: set[str] = set()
    citations: list[Citation] = []
    for match in _MARKER.finditer(answer):
        key = match.group(1)
        entry = lookup.get(key)
        if entry is None or key in seen:
            continue
        seen.add(key)
        index, evidence = entry
        citations.append(
            Citation(
                evidence_id=index,
                repository_name=evidence.repository_name,
                path=evidence.path,
                symbol_name=evidence.symbol_name,
                start_line=evidence.start_line,
                end_line=evidence.end_line,
            )
        )
    return citations
