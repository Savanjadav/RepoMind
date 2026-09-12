from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.repository import Repository


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    code_unit_id: UUID
    file_id: UUID
    path: str
    kind: CodeUnitKind
    content: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    lexical_score: int


def search_code_units_lexically(
    session: Session,
    *,
    repository_id: UUID,
    query: str,
    limit: int = 10,
) -> list[LexicalSearchResult]:
    _validate_query(query)
    _validate_limit(limit)

    if session.get(Repository, repository_id) is None:
        raise ValueError("Repository does not exist")

    lowered_query = func.lower(query)
    symbol_exact = CodeUnit.symbol_name == query
    symbol_exact_case_insensitive = func.lower(CodeUnit.symbol_name) == lowered_query
    symbol_substring = func.strpos(CodeUnit.symbol_name, query) > 0
    symbol_substring_case_insensitive = (
        func.strpos(func.lower(CodeUnit.symbol_name), lowered_query) > 0
    )
    path_substring = func.strpos(File.path, query) > 0
    path_substring_case_insensitive = (
        func.strpos(func.lower(File.path), lowered_query) > 0
    )
    content_substring = func.strpos(CodeUnit.content, query) > 0
    content_substring_case_insensitive = (
        func.strpos(func.lower(CodeUnit.content), lowered_query) > 0
    )
    search_document = func.concat_ws(
        " ",
        func.coalesce(CodeUnit.symbol_name, ""),
        File.path,
        CodeUnit.content,
    )
    full_text_match = func.to_tsvector("simple", search_document).bool_op("@@")(
        func.plainto_tsquery("simple", query)
    )
    score_conditions = (
        (symbol_exact, 9),
        (symbol_exact_case_insensitive, 8),
        (symbol_substring, 7),
        (symbol_substring_case_insensitive, 6),
        (path_substring, 5),
        (path_substring_case_insensitive, 4),
        (content_substring, 3),
        (content_substring_case_insensitive, 2),
        (full_text_match, 1),
    )
    lexical_score = case(*score_conditions, else_=0).label("lexical_score")

    statement = (
        select(
            CodeUnit.id,
            CodeUnit.file_id,
            File.path,
            CodeUnit.kind,
            CodeUnit.content,
            CodeUnit.language,
            CodeUnit.start_line,
            CodeUnit.end_line,
            CodeUnit.symbol_name,
            lexical_score,
        )
        .join(File, CodeUnit.file_id == File.id)
        .where(
            File.repository_id == repository_id,
            or_(*(condition for condition, _ in score_conditions)),
        )
        .order_by(
            lexical_score.desc(),
            File.path.asc(),
            CodeUnit.start_line.asc(),
            CodeUnit.end_line.desc(),
            CodeUnit.kind.asc(),
            CodeUnit.id.asc(),
        )
        .limit(limit)
    )

    return [
        LexicalSearchResult(
            code_unit_id=code_unit_id,
            file_id=file_id,
            path=path,
            kind=CodeUnitKind(kind),
            content=content,
            language=language,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            lexical_score=int(score),
        )
        for (
            code_unit_id,
            file_id,
            path,
            kind,
            content,
            language,
            start_line,
            end_line,
            symbol_name,
            score,
        ) in session.execute(statement)
    ]


def _validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    if not query.strip():
        raise ValueError("Query must contain non-whitespace characters")
    if len(query) > 2000:
        raise ValueError("Query must contain at most 2000 characters")


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer between 1 and 100")
