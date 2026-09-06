from collections.abc import Callable

import pytest

from app.code_parser import CodeParser, CodeUnitKind, ParsedCodeUnit
from app.text_code_parser import (
    DEFAULT_MAX_CHUNK_CHARACTERS,
    ConfigurationTextParser,
    DocumentationTextParser,
)

ParserFactory = Callable[[], CodeParser]


def _parse_with(
    parser: CodeParser,
    *,
    content: str,
    relative_path: str,
) -> list[ParsedCodeUnit]:
    return parser.parse(content=content, relative_path=relative_path)


@pytest.mark.parametrize(
    ("parser_factory", "kind", "language", "relative_path"),
    [
        (
            DocumentationTextParser,
            CodeUnitKind.DOCUMENT,
            "documentation",
            "README.md",
        ),
        (
            ConfigurationTextParser,
            CodeUnitKind.CONFIG,
            "config",
            "compose.yaml",
        ),
    ],
)
def test_text_parsers_structurally_satisfy_code_parser(
    parser_factory: ParserFactory,
    kind: CodeUnitKind,
    language: str,
    relative_path: str,
) -> None:
    units = _parse_with(
        parser_factory(),
        content="meaningful text",
        relative_path=relative_path,
    )

    assert len(units) == 1
    assert units[0].kind is kind
    assert units[0].language == language
    assert units[0].relative_path == relative_path


def test_default_chunk_bound_is_two_thousand_characters() -> None:
    assert DEFAULT_MAX_CHUNK_CHARACTERS == 2_000


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "10"])
@pytest.mark.parametrize(
    "parser_factory",
    [DocumentationTextParser, ConfigurationTextParser],
)
def test_invalid_chunk_limit_is_rejected(
    parser_factory: type[DocumentationTextParser] | type[ConfigurationTextParser],
    limit: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        parser_factory(max_chunk_characters=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("content", ["", " ", "\t\r\n\n"])
@pytest.mark.parametrize(
    "parser_factory",
    [DocumentationTextParser, ConfigurationTextParser],
)
def test_empty_and_whitespace_only_content_returns_no_units(
    parser_factory: ParserFactory,
    content: str,
) -> None:
    assert parser_factory().parse(content=content, relative_path="text.txt") == []


@pytest.mark.parametrize(
    "relative_path",
    ["notes.txt", "guide.rst", "README", "LICENSE", "NOTICE", "CHANGELOG"],
)
def test_generic_documentation_does_not_apply_markdown_heading_rules(
    relative_path: str,
) -> None:
    content = "# literal text\nDetails\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path=relative_path,
    )

    assert len(units) == 1
    assert units[0].content == content
    assert units[0].symbol_name is None
    assert (units[0].start_line, units[0].end_line) == (1, 2)


@pytest.mark.parametrize("relative_path", ["README.md", "guide.markdown", "DOCS.MD"])
def test_markdown_path_enables_heading_sections(relative_path: str) -> None:
    content = "# Overview\nIntro\n\n## Details\nMore\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path=relative_path,
    )

    assert [unit.symbol_name for unit in units] == ["Overview", "Details"]
    assert [unit.content for unit in units] == [
        "# Overview\nIntro\n\n",
        "## Details\nMore\n",
    ]
    assert [(unit.start_line, unit.end_line) for unit in units] == [(1, 3), (4, 5)]


def test_meaningful_markdown_preamble_is_an_unnamed_unit() -> None:
    content = "Introductory text.\n\n# Setup\nInstall locally.\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="README.md",
    )

    assert [unit.symbol_name for unit in units] == [None, "Setup"]
    assert [unit.content for unit in units] == [
        "Introductory text.\n\n",
        "# Setup\nInstall locally.\n",
    ]
    assert [(unit.start_line, unit.end_line) for unit in units] == [(1, 2), (3, 4)]


def test_whitespace_only_preamble_is_not_a_standalone_unit() -> None:
    content = "\n \n# Setup\nInstall locally.\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="README.md",
    )

    assert len(units) == 1
    assert units[0].symbol_name == "Setup"
    assert units[0].content == "# Setup\nInstall locally.\n"
    assert (units[0].start_line, units[0].end_line) == (3, 4)


def test_nested_and_empty_headings_are_sequential_without_hierarchy() -> None:
    content = "# Top\nTop body\n### Nested\nNested body\n##\nEmpty body\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert [unit.symbol_name for unit in units] == ["Top", "Nested", None]
    assert [unit.content for unit in units] == [
        "# Top\nTop body\n",
        "### Nested\nNested body\n",
        "##\nEmpty body\n",
    ]


@pytest.mark.parametrize(
    "line",
    ["#NotAHeading", "####### Too many markers", "    # Too indented"],
)
def test_invalid_atx_heading_forms_do_not_split_document(line: str) -> None:
    content = f"# Valid\nBody\n{line}\nStill body\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert len(units) == 1
    assert units[0].symbol_name == "Valid"
    assert units[0].content == content


def test_heading_metadata_removes_optional_closing_markers_only() -> None:
    content = "# C#\nLanguage\n## OAuth ###\nDetails\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert [unit.symbol_name for unit in units] == ["C#", "OAuth"]
    assert units[0].content.startswith("# C#\n")
    assert units[1].content.startswith("## OAuth ###\n")


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_heading_inside_matching_fence_is_not_a_section(fence: str) -> None:
    content = f"# Examples\n{fence}text\n# Not a heading\n{fence}\n## Real\nBody\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert [unit.symbol_name for unit in units] == ["Examples", "Real"]
    assert "# Not a heading" in units[0].content


def test_mismatched_fence_does_not_close_fenced_mode() -> None:
    content = """# Examples
```text
# Not a heading
~~~
# Still not a heading
```
## Real
Body
"""

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert [unit.symbol_name for unit in units] == ["Examples", "Real"]
    assert "# Still not a heading" in units[0].content


def test_unmatched_fence_remains_open_through_end_of_file() -> None:
    content = "# Examples\n```text\n# Not a heading\n## Also not a heading\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="guide.md",
    )

    assert len(units) == 1
    assert units[0].symbol_name == "Examples"
    assert units[0].content == content


def test_fenced_code_is_document_content_and_is_not_recursively_parsed() -> None:
    content = "# Example\n```python\ndef dangerous():\n    raise SystemExit\n```\n"

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="README.md",
    )

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.DOCUMENT
    assert units[0].content == content


def test_oversized_markdown_section_is_bounded_and_non_overlapping() -> None:
    content = "# Details\nalpha beta gamma\ndelta epsilon\n"
    parser = DocumentationTextParser(max_chunk_characters=12)

    units = parser.parse(content=content, relative_path="README.md")

    assert len(units) > 1
    assert all(len(unit.content) <= 12 for unit in units)
    assert all(unit.symbol_name == "Details" for unit in units)
    assert "".join(unit.content for unit in units) == content
    assert units == parser.parse(content=content, relative_path="README.md")


def test_oversized_document_line_fragments_keep_one_source_line() -> None:
    content = "abcdefghijkl"

    units = DocumentationTextParser(max_chunk_characters=5).parse(
        content=content,
        relative_path="notes.txt",
    )

    assert [unit.content for unit in units] == ["abcde", "fghij", "kl"]
    assert [(unit.start_line, unit.end_line) for unit in units] == [
        (1, 1),
        (1, 1),
        (1, 1),
    ]


@pytest.mark.parametrize(
    ("parser_factory", "relative_path"),
    [
        (DocumentationTextParser, "notes.txt"),
        (ConfigurationTextParser, "settings.cfg"),
    ],
)
def test_hard_chunking_preserves_whitespace_only_internal_fragments(
    parser_factory: type[DocumentationTextParser] | type[ConfigurationTextParser],
    relative_path: str,
) -> None:
    content = "a" + " " * 9 + "b"

    units = parser_factory(max_chunk_characters=5).parse(
        content=content,
        relative_path=relative_path,
    )

    assert [unit.content for unit in units] == ["a    ", "     ", "b"]
    assert "".join(unit.content for unit in units) == content
    assert all(len(unit.content) <= 5 for unit in units)
    assert all((unit.start_line, unit.end_line) == (1, 1) for unit in units)


@pytest.mark.parametrize(
    "content",
    [
        "a\t\t\t\t\t\tb",
        "first\n\n\n   \nlast\n",
        "first\r\n\r\n   \r\nlast\r\n",
        "Café     ✓ Ready",
    ],
)
@pytest.mark.parametrize(
    ("parser_factory", "relative_path"),
    [
        (DocumentationTextParser, "notes.txt"),
        (ConfigurationTextParser, "settings.cfg"),
    ],
)
def test_hard_chunking_is_lossless_for_whitespace_newlines_and_unicode(
    parser_factory: type[DocumentationTextParser] | type[ConfigurationTextParser],
    relative_path: str,
    content: str,
) -> None:
    units = parser_factory(max_chunk_characters=5).parse(
        content=content,
        relative_path=relative_path,
    )

    assert "".join(unit.content for unit in units) == content
    assert all(len(unit.content) <= 5 for unit in units)


@pytest.mark.parametrize(
    ("parser_factory", "relative_path"),
    [
        (DocumentationTextParser, "notes.txt"),
        (ConfigurationTextParser, "settings.cfg"),
    ],
)
def test_whitespace_internal_chunk_has_exact_physical_line_range(
    parser_factory: type[DocumentationTextParser] | type[ConfigurationTextParser],
    relative_path: str,
) -> None:
    content = "a    \n\n    b"

    units = parser_factory(max_chunk_characters=5).parse(
        content=content,
        relative_path=relative_path,
    )

    assert [unit.content for unit in units] == ["a    ", "\n\n   ", " b"]
    assert [(unit.start_line, unit.end_line) for unit in units] == [
        (1, 1),
        (1, 3),
        (3, 3),
    ]
    assert "".join(unit.content for unit in units) == content


def test_unicode_markdown_content_and_crlf_are_preserved_exactly() -> None:
    content = '# Café\r\n\r\ngreeting: "✓ Ready"\r\n## Résumé\r\nFinal'

    units = DocumentationTextParser().parse(
        content=content,
        relative_path="docs/guide.md",
    )

    assert [unit.symbol_name for unit in units] == ["Café", "Résumé"]
    assert [unit.content for unit in units] == [
        '# Café\r\n\r\ngreeting: "✓ Ready"\r\n',
        "## Résumé\r\nFinal",
    ]
    assert [(unit.start_line, unit.end_line) for unit in units] == [(1, 3), (4, 5)]


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("package.json", '{\n  "enabled": true\n}\n'),
        ("compose.yaml", "services:\n  api: {}\n"),
        ("pyproject.toml", '[project]\nname = "repomind"\n'),
        (".editorconfig", "root = true\n"),
        ("Dockerfile", "FROM python:3.12\n"),
    ],
)
def test_small_configuration_is_one_exact_unnamed_unit(
    relative_path: str,
    content: str,
) -> None:
    unit = ConfigurationTextParser().parse(
        content=content,
        relative_path=relative_path,
    )[0]

    assert unit.kind is CodeUnitKind.CONFIG
    assert unit.language == "config"
    assert unit.symbol_name is None
    assert unit.content == content
    assert unit.relative_path == relative_path
    assert (unit.start_line, unit.end_line) == (1, content.count("\n"))


def test_comments_only_configuration_remains_retrievable() -> None:
    content = "# Explain this setting\n# before enabling it\n"

    units = ConfigurationTextParser().parse(
        content=content,
        relative_path="settings.yaml",
    )

    assert len(units) == 1
    assert units[0].content == content
    assert units[0].symbol_name is None


def test_malformed_configuration_is_chunked_without_validation() -> None:
    content = '{ "missing": [ }\n'

    units = ConfigurationTextParser().parse(
        content=content,
        relative_path="broken.json",
    )

    assert [unit.content for unit in units] == [content]


def test_oversized_configuration_is_exact_bounded_and_deterministic() -> None:
    content = "first: 12345\nsecond: 67890\nthird: ✓\n"
    parser = ConfigurationTextParser(max_chunk_characters=14)

    units = parser.parse(content=content, relative_path="settings.yaml")

    assert all(len(unit.content) <= 14 for unit in units)
    assert "".join(unit.content for unit in units) == content
    assert all(unit.symbol_name is None for unit in units)
    assert units == parser.parse(content=content, relative_path="settings.yaml")


def test_oversized_config_line_fragments_keep_one_source_line() -> None:
    content = "greeting=✓✓✓✓✓✓"

    units = ConfigurationTextParser(max_chunk_characters=5).parse(
        content=content,
        relative_path=".env.example",
    )

    assert "".join(unit.content for unit in units) == content
    assert all(len(unit.content) <= 5 for unit in units)
    assert all((unit.start_line, unit.end_line) == (1, 1) for unit in units)


def test_configuration_preserves_whitespace_unicode_crlf_and_final_line() -> None:
    content = 'title = "Café"  \r\n\r\nstatus = "✓"'

    units = ConfigurationTextParser().parse(
        content=content,
        relative_path="settings.toml",
    )

    assert len(units) == 1
    assert units[0].content == content
    assert (units[0].start_line, units[0].end_line) == (1, 3)
