import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.models import File, Repository
from app.repository_file_persistence import persist_repository_files
from app.repository_files import RepositoryFileCandidate

DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture
def database_session() -> Iterator[Session]:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                yield session
        finally:
            transaction.rollback()
    engine.dispose()


def _repository(session: Session, name: str = "Repository") -> Repository:
    repository = Repository(
        name=name,
        source=f"https://example.com/{uuid4()}.git",
    )
    session.add(repository)
    session.flush()
    return repository


@pytest.mark.parametrize("digest", [None, "0" * 64, "abcdef0123456789" * 4])
def test_content_hash_accepts_legacy_null_and_sha256(
    database_session: Session, digest: str | None
) -> None:
    repo = _repository(database_session)
    file = File(repository_id=repo.id, path="hashed.py", content_hash=digest)
    database_session.add(file)
    database_session.flush()
    assert (
        database_session.scalar(select(File.content_hash).where(File.id == file.id))
        == digest
    )


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64, "a" * 63 + "\n"])
def test_content_hash_rejects_invalid_sha256(
    database_session: Session, digest: str
) -> None:
    repo = _repository(database_session)
    with pytest.raises(IntegrityError), database_session.begin_nested():
        database_session.add(
            File(repository_id=repo.id, path="bad.py", content_hash=digest)
        )
        database_session.flush()


def test_content_hash_schema_preserves_file_constraints(
    database_session: Session,
) -> None:
    inspector = inspect(database_session.connection())
    column = next(
        c for c in inspector.get_columns("files") if c["name"] == "content_hash"
    )
    assert str(column["type"]) == "VARCHAR(64)"
    assert column["nullable"] is True and column["default"] is None
    assert {c["name"] for c in inspector.get_unique_constraints("files")} == {
        "uq_files_repository_id_id",
        "uq_files_repository_id_path",
    }
    assert "ck_files_content_hash_sha256" in {
        c["name"] for c in inspector.get_check_constraints("files")
    }


def _candidate(
    relative_path: str,
    *,
    workspace_path: Path | None = None,
) -> RepositoryFileCandidate:
    return RepositoryFileCandidate(
        path=workspace_path or Path("/machine-local-clone") / relative_path,
        relative_path=relative_path,
        size_bytes=100,
    )


def test_persists_one_candidate_with_repository_relationship(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    candidate = _candidate("src/main.py")

    inserted = persist_repository_files(
        database_session,
        repository.id,
        [candidate],
    )

    assert len(inserted) == 1
    stored_file = inserted[0]
    assert isinstance(stored_file.id, UUID)
    assert stored_file.repository_id == repository.id
    assert stored_file.path == candidate.relative_path
    assert stored_file.path != str(candidate.path)
    assert stored_file.repository is repository


def test_persists_multiple_candidates_in_deterministic_order(
    database_session: Session,
) -> None:
    repository = _repository(database_session)

    inserted = persist_repository_files(
        database_session,
        repository.id,
        [
            _candidate("src/z.py"),
            _candidate("README.md"),
            _candidate("src/a.py"),
        ],
    )

    assert [file.path for file in inserted] == [
        "README.md",
        "src/a.py",
        "src/z.py",
    ]
    assert all(isinstance(file.id, UUID) for file in inserted)


def test_repeated_and_duplicate_candidates_are_idempotent(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    candidates = [
        _candidate("src/main.py"),
        _candidate("README.md"),
        _candidate("src/main.py"),
    ]

    first = persist_repository_files(database_session, repository.id, candidates)
    second = persist_repository_files(database_session, repository.id, candidates)
    stored_paths = list(
        database_session.scalars(
            select(File.path)
            .where(File.repository_id == repository.id)
            .order_by(File.path)
        )
    )

    assert [file.path for file in first] == ["README.md", "src/main.py"]
    assert second == []
    assert stored_paths == ["README.md", "src/main.py"]


def test_same_path_is_independent_between_repositories(
    database_session: Session,
) -> None:
    first_repository = _repository(database_session, "First repository")
    second_repository = _repository(database_session, "Second repository")
    candidate = _candidate("src/main.py")

    first = persist_repository_files(
        database_session,
        first_repository.id,
        [candidate],
    )
    second = persist_repository_files(
        database_session,
        second_repository.id,
        [candidate],
    )

    assert first[0].repository_id == first_repository.id
    assert second[0].repository_id == second_repository.id
    assert (
        database_session.scalar(
            select(func.count()).select_from(File).where(File.path == "src/main.py")
        )
        == 2
    )


def test_existing_rows_are_not_duplicated_or_deleted(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    other_repository = _repository(database_session, "Other repository")
    existing = [
        File(repository_id=repository.id, path="README.md"),
        File(repository_id=repository.id, path="src/old.py"),
        File(repository_id=other_repository.id, path="src/other.py"),
    ]
    database_session.add_all(existing)
    database_session.flush()

    inserted = persist_repository_files(
        database_session,
        repository.id,
        [_candidate("README.md"), _candidate("src/main.py")],
    )
    repository_paths = list(
        database_session.scalars(
            select(File.path)
            .where(File.repository_id == repository.id)
            .order_by(File.path)
        )
    )
    other_paths = list(
        database_session.scalars(
            select(File.path).where(File.repository_id == other_repository.id)
        )
    )

    assert [file.path for file in inserted] == ["src/main.py"]
    assert repository_paths == ["README.md", "src/main.py", "src/old.py"]
    assert other_paths == ["src/other.py"]


def test_empty_candidates_are_a_true_no_op() -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    assert persist_repository_files(session, uuid4(), []) == []
    session_mock.assert_not_called()
    assert session_mock.method_calls == []


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "/etc/passwd",
        "../secret",
        "src/../secret",
        "./src/main.py",
        "src/./main.py",
        "src//main.py",
        "src\\main.py",
        "bad\nname.py",
        "bad\x7fname.py",
    ],
)
def test_invalid_relative_paths_are_rejected_before_database_work(
    invalid_path: str,
) -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError):
        persist_repository_files(session, uuid4(), [_candidate(invalid_path)])

    assert session_mock.method_calls == []


def test_all_paths_are_validated_before_database_or_orm_changes() -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)
    candidates = [_candidate("src/main.py"), _candidate("../invalid")]

    with pytest.raises(ValueError):
        persist_repository_files(session, uuid4(), candidates)

    assert session_mock.method_calls == []


def test_unicode_path_is_preserved(database_session: Session) -> None:
    repository = _repository(database_session)

    inserted = persist_repository_files(
        database_session,
        repository.id,
        [_candidate("文档/résumé.py")],
    )

    assert [file.path for file in inserted] == ["文档/résumé.py"]


def test_caller_controls_transaction_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    repository_id: UUID
    file_id: UUID
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection) as session:
            repository = _repository(session)

            def unexpected_transaction_method() -> None:
                pytest.fail("Persistence service must not own the transaction")

            monkeypatch.setattr(session, "commit", unexpected_transaction_method)
            monkeypatch.setattr(session, "rollback", unexpected_transaction_method)
            inserted = persist_repository_files(
                session,
                repository.id,
                [_candidate("src/main.py")],
            )
            repository_id = repository.id
            file_id = inserted[0].id

        transaction.rollback()

    with Session(engine) as verification_session:
        assert verification_session.get(Repository, repository_id) is None
        assert verification_session.get(File, file_id) is None
    engine.dispose()


def test_database_unique_constraint_remains_final_integrity_boundary(
    database_session: Session,
) -> None:
    repository = _repository(database_session)

    with pytest.raises(IntegrityError):
        with database_session.begin_nested():
            database_session.add_all(
                [
                    File(repository_id=repository.id, path="src/main.py"),
                    File(repository_id=repository.id, path="src/main.py"),
                ]
            )
            database_session.flush()
