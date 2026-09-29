from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.code_unit import CodeUnit
    from app.models.repository import Repository


class File(Base):
    __tablename__ = "files"
    __table_args__ = (
        CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_files_content_hash_sha256",
        ),
        UniqueConstraint("repository_id", "id", name="uq_files_repository_id_id"),
        UniqueConstraint(
            "repository_id",
            "path",
            name="uq_files_repository_id_path",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    repository_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "repositories.id",
            name="fk_files_repository_id_repositories",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    repository: Mapped["Repository"] = relationship(back_populates="files")
    code_units: Mapped[list["CodeUnit"]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
