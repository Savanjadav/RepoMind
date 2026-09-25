"""Repository-owned, static CodeUnit call hints (including recursion)."""

from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class CodeRelationship(Base):
    __tablename__ = "code_relationships"
    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "source_code_unit_id",
            "target_code_unit_id",
            "relationship_type",
            name="uq_code_relationships_edge",
        ),
        CheckConstraint(
            "relationship_type = 'calls'", name="ck_code_relationships_type"
        ),
        ForeignKeyConstraint(
            ["repository_id", "source_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_code_relationships_source_file",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["repository_id", "target_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_code_relationships_target_file",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["source_file_id", "source_code_unit_id"],
            ["code_units.file_id", "code_units.id"],
            name="fk_code_relationships_source_unit",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["target_file_id", "target_code_unit_id"],
            ["code_units.file_id", "code_units.id"],
            name="fk_code_relationships_target_unit",
            ondelete="CASCADE",
        ),
        Index("ix_code_relationships_target", "repository_id", "target_code_unit_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    repository_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "repositories.id",
            name="fk_code_relationships_repository",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    source_file_id: Mapped[UUID] = mapped_column(nullable=False)
    source_code_unit_id: Mapped[UUID] = mapped_column(nullable=False)
    target_file_id: Mapped[UUID] = mapped_column(nullable=False)
    target_code_unit_id: Mapped[UUID] = mapped_column(nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False)
