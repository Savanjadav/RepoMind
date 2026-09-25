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


class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "source_file_id",
            "target_file_id",
            "relationship_type",
            name="uq_relationships_edge",
        ),
        CheckConstraint("relationship_type = 'imports'", name="ck_relationships_type"),
        CheckConstraint(
            "source_file_id != target_file_id", name="ck_relationships_not_self"
        ),
        ForeignKeyConstraint(
            ["repository_id", "source_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_relationships_source",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["repository_id", "target_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_relationships_target",
            ondelete="CASCADE",
        ),
        Index("ix_relationships_target", "repository_id", "target_file_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    repository_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "repositories.id", ondelete="CASCADE", name="fk_relationships_repository"
        ),
        nullable=False,
    )
    source_file_id: Mapped[UUID] = mapped_column(nullable=False)
    target_file_id: Mapped[UUID] = mapped_column(nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False)
