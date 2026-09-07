from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.file import File


class CodeUnit(Base):
    __tablename__ = "code_units"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('class', 'config', 'document', 'function', 'import')",
            name="ck_code_units_kind",
        ),
        CheckConstraint(
            "start_line >= 1",
            name="ck_code_units_start_line_positive",
        ),
        CheckConstraint(
            "end_line >= start_line",
            name="ck_code_units_end_line_not_before_start",
        ),
        Index("ix_code_units_file_id", "file_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    file_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "files.id",
            name="fk_code_units_file_id_files",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(64), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    file: Mapped["File"] = relationship(back_populates="code_units")
