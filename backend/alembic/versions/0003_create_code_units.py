"""Create code units table.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "code_units",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("file_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=64), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("symbol_name", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('class', 'config', 'document', 'function', 'import')",
            name="ck_code_units_kind",
        ),
        sa.CheckConstraint(
            "start_line >= 1",
            name="ck_code_units_start_line_positive",
        ),
        sa.CheckConstraint(
            "end_line >= start_line",
            name="ck_code_units_end_line_not_before_start",
        ),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["files.id"],
            name="fk_code_units_file_id_files",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_code_units_file_id",
        "code_units",
        ["file_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_code_units_file_id", table_name="code_units")
    op.drop_table("code_units")
