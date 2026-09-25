"""Create conservative CodeUnit call relationships.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_code_units_file_id_id", "code_units", ["file_id", "id"]
    )
    op.create_table(
        "code_relationships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("repository_id", sa.Uuid(), nullable=False),
        sa.Column("source_file_id", sa.Uuid(), nullable=False),
        sa.Column("source_code_unit_id", sa.Uuid(), nullable=False),
        sa.Column("target_file_id", sa.Uuid(), nullable=False),
        sa.Column("target_code_unit_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_type", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "relationship_type = 'calls'", name="ck_code_relationships_type"
        ),
        sa.UniqueConstraint(
            "repository_id",
            "source_code_unit_id",
            "target_code_unit_id",
            "relationship_type",
            name="uq_code_relationships_edge",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id"],
            ["repositories.id"],
            name="fk_code_relationships_repository",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id", "source_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_code_relationships_source_file",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id", "target_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_code_relationships_target_file",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_file_id", "source_code_unit_id"],
            ["code_units.file_id", "code_units.id"],
            name="fk_code_relationships_source_unit",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_file_id", "target_code_unit_id"],
            ["code_units.file_id", "code_units.id"],
            name="fk_code_relationships_target_unit",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_code_relationships_target",
        "code_relationships",
        ["repository_id", "target_code_unit_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_code_relationships_target", table_name="code_relationships")
    op.drop_table("code_relationships")
    op.drop_constraint("uq_code_units_file_id_id", "code_units", type_="unique")
