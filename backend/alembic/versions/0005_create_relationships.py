"""Create local file import relationships.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_files_repository_id_id", "files", ["repository_id", "id"]
    )
    op.create_table(
        "relationships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("repository_id", sa.Uuid(), nullable=False),
        sa.Column("source_file_id", sa.Uuid(), nullable=False),
        sa.Column("target_file_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_type", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "repository_id",
            "source_file_id",
            "target_file_id",
            "relationship_type",
            name="uq_relationships_edge",
        ),
        sa.CheckConstraint(
            "relationship_type = 'imports'", name="ck_relationships_type"
        ),
        sa.CheckConstraint(
            "source_file_id != target_file_id", name="ck_relationships_not_self"
        ),
        sa.ForeignKeyConstraint(
            ["repository_id"],
            ["repositories.id"],
            name="fk_relationships_repository",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id", "source_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_relationships_source",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id", "target_file_id"],
            ["files.repository_id", "files.id"],
            name="fk_relationships_target",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_relationships_target", "relationships", ["repository_id", "target_file_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_relationships_target", table_name="relationships")
    op.drop_table("relationships")
    op.drop_constraint("uq_files_repository_id_id", "files", type_="unique")
