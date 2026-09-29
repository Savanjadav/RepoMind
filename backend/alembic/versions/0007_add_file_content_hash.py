"""Add nullable raw-byte SHA-256 identity without backfilling legacy files.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("files", sa.Column("content_hash", sa.String(64), nullable=True))
    op.create_check_constraint(
        "ck_files_content_hash_sha256",
        "files",
        "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_files_content_hash_sha256", "files", type_="check")
    op.drop_column("files", "content_hash")
