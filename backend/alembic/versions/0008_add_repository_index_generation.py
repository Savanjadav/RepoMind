"""Add transactional repository snapshot generations.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "repositories",
        sa.Column(
            "index_generation", sa.BigInteger(), server_default="0", nullable=False
        ),
    )
    op.create_check_constraint(
        "ck_repositories_index_generation", "repositories", "index_generation >= 0"
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_repositories_index_generation", "repositories", type_="check"
    )
    op.drop_column("repositories", "index_generation")
