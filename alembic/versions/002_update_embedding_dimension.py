"""Update embedding dimension from 1536 to 1024 for text-embedding-v4.

Revision ID: 002
Revises: 001
Create Date: 2026-02-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade: Change embedding column from Vector(1536) to Vector(1024).
    
    Note: This will truncate existing embeddings. If you have important data,
    you should re-generate embeddings after this migration.
    """
    # Drop the existing index first
    op.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding")
    
    # Alter the column type
    # Note: This requires dropping and recreating as pgvector doesn't support direct alter
    op.execute("""
        ALTER TABLE memory_entries 
        ALTER COLUMN embedding TYPE vector(1024) 
        USING embedding::vector(1024)
    """)
    
    # Recreate the index with the new dimension
    op.execute("""
        CREATE INDEX ix_memory_entries_embedding 
        ON memory_entries 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)


def downgrade() -> None:
    """Downgrade: Revert embedding column from Vector(1024) to Vector(1536)."""
    # Drop the existing index
    op.execute("DROP INDEX IF EXISTS ix_memory_entries_embedding")
    
    # Alter the column type back
    op.execute("""
        ALTER TABLE memory_entries 
        ALTER COLUMN embedding TYPE vector(1536) 
        USING embedding::vector(1536)
    """)
    
    # Recreate the index
    op.execute("""
        CREATE INDEX ix_memory_entries_embedding 
        ON memory_entries 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)
