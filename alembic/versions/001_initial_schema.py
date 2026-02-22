"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2026-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Tasks table
    op.create_table(
        'tasks',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('user_request', sa.Text(), nullable=False),
        sa.Column('status', sa.String(32), default='pending', index=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), default=0),
        sa.Column('input_tokens', sa.Integer(), default=0),
        sa.Column('output_tokens', sa.Integer(), default=0),
        sa.Column('step_count', sa.Integer(), default=0),
        sa.Column('tool_call_count', sa.Integer(), default=0),
        sa.Column('iteration_count', sa.Integer(), default=0),
        sa.Column('max_tokens', sa.Integer(), default=100000),
        sa.Column('max_steps', sa.Integer(), default=50),
        sa.Column('max_tool_calls', sa.Integer(), default=100),
        sa.Column('max_iterations', sa.Integer(), default=10),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_tasks_created_at', 'tasks', ['created_at'])
    op.create_index('ix_tasks_status_created', 'tasks', ['status', 'created_at'])

    # Task steps table
    op.create_table(
        'task_steps',
        sa.Column('id', sa.String(64), primary_key=True),
        sa.Column('task_id', sa.String(64), sa.ForeignKey('tasks.id', ondelete='CASCADE'), index=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('status', sa.String(32), default='pending'),
        sa.Column('sequence', sa.Integer(), default=0),
        sa.Column('dependencies', sa.JSON(), default=list),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_task_steps_task_sequence', 'task_steps', ['task_id', 'sequence'])

    # Tool calls log table
    op.create_table(
        'tool_calls',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('task_id', sa.String(64), sa.ForeignKey('tasks.id', ondelete='CASCADE'), index=True),
        sa.Column('step_id', sa.String(64), nullable=True),
        sa.Column('tool_name', sa.String(128), index=True),
        sa.Column('arguments', sa.JSON(), default=dict),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('success', sa.Boolean(), default=False),
        sa.Column('execution_time_ms', sa.Float(), default=0.0),
        sa.Column('retry_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
    )
    op.create_index('ix_tool_calls_task_created', 'tool_calls', ['task_id', 'created_at'])
    # ix_tool_calls_tool_name is already created by index=True on tool_name column

    # Checkpoints table
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('thread_id', sa.String(64), index=True),
        sa.Column('checkpoint_id', sa.String(64), index=True),
        sa.Column('parent_checkpoint_id', sa.String(64), nullable=True),
        sa.Column('checkpoint_data', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), default=dict),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
    )
    op.create_index(
        'ix_checkpoints_thread_checkpoint',
        'checkpoints',
        ['thread_id', 'checkpoint_id'],
        unique=True
    )

    # Memory entries table
    op.create_table(
        'memory_entries',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('task_id', sa.String(64), nullable=True, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_type', sa.String(32), default='text', index=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('importance', sa.Float(), default=0.5),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
    )
    # Note: IVFFlat index should be created after data is loaded for better performance
    # op.execute('''
    #     CREATE INDEX ix_memory_entries_embedding 
    #     ON memory_entries 
    #     USING ivfflat (embedding vector_cosine_ops)
    #     WITH (lists = 100)
    # ''')


def downgrade() -> None:
    op.drop_table('memory_entries')
    op.drop_table('checkpoints')
    op.drop_table('tool_calls')
    op.drop_table('task_steps')
    op.drop_table('tasks')
    op.execute('DROP EXTENSION IF EXISTS vector')
