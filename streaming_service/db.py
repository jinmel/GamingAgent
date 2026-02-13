"""PostgreSQL persistence layer using asyncpg."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("streaming_service")

_pool = None


async def init_pool(dsn: str) -> None:
    """Create the asyncpg connection pool and ensure tables exist."""
    global _pool
    try:
        import asyncpg

        _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
        await _ensure_tables()
        logger.info("Database pool initialised (%s)", dsn)
    except Exception as exc:
        logger.warning("Could not connect to database: %s — running without persistence", exc)
        _pool = None


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


def is_available() -> bool:
    """Return True when the pool is ready for queries."""
    return _pool is not None


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """\
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'session_status') THEN
        CREATE TYPE session_status AS ENUM (
            'running', 'completed', 'exhausted', 'cancelled', 'failed', 'deleted'
        );
    ELSE
        -- Idempotent migration: add 'exhausted' if upgrading from an older schema.
        BEGIN
            ALTER TYPE session_status ADD VALUE IF NOT EXISTS 'exhausted';
        EXCEPTION WHEN duplicate_object THEN NULL;
        END;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS sessions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_name         TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    prompt            TEXT NOT NULL DEFAULT '',
    max_steps         INT NOT NULL DEFAULT 200,
    observation_mode  TEXT NOT NULL DEFAULT 'vision',
    harness           BOOLEAN NOT NULL DEFAULT FALSE,
    max_memory        INT NOT NULL DEFAULT 10,
    seed              INT,
    status            session_status NOT NULL DEFAULT 'running',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ,
    total_steps       INT NOT NULL DEFAULT 0,
    total_reward      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cache_dir         TEXT,
    parent_session_id UUID REFERENCES sessions(id),
    resumed_from_step INT
);

CREATE TABLE IF NOT EXISTS frames (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    step        INT NOT NULL,
    thought     TEXT NOT NULL DEFAULT '',
    action      TEXT NOT NULL DEFAULT '',
    reward      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    done        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    step           INT NOT NULL,
    env_state      JSONB NOT NULL,
    adapter_state  JSONB NOT NULL,
    agent_state    JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(session_id, step)
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_game_name ON sessions(game_name);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_frames_session_step ON frames(session_id, step);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session_step ON checkpoints(session_id, step);
"""


async def _ensure_tables() -> None:
    """Run idempotent DDL to create tables and indexes."""
    async with _pool.acquire() as conn:
        await conn.execute(_DDL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_to_dict(record) -> Dict[str, Any]:
    """Convert an asyncpg Record to a plain dict with JSON-safe values."""
    d = dict(record)
    for k, v in d.items():
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


# ---------------------------------------------------------------------------
# Session queries
# ---------------------------------------------------------------------------

async def create_session(
    *,
    game_name: str,
    model_name: str,
    prompt: str = "",
    max_steps: int = 200,
    observation_mode: str = "vision",
    harness: bool = False,
    max_memory: int = 10,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Insert a new session and return its UUID as a string."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions
                (game_name, model_name, prompt, max_steps,
                 observation_mode, harness, max_memory, seed, cache_dir)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            RETURNING id
            """,
            game_name, model_name, prompt, max_steps,
            observation_mode, harness, max_memory, seed, cache_dir,
        )
        return str(row["id"])


async def create_resumed_session(
    *,
    parent_session_id: str,
    resumed_from_step: int,
    game_name: str,
    model_name: str,
    prompt: str = "",
    max_steps: int = 200,
    observation_mode: str = "vision",
    harness: bool = False,
    max_memory: int = 10,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Insert a session that resumes from a parent checkpoint."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions
                (game_name, model_name, prompt, max_steps,
                 observation_mode, harness, max_memory, seed, cache_dir,
                 parent_session_id, resumed_from_step)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::uuid,$11)
            RETURNING id
            """,
            game_name, model_name, prompt, max_steps,
            observation_mode, harness, max_memory, seed, cache_dir,
            parent_session_id, resumed_from_step,
        )
        return str(row["id"])


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single session by ID. Returns None if not found or deleted."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM sessions WHERE id = $1::uuid AND status != 'deleted'",
            session_id,
        )
        return _record_to_dict(row) if row else None


async def list_sessions(
    *,
    game_name: Optional[str] = None,
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return a paginated list of sessions (excluding deleted) and total count."""
    conditions = ["status != 'deleted'"]
    params: list[Any] = []
    idx = 1

    if game_name:
        conditions.append(f"game_name = ${idx}")
        params.append(game_name)
        idx += 1
    if model_name:
        conditions.append(f"model_name = ${idx}")
        params.append(model_name)
        idx += 1
    if status:
        conditions.append(f"status = ${idx}::session_status")
        params.append(status)
        idx += 1

    where = " AND ".join(conditions)

    async with _pool.acquire() as conn:
        count_row = await conn.fetchrow(
            f"SELECT count(*) AS cnt FROM sessions WHERE {where}", *params,
        )
        total = count_row["cnt"]

        rows = await conn.fetch(
            f"""
            SELECT * FROM sessions
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params, limit, offset,
        )
        return [_record_to_dict(r) for r in rows], total


async def finish_session(
    session_id: str,
    total_steps: int,
    total_reward: float,
    status: str = "completed",
) -> None:
    """Mark a session as finished (completed or exhausted)."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET status = $4::session_status, total_steps = $2, total_reward = $3,
                completed_at = now(), updated_at = now()
            WHERE id = $1::uuid
            """,
            session_id, total_steps, total_reward, status,
        )


async def fail_session(session_id: str) -> None:
    """Mark a session as failed."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET status = 'failed', updated_at = now()
            WHERE id = $1::uuid
            """,
            session_id,
        )


async def cancel_session(session_id: str, total_steps: int, total_reward: float) -> None:
    """Mark a session as cancelled (client disconnect / stop)."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET status = 'cancelled', total_steps = $2, total_reward = $3,
                updated_at = now()
            WHERE id = $1::uuid
            """,
            session_id, total_steps, total_reward,
        )


async def soft_delete_session(session_id: str) -> bool:
    """Soft-delete a session. Returns True if a row was updated."""
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET status = 'deleted', updated_at = now()
            WHERE id = $1::uuid AND status != 'deleted'
            """,
            session_id,
        )
        return result.endswith("1")  # "UPDATE 1"


# ---------------------------------------------------------------------------
# Frame queries
# ---------------------------------------------------------------------------

async def insert_frame(
    session_id: str,
    step: int,
    thought: str,
    action: str,
    reward: float,
    done: bool,
) -> None:
    """Insert a single frame row."""
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO frames (session_id, step, thought, action, reward, done)
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            """,
            session_id, step, thought, action, reward, done,
        )


async def get_frames(
    session_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return paginated frames for a session and total count."""
    async with _pool.acquire() as conn:
        count_row = await conn.fetchrow(
            "SELECT count(*) AS cnt FROM frames WHERE session_id = $1::uuid",
            session_id,
        )
        total = count_row["cnt"]

        rows = await conn.fetch(
            """
            SELECT id, session_id, step, thought, action, reward, done, created_at
            FROM frames
            WHERE session_id = $1::uuid
            ORDER BY step
            LIMIT $2 OFFSET $3
            """,
            session_id, limit, offset,
        )
        return [_record_to_dict(r) for r in rows], total


# ---------------------------------------------------------------------------
# Checkpoint queries
# ---------------------------------------------------------------------------

async def save_checkpoint(
    session_id: str,
    step: int,
    env_state: dict,
    adapter_state: dict,
    agent_state: dict,
) -> str:
    """Upsert a checkpoint and return its UUID."""
    async with _pool.acquire() as conn:
        import json
        row = await conn.fetchrow(
            """
            INSERT INTO checkpoints (session_id, step, env_state, adapter_state, agent_state)
            VALUES ($1::uuid, $2, $3::jsonb, $4::jsonb, $5::jsonb)
            ON CONFLICT (session_id, step)
            DO UPDATE SET env_state = EXCLUDED.env_state,
                          adapter_state = EXCLUDED.adapter_state,
                          agent_state = EXCLUDED.agent_state,
                          created_at = now()
            RETURNING id
            """,
            session_id, step,
            json.dumps(env_state),
            json.dumps(adapter_state),
            json.dumps(agent_state),
        )
        return str(row["id"])


async def get_checkpoint(session_id: str, step: int) -> Optional[Dict[str, Any]]:
    """Fetch a specific checkpoint by session + step."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM checkpoints
            WHERE session_id = $1::uuid AND step = $2
            """,
            session_id, step,
        )
        return _record_to_dict(row) if row else None


async def get_latest_checkpoint(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the most recent checkpoint for a session."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM checkpoints
            WHERE session_id = $1::uuid
            ORDER BY step DESC
            LIMIT 1
            """,
            session_id,
        )
        return _record_to_dict(row) if row else None


async def list_checkpoints(session_id: str) -> List[Dict[str, Any]]:
    """Return summary list of checkpoints for a session."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, step, created_at
            FROM checkpoints
            WHERE session_id = $1::uuid
            ORDER BY step
            """,
            session_id,
        )
        return [_record_to_dict(r) for r in rows]
