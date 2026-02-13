"""FastAPI application with WebSocket endpoint for game streaming."""

import asyncio
import json
import logging
import os
import queue
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from . import db
from .config import (
    CHECKPOINT_INTERVAL,
    DATABASE_URL,
    GAME_REGISTRY,
    MAX_MODEL_NAME_LENGTH,
    MAX_STEPS_LIMIT,
    MODEL_NAME_PATTERN,
    SessionConfig,
    SessionStatus,
    SUPPORTED_GAMES,
)
from .game_session import GameSession

logger = logging.getLogger("streaming_service")

# Timeout (seconds) for the client to send the initial config message.
CONFIG_RECEIVE_TIMEOUT = 30.0

# Maximum number of concurrent game sessions.
_MAX_CONCURRENT_SESSIONS = 3
_session_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SESSIONS)


# ---------------------------------------------------------------------------
# Lifespan — pool init / teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the DB pool on startup and close it on shutdown."""
    if not os.environ.get("STREAMING_NO_DB"):
        await db.init_pool(DATABASE_URL)
    yield
    await db.close_pool()


app = FastAPI(
    title="GamingAgent Streaming Service",
    description="WebSocket-based streaming service for watching LLM agents play games.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "db": db.is_available()}


@app.get("/games")
async def list_games():
    """Return the list of supported games and their accepted config."""
    return {"games": SUPPORTED_GAMES}


@app.get("/ws-docs")
async def ws_docs():
    """WebSocket protocol documentation for /ws/play (not covered by OpenAPI)."""
    return {
        "endpoint": "ws://<host>/ws/play",
        "protocol": {
            "1_connect": "Open a WebSocket connection to /ws/play.",
            "2_send_config": (
                "Send a JSON config message within 30 seconds. "
                "To resume from a previous session, send "
                '{\"resume_from\": \"<parent-session-id>\"} instead of a full config.'
            ),
            "3_receive_frames": "Server streams JSON frame messages until done.",
            "4_session_end": "Server sends {\"type\": \"session_end\"} when the game finishes.",
            "5_client_stop": "Client may send {\"type\": \"stop\"} at any time to abort.",
        },
        "config_message": {
            "game_name": {
                "type": "string",
                "required": True,
                "enum": SUPPORTED_GAMES,
                "description": "Game to play.",
            },
            "model_name": {
                "type": "string",
                "required": True,
                "description": f"LLM model identifier. Max {MAX_MODEL_NAME_LENGTH} chars, "
                               f"allowed chars: a-z A-Z 0-9 _ . / : @ -",
            },
            "prompt": {
                "type": "string",
                "required": False,
                "default": "",
                "description": "Custom prompt or instructions for the agent.",
            },
            "max_steps": {
                "type": "integer",
                "required": False,
                "default": 200,
                "description": f"Maximum game steps. Range: 1-{MAX_STEPS_LIMIT}.",
            },
            "observation_mode": {
                "type": "string",
                "required": False,
                "default": "vision",
                "enum": ["vision", "text", "both"],
                "description": "How the agent observes the game state.",
            },
            "harness": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Enable perception-memory-reasoning pipeline.",
            },
            "max_memory": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Max trajectory entries in agent memory. Range: 1-100.",
            },
            "seed": {
                "type": "integer|null",
                "required": False,
                "default": None,
                "description": "Random seed for environment reproducibility.",
            },
            "resume_from": {
                "type": "string|null",
                "required": False,
                "default": None,
                "description": (
                    "Parent session ID to resume from its latest checkpoint. "
                    "When set, all other config fields are ignored — they are "
                    "inherited from the parent session."
                ),
            },
        },
        "session_statuses": {
            s.value: s.__doc__ or s.name for s in SessionStatus
        },
        "server_messages": {
            "session_start": {
                "type": "session_start",
                "game": "string - game name",
                "model": "string - model name",
                "session_id": "string - UUID of the session (if DB available)",
            },
            "frame": {
                "type": "frame",
                "step": "integer - 0-indexed step number",
                "image": "string - base64-encoded PNG screenshot",
                "thought": "string - LLM reasoning for the action",
                "action": "string - action taken (e.g. 'up', 'down', 'left', 'right')",
                "reward": "float - reward received this step",
                "done": "boolean - true if game has ended",
            },
            "session_end": {"type": "session_end"},
            "error": {"type": "error", "message": "string - error description"},
        },
        "limits": {
            "max_concurrent_sessions": _MAX_CONCURRENT_SESSIONS,
            "config_timeout_seconds": CONFIG_RECEIVE_TIMEOUT,
            "max_steps_limit": MAX_STEPS_LIMIT,
        },
    }


# ---------------------------------------------------------------------------
# CRUD REST endpoints
# ---------------------------------------------------------------------------

@app.get("/sessions")
async def api_list_sessions(
    game_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List sessions (paginated, filterable)."""
    if not db.is_available():
        return JSONResponse({"error": "Database not available"}, status_code=503)
    rows, total = await db.list_sessions(
        game_name=game_name,
        model_name=model_name,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"sessions": rows, "total": total, "limit": limit, "offset": offset}


@app.get("/sessions/{session_id}")
async def api_get_session(session_id: str):
    """Get a single session by ID."""
    if not db.is_available():
        return JSONResponse({"error": "Database not available"}, status_code=503)
    row = await db.get_session(session_id)
    if row is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return row


@app.get("/sessions/{session_id}/frames")
async def api_get_frames(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get frames for a session (paginated)."""
    if not db.is_available():
        return JSONResponse({"error": "Database not available"}, status_code=503)
    rows, total = await db.get_frames(session_id, limit=limit, offset=offset)
    return {"frames": rows, "total": total, "limit": limit, "offset": offset}


@app.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """Soft-delete a session."""
    if not db.is_available():
        return JSONResponse({"error": "Database not available"}, status_code=503)
    ok = await db.soft_delete_session(session_id)
    if not ok:
        return JSONResponse({"error": "Session not found or already deleted"}, status_code=404)
    return {"deleted": True}


@app.get("/sessions/{session_id}/checkpoints")
async def api_list_checkpoints(session_id: str):
    """List checkpoints for a session."""
    if not db.is_available():
        return JSONResponse({"error": "Database not available"}, status_code=503)
    rows = await db.list_checkpoints(session_id)
    return {"checkpoints": rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_config(payload: dict) -> tuple[Optional[SessionConfig], list[str]]:
    """Validate a client config payload and return (config, errors)."""
    errors: list[str] = []

    game_name = payload.get("game_name")
    model_name = payload.get("model_name")

    if not game_name:
        errors.append("'game_name' is required.")
    elif game_name not in GAME_REGISTRY:
        errors.append(f"Unsupported game '{game_name}'. Supported: {SUPPORTED_GAMES}")

    if not model_name:
        errors.append("'model_name' is required.")
    elif not isinstance(model_name, str):
        errors.append("'model_name' must be a string.")
    elif len(model_name) > MAX_MODEL_NAME_LENGTH:
        errors.append(f"'model_name' exceeds max length of {MAX_MODEL_NAME_LENGTH}.")
    elif not MODEL_NAME_PATTERN.match(model_name):
        errors.append("'model_name' contains invalid characters.")

    max_steps = payload.get("max_steps", 200)
    if not isinstance(max_steps, int) or max_steps < 1:
        errors.append("'max_steps' must be a positive integer.")
        max_steps = 200
    elif max_steps > MAX_STEPS_LIMIT:
        errors.append(f"'max_steps' exceeds limit of {MAX_STEPS_LIMIT}.")
        max_steps = MAX_STEPS_LIMIT

    obs_mode = payload.get("observation_mode", "vision")
    if obs_mode not in ("vision", "text", "both"):
        errors.append(f"'observation_mode' must be one of: vision, text, both. Got '{obs_mode}'.")
        obs_mode = "vision"

    max_memory = payload.get("max_memory", 10)
    if not isinstance(max_memory, int) or max_memory < 1 or max_memory > 100:
        errors.append("'max_memory' must be an integer between 1 and 100.")
        max_memory = 10

    seed = payload.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append("'seed' must be an integer or null.")
        seed = None

    if errors:
        return None, errors

    return SessionConfig(
        game_name=game_name,
        model_name=model_name,
        prompt=payload.get("prompt", ""),
        max_steps=max_steps,
        observation_mode=obs_mode,
        harness=bool(payload.get("harness", False)),
        max_memory=max_memory,
        seed=seed,
    ), []


async def _safe_send(ws: WebSocket, data: dict) -> bool:
    """Send JSON, returning False if the connection is already closed."""
    if ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DB persistence helpers (best-effort — never break the game loop)
# ---------------------------------------------------------------------------

async def _db_create_session(cfg: SessionConfig, cache_dir: str) -> Optional[str]:
    if not db.is_available():
        return None
    try:
        return await db.create_session(
            game_name=cfg.game_name,
            model_name=cfg.model_name,
            prompt=cfg.prompt,
            max_steps=cfg.max_steps,
            observation_mode=cfg.observation_mode,
            harness=cfg.harness,
            max_memory=cfg.max_memory,
            seed=cfg.seed,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        logger.warning("DB create_session failed: %s", exc)
        return None


async def _db_insert_frame(
    session_id: Optional[str], step: int, thought: str,
    action: str, reward: float, done: bool,
) -> None:
    if session_id is None or not db.is_available():
        return
    try:
        await db.insert_frame(session_id, step, thought, action, reward, done)
    except Exception as exc:
        logger.warning("DB insert_frame failed (step %d): %s", step, exc)


async def _db_save_checkpoint(
    session_id: Optional[str], session: GameSession, step: int,
) -> None:
    if session_id is None or not db.is_available():
        return
    try:
        cp = session.get_checkpoint_data()
        if cp is not None:
            await db.save_checkpoint(
                session_id, step,
                cp["env_state"], cp["adapter_state"], cp["agent_state"],
            )
    except Exception as exc:
        logger.warning("DB save_checkpoint failed (step %d): %s", step, exc)


async def _db_finish_session(
    session_id: Optional[str],
    status: SessionStatus,
    session: Optional[GameSession],
) -> None:
    """Update DB session status.  Never raises."""
    if session_id is None or not db.is_available():
        return
    steps = session.total_steps if session else 0
    reward = session.total_reward if session else 0.0
    try:
        if status == SessionStatus.FAILED:
            await db.fail_session(session_id)
        else:
            await db.finish_session(session_id, steps, reward, status.value)
    except Exception as exc:
        logger.warning("DB finish_session (%s) failed: %s", status.value, exc)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/play")
async def ws_play(ws: WebSocket):
    """Stream a game session over a WebSocket connection.

    Protocol
    --------
    1. Client connects.
    2. Client sends a JSON config message (within 30 s).

       **Fresh game**::

           {
               "game_name": "2048",
               "model_name": "claude-3-7-sonnet-latest",
               ...
           }

       **Resume from checkpoint**::

           { "resume_from": "<parent-session-id>" }

    3. Server streams back JSON frame messages.
    4. Server sends ``{"type": "session_end"}`` when done.

    Client can send ``{"type": "stop"}`` at any time to abort the session.
    """
    await ws.accept()

    # Enforce concurrency limit.
    if _session_semaphore.locked():
        await _safe_send(ws, {
            "type": "error",
            "message": f"Server at capacity ({_MAX_CONCURRENT_SESSIONS} concurrent sessions).",
        })
        await ws.close(code=1013)
        return

    session: Optional[GameSession] = None
    db_session_id: Optional[str] = None
    final_status = SessionStatus.FAILED  # pessimistic default

    async with _session_semaphore:
        try:
            # ── 1. Receive config (with timeout) ─────────────────────────
            try:
                raw = await asyncio.wait_for(
                    ws.receive_text(), timeout=CONFIG_RECEIVE_TIMEOUT
                )
            except asyncio.TimeoutError:
                await _safe_send(ws, {
                    "type": "error",
                    "message": f"No config received within {CONFIG_RECEIVE_TIMEOUT}s.",
                })
                return

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _safe_send(ws, {"type": "error", "message": "Invalid JSON."})
                return

            # ── Determine fresh start vs resume ──────────────────────────
            resume_from_id = payload.get("resume_from")
            resume_from = None
            cfg = None

            if resume_from_id:
                # Resume: look up parent session + latest checkpoint in DB.
                if not db.is_available():
                    await _safe_send(ws, {
                        "type": "error",
                        "message": "Database not available — cannot resume.",
                    })
                    return

                parent = await db.get_session(resume_from_id)
                if parent is None:
                    await _safe_send(ws, {
                        "type": "error",
                        "message": f"Session '{resume_from_id}' not found.",
                    })
                    return

                cp = await db.get_latest_checkpoint(resume_from_id)
                if cp is None:
                    await _safe_send(ws, {
                        "type": "error",
                        "message": f"No checkpoint found for session '{resume_from_id}'.",
                    })
                    return

                # Build config from parent record.
                cfg = SessionConfig(
                    game_name=parent["game_name"],
                    model_name=parent["model_name"],
                    prompt=parent.get("prompt", ""),
                    max_steps=parent.get("max_steps", 200),
                    observation_mode=parent.get("observation_mode", "vision"),
                    harness=parent.get("harness", False),
                    max_memory=parent.get("max_memory", 10),
                    seed=parent.get("seed"),
                )

                # Parse JSONB checkpoint fields (asyncpg may return dicts directly).
                env_state = cp["env_state"]
                adapter_state = cp["adapter_state"]
                agent_state = cp["agent_state"]
                if isinstance(env_state, str):
                    env_state = json.loads(env_state)
                if isinstance(adapter_state, str):
                    adapter_state = json.loads(adapter_state)
                if isinstance(agent_state, str):
                    agent_state = json.loads(agent_state)

                resume_from = {
                    "step": cp["step"],
                    "env_state": env_state,
                    "adapter_state": adapter_state,
                    "agent_state": agent_state,
                }
            else:
                # Fresh session.
                cfg, errors = _validate_config(payload)
                if errors:
                    await _safe_send(ws, {"type": "error", "message": " ".join(errors)})
                    return

            # ── 2. Create GameSession + DB record ────────────────────────
            session = GameSession(cfg, resume_from=resume_from)

            if resume_from_id and db.is_available():
                # Create a child session linked to the parent.
                try:
                    db_session_id = await db.create_resumed_session(
                        parent_session_id=resume_from_id,
                        resumed_from_step=resume_from["step"],
                        game_name=cfg.game_name,
                        model_name=cfg.model_name,
                        prompt=cfg.prompt,
                        max_steps=cfg.max_steps,
                        observation_mode=cfg.observation_mode,
                        harness=cfg.harness,
                        max_memory=cfg.max_memory,
                        seed=cfg.seed,
                        cache_dir=session.cache_dir,
                    )
                except Exception as exc:
                    logger.warning("DB create_resumed_session failed: %s", exc)
            else:
                db_session_id = await _db_create_session(cfg, session.cache_dir)

            start_msg = {
                "type": "session_start",
                "game": cfg.game_name,
                "model": cfg.model_name,
            }
            if db_session_id:
                start_msg["session_id"] = db_session_id
            await _safe_send(ws, start_msg)

            # ── 3. Run game loop in a thread ─────────────────────────────
            frame_queue: queue.Queue = queue.Queue(maxsize=2)
            loop = asyncio.get_running_loop()
            cancelled = False
            game_done = False  # True when env signals done

            def _produce_frames():
                try:
                    for frame in session.run():
                        frame_queue.put(frame)
                except Exception as exc:
                    frame_queue.put(exc)
                finally:
                    frame_queue.put(None)

            producer_future = loop.run_in_executor(None, _produce_frames)

            stop_event = asyncio.Event()

            async def _listen_for_stop():
                try:
                    while True:
                        raw = await ws.receive_text()
                        try:
                            msg = json.loads(raw)
                            if msg.get("type") == "stop":
                                stop_event.set()
                                return
                        except json.JSONDecodeError:
                            pass
                except (WebSocketDisconnect, Exception):
                    stop_event.set()

            listener_task = asyncio.create_task(_listen_for_stop())

            try:
                while True:
                    item = await loop.run_in_executor(None, frame_queue.get)

                    if item is None:
                        break

                    if isinstance(item, Exception):
                        raise item

                    # Persist frame to DB before sending over WS.
                    step = item.get("step", 0)
                    await _db_insert_frame(
                        db_session_id, step,
                        item.get("thought", ""),
                        item.get("action", ""),
                        item.get("reward", 0.0),
                        item.get("done", False),
                    )

                    # Auto-checkpoint every CHECKPOINT_INTERVAL steps.
                    if (
                        step > 0
                        and step % CHECKPOINT_INTERVAL == 0
                        and not item.get("done", False)
                    ):
                        await _db_save_checkpoint(db_session_id, session, step)

                    if not await _safe_send(ws, item):
                        cancelled = True
                        break

                    if item.get("done"):
                        game_done = True
                        break

                    if stop_event.is_set():
                        cancelled = True
                        break
            finally:
                listener_task.cancel()

            await producer_future

            # ── 4. Signal completion & decide final status ───────────────
            if cancelled:
                final_status = SessionStatus.CANCELLED
            elif game_done:
                final_status = SessionStatus.COMPLETED
            else:
                final_status = SessionStatus.EXHAUSTED

            if not cancelled:
                await _safe_send(ws, {"type": "session_end"})

        except WebSocketDisconnect:
            final_status = SessionStatus.CANCELLED
        except Exception as exc:
            logger.error("[ws_play] Error: %s\n%s", exc, traceback.format_exc())
            await _safe_send(ws, {
                "type": "error",
                "message": "Internal server error. Check server logs.",
            })
            final_status = SessionStatus.FAILED
        finally:
            await _db_finish_session(db_session_id, final_status, session)
            if session is not None:
                session.close()
            try:
                await ws.close()
            except Exception:
                pass
