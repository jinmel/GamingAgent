"""FastAPI application with WebSocket endpoint for game streaming."""

import asyncio
import json
import logging
import queue
import traceback
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .config import (
    SUPPORTED_GAMES,
    GAME_REGISTRY,
    SessionConfig,
    MAX_STEPS_LIMIT,
    MODEL_NAME_PATTERN,
    MAX_MODEL_NAME_LENGTH,
)
from .game_session import GameSession

logger = logging.getLogger("streaming_service")

app = FastAPI(
    title="GamingAgent Streaming Service",
    description="WebSocket-based streaming service for watching LLM agents play games.",
    version="0.1.0",
)

# Timeout (seconds) for the client to send the initial config message.
CONFIG_RECEIVE_TIMEOUT = 30.0

# Maximum number of concurrent game sessions.
_MAX_CONCURRENT_SESSIONS = 3
_session_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SESSIONS)


# ── Health / info endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


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
            "2_send_config": "Send a JSON config message within 30 seconds.",
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
                "description": f"Maximum game steps. Range: 1–{MAX_STEPS_LIMIT}.",
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
                "description": "Max trajectory entries in agent memory. Range: 1–100.",
            },
            "seed": {
                "type": "integer|null",
                "required": False,
                "default": None,
                "description": "Random seed for environment reproducibility.",
            },
        },
        "server_messages": {
            "session_start": {
                "type": "session_start",
                "game": "string — game name",
                "model": "string — model name",
            },
            "frame": {
                "type": "frame",
                "step": "integer — 0-indexed step number",
                "image": "string — base64-encoded PNG screenshot",
                "thought": "string — LLM reasoning for the action",
                "action": "string — action taken (e.g. 'up', 'down', 'left', 'right')",
                "reward": "float — reward received this step",
                "done": "boolean — true if game has ended",
            },
            "session_end": {
                "type": "session_end",
            },
            "error": {
                "type": "error",
                "message": "string — error description",
            },
        },
        "limits": {
            "max_concurrent_sessions": _MAX_CONCURRENT_SESSIONS,
            "config_timeout_seconds": CONFIG_RECEIVE_TIMEOUT,
            "max_steps_limit": MAX_STEPS_LIMIT,
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/play")
async def ws_play(ws: WebSocket):
    """Stream a game session over a WebSocket connection.

    Protocol
    --------
    1. Client connects.
    2. Client sends a JSON config message (within 30 s)::

           {
               "game_name": "2048",
               "model_name": "claude-3-7-sonnet-latest",
               "prompt": "...",
               "max_steps": 200,
               "observation_mode": "vision",
               "harness": false,
               "max_memory": 10,
               "seed": null
           }

    3. Server streams back JSON frame messages::

           { "type": "frame", "step": int, "image": str,
             "thought": str, "action": str, "reward": float, "done": bool }

    4. Server sends ``{"type": "session_end"}`` when done.

    Client can send ``{"type": "stop"}`` at any time to abort the session.
    """
    await ws.accept()

    # ── Enforce concurrency limit ────────────────────────────────────────
    if _session_semaphore.locked():
        await _safe_send(ws, {
            "type": "error",
            "message": f"Server at capacity ({_MAX_CONCURRENT_SESSIONS} concurrent sessions).",
        })
        await ws.close(code=1013)
        return

    session: Optional[GameSession] = None
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

            cfg, errors = _validate_config(payload)
            if errors:
                await _safe_send(ws, {"type": "error", "message": " ".join(errors)})
                return

            await _safe_send(ws, {
                "type": "session_start",
                "game": cfg.game_name,
                "model": cfg.model_name,
            })

            # ── 2. Run game loop in a thread ─────────────────────────────
            # Use a thread-safe queue to decouple the synchronous game
            # generator (running in a worker thread) from the async
            # WebSocket sender. This avoids sharing the generator across
            # threads and prevents the StopIteration-through-await bug.
            session = GameSession(cfg)
            frame_queue: queue.Queue = queue.Queue(maxsize=2)
            loop = asyncio.get_running_loop()
            cancelled = False

            def _produce_frames():
                """Run the entire sync generator in a single thread,
                pushing each frame into the queue. A None sentinel
                signals the end of the stream."""
                try:
                    for frame in session.run():
                        frame_queue.put(frame)
                except Exception as exc:
                    frame_queue.put(exc)
                finally:
                    frame_queue.put(None)  # sentinel

            # Start the producer in a background thread.
            producer_future = loop.run_in_executor(None, _produce_frames)

            # Listen for client stop/disconnect in a concurrent task
            # instead of polling with a 10ms timeout on every step.
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

            # Consume frames from the queue and send over WebSocket.
            try:
                while True:
                    item = await loop.run_in_executor(None, frame_queue.get)

                    # Sentinel: generator finished.
                    if item is None:
                        break

                    # Exception from the producer thread.
                    if isinstance(item, Exception):
                        raise item

                    # Normal frame dict.
                    if not await _safe_send(ws, item):
                        cancelled = True
                        break

                    if item.get("done"):
                        break

                    if stop_event.is_set():
                        cancelled = True
                        break
            finally:
                listener_task.cancel()

            # Wait for the producer thread to finish.
            await producer_future

            # ── 3. Signal completion ─────────────────────────────────────
            if not cancelled:
                await _safe_send(ws, {"type": "session_end"})

        except WebSocketDisconnect:
            pass  # Client disconnected — clean up silently.
        except Exception as exc:
            logger.error("[ws_play] Error: %s\n%s", exc, traceback.format_exc())
            await _safe_send(ws, {
                "type": "error",
                "message": "Internal server error. Check server logs.",
            })
        finally:
            if session is not None:
                session.close()
            try:
                await ws.close()
            except Exception:
                pass
