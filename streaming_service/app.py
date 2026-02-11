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

    if errors:
        return None, errors

    return SessionConfig(
        game_name=game_name,
        model_name=model_name,
        prompt=payload.get("prompt", ""),
        max_steps=max_steps,
        observation_mode=obs_mode,
        harness=bool(payload.get("harness", False)),
        max_memory=int(payload.get("max_memory", 10)),
        seed=payload.get("seed"),
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

            # Consume frames from the queue and send over WebSocket.
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

                # Check for a client "stop" message (non-blocking).
                try:
                    client_msg = await asyncio.wait_for(
                        ws.receive_text(), timeout=0.01
                    )
                    try:
                        client_payload = json.loads(client_msg)
                        if client_payload.get("type") == "stop":
                            cancelled = True
                            break
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass

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
