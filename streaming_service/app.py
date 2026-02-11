"""FastAPI application with WebSocket endpoint for game streaming."""

import asyncio
import json
import traceback
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .config import SUPPORTED_GAMES, GAME_REGISTRY, SessionConfig
from .game_session import GameSession

app = FastAPI(
    title="GamingAgent Streaming Service",
    description="WebSocket-based streaming service for watching LLM agents play games.",
    version="0.1.0",
)

# Timeout (seconds) for the client to send the initial config message.
CONFIG_RECEIVE_TIMEOUT = 30.0


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

    max_steps = payload.get("max_steps", 200)
    if not isinstance(max_steps, int) or max_steps < 1:
        errors.append("'max_steps' must be a positive integer.")
        max_steps = 200

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

    session: Optional[GameSession] = None
    try:
        # ── 1. Receive config (with timeout) ─────────────────────────────
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

        # ── 2. Run game loop in a thread ─────────────────────────────────
        session = GameSession(cfg)
        loop = asyncio.get_running_loop()
        gen = session.run()
        cancelled = False

        def _next_frame():
            return next(gen)

        while True:
            # Run the synchronous generator step in a thread pool.
            try:
                frame = await loop.run_in_executor(None, _next_frame)
            except StopIteration:
                break

            # Send the frame; bail if the client disconnected.
            if not await _safe_send(ws, frame):
                cancelled = True
                break

            if frame.get("done"):
                break

            # Check for a client "stop" message (non-blocking).
            try:
                client_msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                try:
                    client_payload = json.loads(client_msg)
                    if client_payload.get("type") == "stop":
                        cancelled = True
                        break
                except json.JSONDecodeError:
                    pass  # Ignore malformed client messages mid-session.
            except asyncio.TimeoutError:
                pass  # No client message — continue game loop.

        # ── 3. Signal completion ─────────────────────────────────────────
        if not cancelled:
            await _safe_send(ws, {"type": "session_end"})

    except WebSocketDisconnect:
        pass  # Client disconnected — clean up silently.
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ws_play] Error: {exc}\n{tb}")
        await _safe_send(ws, {"type": "error", "message": str(exc)})
    finally:
        if session is not None:
            session.close()
        try:
            await ws.close()
        except Exception:
            pass
