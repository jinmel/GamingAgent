"""FastAPI application with WebSocket endpoint for game streaming."""

import asyncio
import json
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import SUPPORTED_GAMES, GAME_REGISTRY, SessionConfig
from .game_session import GameSession

app = FastAPI(
    title="GamingAgent Streaming Service",
    description="WebSocket-based streaming service for watching LLM agents play games.",
    version="0.1.0",
)


# ── Health / info endpoints ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/games")
async def list_games():
    """Return the list of supported games."""
    return {"games": SUPPORTED_GAMES}


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/play")
async def ws_play(ws: WebSocket):
    """Stream a game session over a WebSocket connection.

    Protocol
    --------
    1. Client connects.
    2. Client sends a JSON config message::

           {
               "game_name": "2048",       # required — one of SUPPORTED_GAMES
               "model_name": "claude-3-7-sonnet-latest",  # required
               "prompt": "...",            # optional
               "max_steps": 200,          # optional (default 200)
               "observation_mode": "vision",  # optional
               "harness": false,          # optional
               "max_memory": 10,          # optional
               "seed": null               # optional
           }

    3. Server streams back JSON frame messages (one per game step)::

           {
               "type": "frame",
               "step": 0,
               "image": "<base64 png>",
               "thought": "...",
               "action": "left",
               "reward": 0.0,
               "done": false
           }

    4. When the game ends (``done == true``) or an error occurs, the server
       sends a final message and closes the connection.
    """
    await ws.accept()

    session: GameSession | None = None
    try:
        # ── 1. Receive config ────────────────────────────────────────────
        raw = await ws.receive_text()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON in config message."})
            await ws.close(code=1003)
            return

        # Validate required fields
        game_name = payload.get("game_name")
        model_name = payload.get("model_name")

        errors = []
        if not game_name:
            errors.append("'game_name' is required.")
        elif game_name not in GAME_REGISTRY:
            errors.append(f"Unsupported game '{game_name}'. Supported: {SUPPORTED_GAMES}")
        if not model_name:
            errors.append("'model_name' is required.")

        if errors:
            await ws.send_json({"type": "error", "message": " ".join(errors)})
            await ws.close(code=1003)
            return

        cfg = SessionConfig(
            game_name=game_name,
            model_name=model_name,
            prompt=payload.get("prompt", ""),
            max_steps=payload.get("max_steps", 200),
            observation_mode=payload.get("observation_mode", "vision"),
            harness=payload.get("harness", False),
            max_memory=payload.get("max_memory", 10),
            seed=payload.get("seed"),
        )

        await ws.send_json({"type": "session_start", "game": game_name, "model": model_name})

        # ── 2. Run game loop in a thread (env + agent are synchronous) ───
        session = GameSession(cfg)

        # Run the synchronous generator in a thread pool to avoid blocking
        # the async event loop.
        loop = asyncio.get_running_loop()

        def _run_iter():
            """Return the next frame from the session generator."""
            return next(gen)

        gen = session.run()
        while True:
            try:
                frame = await loop.run_in_executor(None, _run_iter)
            except StopIteration:
                break

            await ws.send_json(frame)

            if frame.get("done"):
                break

        # ── 3. Signal completion ─────────────────────────────────────────
        await ws.send_json({"type": "session_end"})

    except WebSocketDisconnect:
        # Client disconnected — clean up silently.
        pass
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ws_play] Error: {exc}\n{tb}")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            session.close()
        try:
            await ws.close()
        except Exception:
            pass
