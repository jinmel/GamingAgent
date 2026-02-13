"""Entry point: ``python -m streaming_service``."""

import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the GamingAgent streaming service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL DSN (overrides DATABASE_URL env var).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database persistence entirely.",
    )
    args = parser.parse_args()

    # Propagate CLI flags via env vars so the app module can read them at
    # import time / lifespan without needing global state.
    if args.no_db:
        os.environ["STREAMING_NO_DB"] = "1"
    elif args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    uvicorn.run(
        "streaming_service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
