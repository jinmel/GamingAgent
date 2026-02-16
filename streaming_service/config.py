"""Configuration and game registry for the streaming service."""

import enum
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional


class SessionStatus(str, enum.Enum):
    """Lifecycle status of a game session.

    running    – game loop is actively producing frames
    completed  – game ended naturally (env returned done=True)
    exhausted  – hit max_steps without the env signalling done
    cancelled  – client sent {"type":"stop"} or disconnected
    failed     – unhandled server-side exception
    deleted    – soft-deleted via the REST API
    """
    RUNNING = "running"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DELETED = "deleted"

# Database
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/gamingagent"
)

# How often (in steps) to auto-save a checkpoint during a game session.
CHECKPOINT_INTERVAL: int = int(os.environ.get("CHECKPOINT_INTERVAL", "50"))

# Map user-facing game names to internal names and config directory names
GAME_REGISTRY: Dict[str, Dict[str, str]] = {
    "2048": {
        "internal_name": "twenty_forty_eight",
        "config_dir": "custom_01_2048",
    },
    "sokoban": {
        "internal_name": "sokoban",
        "config_dir": "custom_02_sokoban",
    },
    "pokemon": {
        "internal_name": "pokemon_red",
        "config_dir": "custom_06_pokemon_red",
    },
}

SUPPORTED_GAMES = list(GAME_REGISTRY.keys())

# Base paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_ROOT = os.path.join(PROJECT_ROOT, "gamingagent", "configs")
ENVS_ROOT = os.path.join(PROJECT_ROOT, "gamingagent", "envs")

# Limits
MAX_STEPS_LIMIT = 500
MAX_MODEL_NAME_LENGTH = 128
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_./:@-]+$")


@dataclass
class SessionConfig:
    """Configuration for a single game session, derived from client request."""

    game_name: str  # User-facing name: "2048", "sokoban", "pokemon"
    model_name: str
    prompt: str = ""
    max_steps: int = 200
    observation_mode: str = "vision"
    harness: bool = False
    max_memory: int = 10
    seed: Optional[int] = None

    def __post_init__(self):
        self.max_steps = max(1, min(self.max_steps, MAX_STEPS_LIMIT))

    @property
    def internal_game_name(self) -> str:
        return GAME_REGISTRY[self.game_name]["internal_name"]

    @property
    def config_dir_name(self) -> str:
        return GAME_REGISTRY[self.game_name]["config_dir"]

    @property
    def agent_prompts_path(self) -> Optional[str]:
        path = os.path.join(CONFIGS_ROOT, self.config_dir_name, "module_prompts.json")
        return path if os.path.isfile(path) else None

    @property
    def env_config_path(self) -> str:
        return os.path.join(ENVS_ROOT, self.config_dir_name, "game_env_config.json")
