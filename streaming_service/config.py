"""Configuration and game registry for the streaming service."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

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

    @property
    def internal_game_name(self) -> str:
        return GAME_REGISTRY[self.game_name]["internal_name"]

    @property
    def config_dir_name(self) -> str:
        return GAME_REGISTRY[self.game_name]["config_dir"]

    @property
    def agent_prompts_path(self) -> str:
        path = os.path.join(CONFIGS_ROOT, self.config_dir_name, "module_prompts.json")
        return path if os.path.isfile(path) else None

    @property
    def env_config_path(self) -> str:
        return os.path.join(ENVS_ROOT, self.config_dir_name, "game_env_config.json")
