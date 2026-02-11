"""Game session management — wraps existing agent + environment pipeline."""

import base64
import datetime
import json
import os
import time
from typing import AsyncGenerator, Dict, Any, Optional

from .config import SessionConfig, PROJECT_ROOT, ENVS_ROOT


def _create_environment(cfg: SessionConfig, cache_dir: str):
    """Create a game environment using the existing env classes.

    Mirrors the logic in ``lmgame-bench/single_agent_runner.py:create_environment``
    but only for the three supported games.
    """
    game = cfg.internal_game_name
    env_config_path = cfg.env_config_path
    obs_mode = cfg.observation_mode

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Env config not found: {env_config_path}")

    with open(env_config_path, "r") as f:
        env_cfg = json.load(f)

    if game == "twenty_forty_eight":
        from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv

        init_kw = env_cfg.get("env_init_kwargs", {})
        return TwentyFortyEightEnv(
            render_mode="rgb_array",
            size=init_kw.get("size", 4),
            max_pow=init_kw.get("max_pow", 16),
            game_name_for_adapter=game,
            observation_mode_for_adapter=obs_mode,
            agent_cache_dir_for_adapter=cache_dir,
            game_specific_config_path_for_adapter=env_config_path,
            max_stuck_steps_for_adapter=env_cfg.get("max_unchanged_steps_for_termination", 10),
        )

    if game == "sokoban":
        from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv

        init_kw = env_cfg.get("env_init_kwargs", {})
        return SokobanEnv(
            render_mode="rgb_array",
            dim_room=tuple(init_kw.get("dim_room", (10, 10))),
            max_steps_episode=init_kw.get("max_steps_episode", 200),
            num_boxes=init_kw.get("num_boxes", 3),
            num_gen_steps=init_kw.get("num_gen_steps"),
            level_to_load=env_cfg.get("level_to_load"),
            tile_size_for_render=env_cfg.get("tile_size_for_render", 32),
            game_name_for_adapter=game,
            observation_mode_for_adapter=obs_mode,
            agent_cache_dir_for_adapter=cache_dir,
            game_specific_config_path_for_adapter=env_config_path,
            max_stuck_steps_for_adapter=env_cfg.get("max_unchanged_steps_for_termination", 20),
        )

    if game == "pokemon_red":
        from gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv import PokemonRedEnv

        init_kw = env_cfg.get("env_init_kwargs", {})
        return PokemonRedEnv(
            render_mode="rgb_array",
            rom_path=init_kw.get("rom_path"),
            sound=init_kw.get("sound", False),
            game_name_for_adapter=game,
            observation_mode_for_adapter=obs_mode,
            agent_cache_dir_for_adapter=cache_dir,
            game_specific_config_path_for_adapter=env_config_path,
            max_stuck_steps_for_adapter=env_cfg.get("max_unchanged_steps_for_termination", 20),
            harness=cfg.harness,
        )

    raise ValueError(f"Unsupported game: {game}")


def _create_agent(cfg: SessionConfig, cache_dir: str):
    """Create a BaseAgent configured for the given session."""
    from gamingagent.agents.base_agent import BaseAgent

    custom_modules = None
    if cfg.harness:
        from gamingagent.modules import PerceptionModule, ReasoningModule
        custom_modules = {
            "perception_module": PerceptionModule,
            "reasoning_module": ReasoningModule,
        }

    return BaseAgent(
        game_name=cfg.internal_game_name,
        model_name=cfg.model_name,
        config_path=cfg.agent_prompts_path,
        harness=cfg.harness,
        max_memory=cfg.max_memory,
        use_reflection=True,
        use_perception=True,
        custom_modules=custom_modules,
        observation_mode=cfg.observation_mode,
        cache_dir=cache_dir,
        token_limit=50000,
    )


def _image_to_base64(img_path: Optional[str]) -> str:
    """Read a PNG file and return its base64-encoded contents."""
    if not img_path or not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class GameSession:
    """Manages a single game play-through, yielding per-step frame messages."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = cfg.model_name[:15].replace("-", "_")
        self.cache_dir = os.path.join(
            PROJECT_ROOT, "cache", cfg.internal_game_name, model_short, ts
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.env = None
        self.agent = None

    def run(self) -> "Generator[Dict[str, Any], None, None]":
        """Synchronous generator that yields one dict per game step.

        Each dict matches the wire format:
            { "type": "frame", "step": int, "image": base64_png,
              "thought": str, "action": str, "reward": float, "done": bool }
        """
        self.env = _create_environment(self.cfg, self.cache_dir)
        self.agent = _create_agent(self.cfg, self.cache_dir)

        episode_id = 1
        obs, info = self.env.reset(
            max_memory=self.cfg.max_memory,
            seed=self.cfg.seed,
            episode_id=episode_id,
        )

        # Yield the initial frame (step 0) before any action
        yield {
            "type": "frame",
            "step": 0,
            "image": _image_to_base64(obs.get_img_path()),
            "thought": "",
            "action": "",
            "reward": 0.0,
            "done": False,
        }

        for step in range(1, self.cfg.max_steps + 1):
            start = time.time()
            action_dict, processed_obs = self.agent.get_action(obs)
            elapsed = time.time() - start

            action_str = ""
            thought = ""
            if action_dict:
                action_str = str(action_dict.get("action", "")).strip().lower()
                thought = action_dict.get("thought", "")

            obs, reward, terminated, truncated, info, perf = self.env.step(
                agent_action_str=action_str or "None",
                thought_process=thought,
                time_taken_s=elapsed,
            )

            # Carry over game trajectory from agent's processed observation
            obs.game_trajectory = processed_obs.game_trajectory

            done = terminated or truncated

            yield {
                "type": "frame",
                "step": step,
                "image": _image_to_base64(obs.get_img_path()),
                "thought": thought,
                "action": action_str,
                "reward": float(reward),
                "done": done,
            }

            if done:
                break

    def close(self):
        """Release environment resources."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
