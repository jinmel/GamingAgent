"""Game session management — wraps existing agent + environment pipeline."""

import base64
import datetime
import json
import os
import time
import uuid
from collections import deque
from collections.abc import Iterator
from typing import Any, Dict, Optional

from .config import SessionConfig, PROJECT_ROOT


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

    def __init__(
        self,
        cfg: SessionConfig,
        resume_from: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.resume_from = resume_from

        # Use microseconds + short UUID to avoid cache directory collisions
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:6]
        model_short = cfg.model_name[:15].replace("-", "_")
        self.cache_dir = os.path.join(
            PROJECT_ROOT, "cache", cfg.internal_game_name,
            model_short, f"{ts}_{suffix}",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.env = None
        self.agent = None

        # Accumulated totals for DB updates.
        self.total_steps = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def get_checkpoint_data(self) -> Optional[Dict[str, Any]]:
        """Return a JSON-serialisable dict capturing full game state.

        Returns None if the env/agent are not yet initialised.
        """
        if self.env is None or self.agent is None:
            return None

        # Environment state
        env_state = self.env.get_state()

        # Adapter state
        adapter_state = self.env.adapter.get_state()

        # Agent trajectory — serialise the deque entries as a list of strings
        # and include the reflection text.
        trajectory = self._current_obs
        agent_state: Dict[str, Any] = {"trajectory_entries": [], "reflection": None}
        if trajectory is not None and hasattr(trajectory, "game_trajectory"):
            gt = trajectory.game_trajectory
            agent_state["trajectory_entries"] = list(gt.trajectory)
            agent_state["trajectory_max_length"] = gt.max_length
            agent_state["trajectory_background"] = gt.background
            agent_state["trajectory_need_background"] = gt.need_background
            agent_state["reflection"] = trajectory.reflection

        return {
            "env_state": env_state,
            "adapter_state": adapter_state,
            "agent_state": agent_state,
        }

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore env, adapter, and agent trajectory from *checkpoint*.

        Must be called **after** ``_create_environment`` / ``_create_agent``
        but **before** the game loop begins.
        """
        # Environment
        self.env.set_state(checkpoint["env_state"])

        # Adapter
        self.env.adapter.set_state(checkpoint["adapter_state"])

        # Agent trajectory
        agent_state = checkpoint.get("agent_state", {})
        entries = agent_state.get("trajectory_entries", [])
        max_len = agent_state.get("trajectory_max_length", self.cfg.max_memory)

        from gamingagent.modules.core_module import GameTrajectory

        gt = GameTrajectory(
            max_length=max_len,
            need_background=agent_state.get("trajectory_need_background", False),
        )
        bg = agent_state.get("trajectory_background")
        if bg is not None:
            gt.set_background(bg)
        for entry in entries:
            gt.add(entry)

        # Stash so that ``run()`` can inject it into the first observation.
        self._restored_trajectory = gt
        self._restored_reflection = agent_state.get("reflection")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Iterator[Dict[str, Any]]:
        """Synchronous generator that yields one dict per game step.

        Each dict matches the wire format:
            { "type": "frame", "step": int, "image": base64_png,
              "thought": str, "action": str, "reward": float, "done": bool }
        """
        self.env = _create_environment(self.cfg, self.cache_dir)
        self.agent = _create_agent(self.cfg, self.cache_dir)

        # Determine starting step.
        start_step = 0
        self._current_obs = None
        self._restored_trajectory = None
        self._restored_reflection = None

        if self.resume_from is not None:
            # Resume path: restore state instead of resetting.
            start_step = self.resume_from.get("step", 0)

            # We still need to call reset to wire up adapter internals,
            # but immediately overwrite the state.
            episode_id = 1
            obs, info = self.env.reset(
                max_memory=self.cfg.max_memory,
                seed=self.cfg.seed,
                episode_id=episode_id,
            )

            self.restore_from_checkpoint(self.resume_from)

            # Rebuild the observation image for the restored state.
            obs = self.env.adapter.create_agent_observation(
                img_path=self._render_current_state(),
                max_memory=self.cfg.max_memory,
            )

            # Re-attach the restored trajectory.
            if self._restored_trajectory is not None:
                obs.game_trajectory = self._restored_trajectory
            if self._restored_reflection is not None:
                obs.reflection = self._restored_reflection

            self._current_obs = obs

            # Yield a "restored" frame at step 0 so the client sees the state.
            yield {
                "type": "frame",
                "step": start_step,
                "image": _image_to_base64(obs.get_img_path()),
                "thought": "(resumed from checkpoint)",
                "action": "",
                "reward": 0.0,
                "done": False,
            }
        else:
            # Normal start.
            episode_id = 1
            obs, info = self.env.reset(
                max_memory=self.cfg.max_memory,
                seed=self.cfg.seed,
                episode_id=episode_id,
            )

            self._current_obs = obs

            yield {
                "type": "frame",
                "step": 0,
                "image": _image_to_base64(obs.get_img_path()),
                "thought": "",
                "action": "",
                "reward": 0.0,
                "done": False,
            }

        for step in range(start_step + 1, self.cfg.max_steps + 1):
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

            self._current_obs = obs

            done = terminated or truncated
            self.total_steps = step
            self.total_reward += float(reward)

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_current_state(self) -> Optional[str]:
        """Render the current env state to an image and return the path."""
        try:
            frame = self.env.render()
            if frame is not None:
                return self.env.adapter.save_frame_and_get_path(frame)
        except Exception:
            pass
        return None

    def close(self):
        """Release environment resources."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
