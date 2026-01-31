"""
Environment setup utilities for different environment types (standard, Atari, etc.)
"""

from typing import Any
import gymnasium as gym
import numpy as np
from ale_py.vector_env import AtariVectorEnv


def _env_name_to_rom_id(env_name: str) -> str:
    """
    Convert gymnasium-style env name to ALE ROM id.
    e.g., "Breakout-v5" -> "breakout", "ALE/Breakout-v5" -> "breakout"
    """
    name = env_name.replace("ALE/", "")
    name = name.split("-")[0]
    return name.lower()


def make_atari_vec_envs(
    env_name: str,
    num_envs: int,
    stack_size: int = 4,
    max_episode_steps: int | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
    vectorization_mode: str = "async",
) -> AtariVectorEnv:
    """
    Create vectorized Atari environments using ale_py's AtariVectorEnv.
    
    return: AtariVectorEnv with standard Atari preprocessing.
    """
    rom_id = _env_name_to_rom_id(env_name)
    
    kwargs = dict(
        game=rom_id,
        num_envs=num_envs,
        frameskip=4,
        grayscale=True,
        stack_num=stack_size,
        img_height=84,
        img_width=84,
        maxpool=True,
        reward_clipping=False,
        noop_max=30,
        use_fire_reset=True,
        episodic_life=False,
        full_action_space=False,
    )
    
    if max_episode_steps is not None:
        kwargs["max_num_frames_per_episode"] = max_episode_steps * 4  # account for frameskip
    
    envs = AtariVectorEnv(**kwargs)
    if render_mode is not None and hasattr(envs, "render_mode"):
        envs.render_mode = render_mode
    
    return envs


def make_standard_vec_envs(
    env_name: str,
    num_envs: int,
    max_episode_steps: int | None = None,
    env_args: dict[str, Any] | None = None,
    render_mode: str | None = None,
    vectorization_mode: str = "async",
) -> gym.vector.VectorEnv:
    """
    Create standard vectorized environments without special preprocessing.
    
    return: VectorEnv
    """
    kwargs = dict(env_args or {})
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    envs = gym.make_vec(
        env_name,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    return envs


def make_vec_envs(env_cfg: Any, vectorization_mode: str = "async", render_mode: str | None = None) -> gym.vector.VectorEnv:
    """
    Factory function to create vectorized environments.
    
    return: VectorEnv (Atari with preprocessing if is_atari=True, else standard)
    """
    env_args = env_cfg.env_args
    render_mode = render_mode if render_mode is not None else env_args.get("render_mode", None)
    seed = env_args.get("seed", None)

    if env_cfg.is_atari:
        return make_atari_vec_envs(
            env_name=env_cfg.name,
            num_envs=env_cfg.num_envs,
            stack_size=env_cfg.stack_stamples,
            max_episode_steps=env_cfg.max_episode_steps,
            seed=seed,
            render_mode=render_mode,
            vectorization_mode=vectorization_mode,
        )
    else:
        return make_standard_vec_envs(
            env_name=env_cfg.name,
            num_envs=env_cfg.num_envs,
            max_episode_steps=env_cfg.max_episode_steps,
            env_args=env_args,
            render_mode=render_mode,
            vectorization_mode=vectorization_mode,
        )
