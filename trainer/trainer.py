import time
import random

import torch
import numpy as np
import gymnasium as gym

from algorithms.get_algorithm import get_algorithm
from dataclass.primitives import BatchedTransition



class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._set_global_seeds(cfg.algo.seed)

        # Initialize env / algo
        start_time = time.time()
        self.train_envs = gym.make_vec(
            cfg.env.name, 
            num_envs = cfg.env.num_envs, 
            vectorization_mode="async", 
        )
        self.eval_envs = gym.make_vec(
            cfg.env.name, 
            num_envs = cfg.env.num_envs, 
            vectorization_mode="sync", 
        )
        self.algo = get_algorithm(cfg.algo.name)(cfg, self.train_envs.single_observation_space, self.train_envs.single_action_space, self.device)
        print(f"Successfully initialized env + algo in {(time.time() - start_time):.2f}s")

        # TODO: Setup for WandB
        # Start WandB --> hold on this let's just print via console for now


    def train(self):
        train_start_time, last_eval_log_time = time.time(), time.time()
        print(f"Starting training ({self.cfg.train.total_env_steps} steps).")
        num_env_steps, last_evaluation_step = 0, 0
        last_log_step, log_interval = 0, self.cfg.train.eval_interval
        env_seed = self.cfg.algo.seed

        cur_obs, _ = self.train_envs.reset(seed=env_seed)
        _, _ = self.eval_envs.reset(seed=env_seed + 10_000)
        while num_env_steps <= self.cfg.train.total_env_steps:
            cur_obs_tensor = self._to_tensor_obs(cur_obs)
            batched_actions = self.algo.act(cur_obs_tensor, eval_mode=False)
            env_actions = self._actions_to_env(batched_actions.action)
            next_obs, rewards, terminations, truncations, infos = self.train_envs.step(env_actions)
            transition = BatchedTransition(
                obs = cur_obs_tensor,
                act = batched_actions,
                reward = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1),
                next_obs = self._to_tensor_obs(next_obs),
                terminated = torch.as_tensor(terminations, device=self.device, dtype=torch.bool).unsqueeze(1),
                truncated = torch.as_tensor(truncations, device=self.device, dtype=torch.bool).unsqueeze(1),
                info = None
            )
            num_env_steps += rewards.shape[0]
            observe_results = self.algo.observe(transition)

            update_results = None
            if self.algo.ready_to_update():
                update_results = self.algo.update()
                
            # Console logging for now (until WandB is set up)
            if (num_env_steps - last_log_step) >= log_interval:
                log_parts = [f"step={num_env_steps}"]
                if observe_results is not None:
                    log_parts.append(", ".join([f"{k}={v:.4f}" for k, v in observe_results.items()]))
                if update_results is not None:
                    log_parts.append(", ".join([f"{k}={v:.4f}" for k, v in update_results.items()]))
                print("[train] " + " | ".join([p for p in log_parts if p]))
                last_log_step = num_env_steps

            if (num_env_steps - last_evaluation_step) >= self.cfg.train.eval_interval:
                eval_results = self.eval()
                last_eval_log_time = self._log_eval_results(eval_results, last_eval_log_time)
                last_evaluation_step = num_env_steps

            cur_obs = next_obs

        # Run final eval
        eval_results = self.eval()
        _ = self._log_eval_results(eval_results, last_eval_log_time)
        # self.algo.save() # Will implement in the future

    def eval(self):
        eval_envs = self.cfg.train.extra["eval_envs"]
        num_envs = self.cfg.env.num_envs
        num_batches = eval_envs // num_envs

        all_episode_returns = []
        selected_action_value_sum = 0.0
        selected_action_value_count = 0

        for _ in range(num_batches):
            cur_obs, _ = self.eval_envs.reset()
            done = np.zeros(num_envs, dtype=bool)
            episode_returns = np.zeros(num_envs, dtype=np.float32)

            while not done.all():
                cur_obs_tensor = self._to_tensor_obs(cur_obs)
                batched_actions = self.algo.act(cur_obs_tensor, eval_mode=True)
                env_actions = self._actions_to_env(batched_actions.action)
                next_obs, rewards, terminations, truncations, infos = self.eval_envs.step(env_actions)

                active = ~done
                episode_returns[active] += rewards[active]

                if batched_actions.info is not None and "action_values" in batched_actions.info:
                    action_values = batched_actions.info["action_values"]
                    actions = batched_actions.action
                    selected = action_values.gather(1, actions.long())
                    active_mask = torch.as_tensor(active, device=selected.device)
                    selected_action_value_sum += selected[active_mask].sum().item()
                    selected_action_value_count += int(active_mask.sum().item())

                done |= (terminations | truncations)
                cur_obs = next_obs

            all_episode_returns.append(episode_returns)

        all_episode_returns = np.concatenate(all_episode_returns, axis=0)
        mean_return = float(np.mean(all_episode_returns))
        mean_selected_action_value = None
        if selected_action_value_count > 0:
            mean_selected_action_value = selected_action_value_sum / selected_action_value_count

        return {
            "Eval Mean Return": mean_return,
            "Eval Mean Selected Action Value": mean_selected_action_value if mean_selected_action_value is not None else 0.0,
            "Eval Episodes": int(all_episode_returns.shape[0]),
        }


    # =======================
    # Other helpers
    # =======================
    def _set_global_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _log_eval_results(self, eval_results, last_eval_log_time):
        now = time.time()
        elapsed = now - last_eval_log_time
        metrics = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in eval_results.items()])
        print(f"[eval] +{elapsed:.2f}s | {metrics}")
        return now

    def _to_tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def _actions_to_env(self, actions: torch.Tensor) -> np.ndarray:
        actions_np = actions.detach().cpu().numpy()
        if isinstance(self.train_envs.single_action_space, gym.spaces.Discrete) and actions_np.ndim == 2 and actions_np.shape[1] == 1:
            actions_np = actions_np.squeeze(1)
        return actions_np