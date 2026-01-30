from dataclasses import dataclass
from typing import Any
import os
import matplotlib.pyplot as plt
import torch

from trainer.helper.plot_server import update_manifest, LIVE_PLOTS_DIR

@dataclass
class RunResults:
    name: str
    value: Any
    accumulator: str    # Should be 'mean', 'max', 'concat', 'dict_concat', 'accumulating_writes'

@dataclass
class ResultAccumulator:
    def __init__(self, run_results: RunResults):
        self.name = run_results.name
        if run_results.accumulator == "concat":
            self.value = list(run_results.value)
        elif run_results.accumulator == "dict_concat":
            self.value = {}
            self._extend_dict_concat(self.value, run_results.value)
        elif run_results.accumulator == "accumulating_writes":
            self.value = self._normalize_accumulating_values(run_results.value)
        else:
            self.value = run_results.value
        self.accumulator = run_results.accumulator
        self.accumulated_counter = 1

    def update(self, run_results: RunResults):
        if self.accumulator == "mean":
            self.value = (self.value * self.accumulated_counter + run_results.value) / (self.accumulated_counter + 1)
            self.accumulated_counter += 1
        elif self.accumulator == "max":
            self.value = max(self.value, run_results.value)
        elif self.accumulator == "concat":
            self.value.extend(list(run_results.value))
        elif self.accumulator == "dict_concat":
            self._extend_dict_concat(self.value, run_results.value)
        elif self.accumulator == "accumulating_writes":
            self.value.extend(self._normalize_accumulating_values(run_results.value))

    def reset(self):
        if self.accumulator == "concat":
            self.value = []
        elif self.accumulator == "dict_concat":
            self.value = {k: [] for k in self.value.keys()}
        elif self.accumulator == "accumulating_writes":
            self.value = []
        else:
            self.value = 0
        self.accumulated_counter = 0

    @staticmethod
    def _extend_dict_concat(target: dict[str, list[float]], new_values: dict[str, Any]) -> None:
        for key, value in new_values.items():
            if key not in target:
                target[key] = []
            if isinstance(value, (list, tuple)):
                target[key].extend(list(value))
            else:
                target[key].append(value)

    @staticmethod
    def _normalize_accumulating_values(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 1 and value.numel() == 2:
                return [value]
            if value.ndim == 2 and value.shape[1] == 2:
                return list(value)
            return [value]
        if isinstance(value, (list, tuple)):
            if len(value) == 2 and not any(isinstance(v, (list, tuple, torch.Tensor)) for v in value):
                return [value]
            return list(value)
        return [value]


class ResultLogger():
    def __init__(self, name: str, logging_method: list[str], max_steps: int, wandb_run=None, *, dict_concat_stacked: bool = True):
        self.name = name
        self.tracked_logs: dict[str, ResultAccumulator] | None = None
        self.logging_method = logging_method
        self.max_steps = max_steps
        self.wandb_run = wandb_run
        self.dict_concat_stacked = dict_concat_stacked
        self._max_step_width = len(self._format_steps_short(self.max_steps)) if self.max_steps else 0
        self._plot_registry: dict[str, list[str]] = {}  # category -> list of relative paths
        self._update_counter: dict[str, int] = {}  # name -> running counter for accumulated writes
        
    def update(self, results: list[RunResults]):
        if results is None:
            return
        if self.tracked_logs is None:
            self.tracked_logs = {r.name: ResultAccumulator(r) for r in results}
            return

        for r in results:
            if r.name not in self.tracked_logs:
                self.tracked_logs[r.name] = ResultAccumulator(r)
            else:
                self.tracked_logs[r.name].update(r)
    
    def zero(self):
        if self.tracked_logs is None:
            return
        for acc in self.tracked_logs.values():
            acc.reset()

    def log(self, step: int, console_log=True):
        if self.tracked_logs is None:
            return

        log_dict = {name: acc.value for name, acc in self.tracked_logs.items() if acc.accumulator not in ("concat", "dict_concat", "accumulating_writes")}
        if "console" in self.logging_method and console_log:
            parts = []
            parts.append(f"step ={self._format_step(step)}")
            parts.extend([f"{k}={v:<6.2f}" if k != "Learning Rate" else f"{k}={v:<6.2e}" for k, v in log_dict.items()])
            line = " | ".join([p for p in parts if p])
            line = f"{self.name} {line}" if line else self.name
            print(line)

        new_plots = []
        for name, acc in self.tracked_logs.items():
            if acc.accumulator == "concat" and acc.value:
                new_plots.append(self._save_histogram(name, acc.value, step))
            if acc.accumulator == "dict_concat":
                if self.dict_concat_stacked:
                    filename = self._save_stacked_histogram(name, acc.value, step)
                    if filename is not None:
                        new_plots.append(filename)
                else:
                    for key, values in acc.value.items():
                        if values:
                            new_plots.append(self._save_histogram(f"{name}/{key}", values, step))
            if acc.accumulator == "accumulating_writes" and acc.value:
                self._append_accumulated_data(name, acc.value)
                filename = self._save_line_plot(name)
                if filename is not None:
                    new_plots.append(filename)
                acc.reset()

        if new_plots:
            self._update_plot_registry(new_plots)

        if "wandb" in self.logging_method and self.wandb_run is not None:
            wandb_prefix = self.name.strip()
            if wandb_prefix.startswith("[") and wandb_prefix.endswith("]"):
                wandb_prefix = wandb_prefix[1:-1]
            wandb_log_dict = {
                f"{wandb_prefix}/{k}" if wandb_prefix else k: v
                for k, v in log_dict.items()
            }
            self.wandb_run.log(wandb_log_dict, step=step)

    def _format_step(self, step: int) -> str:
        if not self.max_steps:
            return str(step)
        step_str = self._format_steps_short(step).rjust(self._max_step_width)
        return f"{step_str}/{self._format_steps_short(self.max_steps)}"

    @staticmethod
    def _format_steps_short(step: int) -> str:
        if step >= 1_000_000:
            return f"{step / 1_000_000:.1f}m"
        if step >= 1_000:
            return f"{step / 1_000:.1f}k"
        return str(step)

    # =======================================
    # Helper funcs I'm using to visualize various histograms live during training
    # =======================================
    @staticmethod
    def _get_category(name: str) -> str:
        name_lower = name.lower()
        if "surprise" in name_lower:
            return "surprise"
        if "grad" in name_lower:
            return "grad_magnitudes"
        if "loss" in name_lower:
            return "loss"
        return "other"

    @staticmethod
    def _save_histogram(name: str, values: list[float], step: int) -> tuple[str, str]:
        """Returns (category, relative_filepath)."""
        category = ResultLogger._get_category(name)
        save_dir = os.path.join(LIVE_PLOTS_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(5, 3))
        plt.hist(values, bins=30)
        plt.title(f"{name} @ {step}", fontsize=10)
        plt.tight_layout()
        
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{step}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=80)
        plt.close()
        return (category, f"{category}/{filename}")

    @staticmethod
    def _save_stacked_histogram(name: str, values: dict[str, list[float]], step: int) -> tuple[str, str] | None:
        """Returns (category, relative_filepath) or None."""
        keys = [k for k, v in values.items() if v]
        if not keys:
            return None
        data = [values[k] for k in keys]
        
        category = ResultLogger._get_category(name)
        save_dir = os.path.join(LIVE_PLOTS_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(5, 3))
        plt.hist(data, bins=30, stacked=True, label=keys)
        plt.legend(fontsize=7)
        plt.title(f"{name} @ {step}", fontsize=10)
        plt.tight_layout()
        
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{step}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=80)
        plt.close()
        return (category, f"{category}/{filename}")

    def _append_accumulated_data(self, name: str, values: list[Any]) -> None:
        category = ResultLogger._get_category(name)
        save_dir = os.path.join(LIVE_PLOTS_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(save_dir, f"{safe_name}_data.csv")
        write_header = not os.path.exists(file_path)
        
        if name not in self._update_counter:
            self._update_counter[name] = 0
        
        with open(file_path, "a") as f:
            if write_header:
                f.write("step,value\n")
            for value in values:
                step, val = self._parse_step_value(name, value)
                f.write(f"{step},{val}\n")

    def _parse_step_value(self, name: str, value: Any) -> tuple[int, float]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 1 and value.numel() == 2:
                step = int(value[0].item())
                val = float(value[1].item())
                return step, val
        if isinstance(value, (list, tuple)) and len(value) == 2:
            step = int(value[0])
            val = float(value[1])
            return step, val

        if name not in self._update_counter:
            self._update_counter[name] = 0
        step = self._update_counter[name]
        self._update_counter[name] += 1
        return step, float(value)

    @staticmethod
    def _ema_smooth(values: list[float], alpha: float = 0.9) -> list[float]:
        """Exponential moving average smoothing."""
        if not values:
            return []
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
        return smoothed

    @staticmethod
    def _save_line_plot(name: str, smooth_alpha: float = 0.9) -> tuple[str, str] | None:
        """Returns (category, relative_filepath) or None."""
        category = ResultLogger._get_category(name)
        data_dir = os.path.join(LIVE_PLOTS_DIR, category)
        safe_name = name.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(data_dir, f"{safe_name}_data.csv")
        if not os.path.exists(file_path):
            return None

        steps: list[int] = []
        values: list[float] = []
        with open(file_path, "r") as f:
            lines = f.read().strip().splitlines()
        if len(lines) <= 1:
            return None

        for line in lines[1:]:
            step_str, value_str = line.split(",", 1)
            steps.append(int(step_str))
            values.append(float(value_str))

        name_lower = name.lower()
        disable_smoothing = "eval reward" in name_lower or "eval_reward" in name_lower
        smoothed = [] if disable_smoothing else ResultLogger._ema_smooth(values, smooth_alpha)

        save_dir = os.path.join(LIVE_PLOTS_DIR, category)
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(5, 3))
        if disable_smoothing:
            plt.plot(steps, values, color='blue', linewidth=1.0)
        else:
            plt.plot(steps, values, alpha=0.3, color='blue', linewidth=0.5)
            plt.plot(steps, smoothed, color='blue', linewidth=1.5)
        plt.title(f"{name}", fontsize=10)
        plt.tight_layout()
        
        filename = f"{safe_name}_line.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=80)
        plt.close()
        return (category, f"{category}/{filename}")

    def _update_plot_registry(self, new_plots: list[tuple[str, str]]) -> None:
        for category, filepath in new_plots:
            scoped_category = f"{self.name.lower()}_{category}"
            if scoped_category not in self._plot_registry:
                self._plot_registry[scoped_category] = []
            existing = self._plot_registry[scoped_category]
            base_name = filepath.rsplit("_", 1)[0]
            self._plot_registry[scoped_category] = [p for p in existing if not p.startswith(base_name)]
            self._plot_registry[scoped_category].append(filepath)
        update_manifest(self._plot_registry)