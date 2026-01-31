"""
Save config snapshots alongside final outputs.
return
"""

import json
import os
from typing import Any

from configs.config import config_to_dict, TrainConfig


def save_config_snapshot(run_name: str, cfg: TrainConfig, run_info: dict[str, Any], filename: str = "config.json") -> str:
    """
    Save config + run_info to saved_data/saved_plots/{run_name}.
    return
    """
    save_dir = os.path.join("saved_data", "saved_plots", run_name)
    os.makedirs(save_dir, exist_ok=True)

    payload = config_to_dict(cfg)
    payload["run_info"] = run_info

    file_path = os.path.join(save_dir, filename)
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)

    return file_path