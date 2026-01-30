from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    name:               str
    num_envs:           int = 1
    max_episode_steps:  int | None = None
    is_atari:           bool = False
    stack_stamples:     int = 1
    env_args:           Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlgoConfig:
    name:                       str
    gamma:                      float = 0.99
    lr_start:                   float = 2.5e-4
    lr_end:                     float = 2.5e-4
    lr_warmup_env_steps:        int = 0
    update_every_steps:         int = 1    # 
    use_action_for_steps_train: int = 1    # Number of env steps to use same action for
    use_action_for_steps_eval:  int = 1
    batch_size:                 int = 32
    seed:               int | None = None  # If None (unset), random seed is generated
    extra:              Dict[str, Any] = field(default_factory=dict)

@dataclass
class SingleNetworkConfig:
    name:               str
    network_type:       str
    network_args:       dict

@dataclass
class NetworksConfig:
    networks:           Dict[str, SingleNetworkConfig]

@dataclass
class TrainParams:
    total_env_steps:    int
    threshold_exit_training_on_eval_score: float | None = None   # Score threshold to end training early if exceeds this
    log_interval:       int
    eval_interval:      int
    logging_method:     list[str]
    console_log_train:  bool = False
    wandb_project:      str
    wandb_group:        str | None = None
    wandb_run:          str | None = None
    video_save_dir:     str = "saved_data/saved_videos"
    algo_save_dir:      str = "saved_data/saved_algos"
    save_video_at_end:  bool = True
    save_algo_at_end:   bool = True
    extra:              Dict[str, Any] | None = None

@dataclass
class SamplerConfig:
    name:               str
    args:               Dict[str, Any]


# Set this true w/ values if you want to load a pretrained model for inference
@dataclass
class InferenceConfig:
    inference_only:     bool = False
    override_cfg:       bool
    algo_path:          str


# Config that encompasses all our configs used
@dataclass
class TrainConfig:
    env:                EnvConfig
    algo:               AlgoConfig
    networks:           NetworksConfig
    train:              TrainParams
    sampler:            SamplerConfig
    inference:          InferenceConfig