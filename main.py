import time
import os
import shutil

from configs.config_loader import load_yaml_config
from trainer.trainer import Trainer
from trainer.helper import start_plot_server
from trainer.helper.plot_server import LIVE_PLOTS_DIR



configs = [
    # "configs/discrete_actions/ddqn/breakout.yaml",
    # "configs/discrete_actions/ddqn/lunarlander.yaml",
    "configs/discrete_actions/ddqn/cartpole.yaml",
]

START_PLOT_SERVER = False



def main():
    if os.path.exists(LIVE_PLOTS_DIR):
        for entry in os.listdir(LIVE_PLOTS_DIR):
            if entry == "index.html":
                continue
            path = os.path.join(LIVE_PLOTS_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    else:
        os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    if START_PLOT_SERVER:
        start_plot_server(open_browser=True)
    
    for config in configs:
        start_time = time.time()
        cfg = load_yaml_config(config)

        if cfg.inference.inference_only:
            print(f"{'='*70}\n# Evaluating on Config: {config:<44} #\n{'='*70}\n")
            trainer = Trainer(cfg)
            override_cfg = cfg if cfg.inference.override_cfg else None
            trainer.algo = trainer.algo.load(path = cfg.inference.algo_path, override_cfg = override_cfg)
            trainer.send_networks_to_device()
            trainer.eval(save_video=cfg.train.save_video_at_end)
            print(f"\nEval complete. Took {(time.time() - start_time):.1f}s.\n\n\n")

        else:            
            print(f"{'='*70}\n# Training on Config: {config:<46} #\n{'='*70}\n")
            trainer = Trainer(cfg)
            trainer.train()

            # Run final eval / save
            print(f"\n--- Final Evaluation + Saving ---")
            trainer.eval(save_video=cfg.train.save_video_at_end)
            if cfg.train.save_algo_at_end:
                trainer.save_algo()
            print(f"\nTraining complete. Took {(time.time() - start_time):.1f}s.\n\n\n")

if __name__ == "__main__":
    main()