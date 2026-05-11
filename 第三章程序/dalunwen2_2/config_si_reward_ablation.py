import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        mode = os.getenv("SI_REWARD_MODE", "si_only_norm").strip().lower()
        if mode not in ("si_only_norm", "si_only_raw"):
            raise ValueError("SI_REWARD_MODE must be si_only_norm or si_only_raw")

        self.final_reward_mode = mode
        self.continuous_final_reward = True
        self.smoothness_reward_weight = 1.0
        self.load_model = False
        self.model_tag = os.getenv("SI_REWARD_MODEL_TAG", "latest")
        self.save_frequency = int(os.getenv("SI_REWARD_SAVE_FREQUENCY", "50"))
        self.model_dir = os.path.join(os.path.dirname(__file__), f"models_{mode}")
        self.verbose = False
