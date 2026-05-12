import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.use_si_shaping_reward = True
        self.si_shaping_beta = float(os.getenv("SI_SHAPING_BETA", "0.1"))
        beta_tag = str(self.si_shaping_beta).replace(".", "")
        self.model_dir = os.path.join(
            os.path.dirname(__file__), f"models_qatten_si_shaping_beta{beta_tag}"
        )
        self.model_tag = os.getenv("SI_SHAPING_MODEL_TAG", "latest")
        self.save_frequency = int(os.getenv("SI_SHAPING_SAVE_FREQUENCY", "50"))
        self.load_model = False
        self.verbose = False
