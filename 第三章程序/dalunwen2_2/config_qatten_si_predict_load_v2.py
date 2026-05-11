import os

from config_qatten_si_predict_load import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.si_predict_target_mode = "mean_centered"
        self.si_predict_output_dim = 5
        self.si_predict_loss_type = "smooth_l1"
        self.si_predict_deviation_scale = float(os.getenv("SI_PREDICT_DEVIATION_SCALE", "120.0"))
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_qatten_si_predict_load_v2")
