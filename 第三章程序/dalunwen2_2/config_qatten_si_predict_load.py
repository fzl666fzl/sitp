import os

from config_qatten_si_predict import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.use_si_predict_load_features = True
        self.si_predict_feature_dim = int(os.getenv("SI_PREDICT_FEATURE_DIM", "15"))
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_qatten_si_predict_load")
