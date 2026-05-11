import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.use_si_predict_aux = True
        self.si_predict_hidden_dim = int(os.getenv("SI_PREDICT_HIDDEN_DIM", "64"))
        self.si_predict_aux_weight = float(os.getenv("SI_PREDICT_AUX_WEIGHT", "0.05"))
        self.si_predict_si_loss_weight = float(os.getenv("SI_PREDICT_SI_LOSS_WEIGHT", "1.0"))
        self.si_predict_time_scale = float(os.getenv("SI_PREDICT_TIME_SCALE", "700.0"))
        self.si_predict_si_scale = float(os.getenv("SI_PREDICT_SI_SCALE", "30.0"))
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_qatten_si_predict")
        self.model_tag = os.getenv("SI_PREDICT_MODEL_TAG", "latest")
        self.save_frequency = int(os.getenv("SI_PREDICT_SAVE_FREQUENCY", "5000"))
        self.load_model = False
        self.verbose = False
