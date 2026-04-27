import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mixer = "qmix"
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_qmix_baseline")
        self.load_model = True
        self.model_tag = "4"
        self.verbose = False
