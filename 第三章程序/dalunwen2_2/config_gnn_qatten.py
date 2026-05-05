import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mixer = "qatten"
        self.use_gnn = True
        self.gnn_hidden_dim = 32
        self.gnn_embed_dim = 8
        self.gnn_layers = 2
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_gnn_qatten")
        self.load_model = True
        self.model_tag = "latest"
        self.save_frequency = 50
        self.verbose = False
