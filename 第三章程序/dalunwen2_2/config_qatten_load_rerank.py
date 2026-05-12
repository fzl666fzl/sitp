import os

from config import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.train = False
        self.load_model = True
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_tag = os.getenv("QATTEN_LOAD_MODEL_TAG", "5")

        self.use_gnn = False
        self.use_gnn_graph_embedding = False
        self.use_gnn_action_bias = False
        self.use_qatten_load_rerank = True
        self.record_load_rerank_diagnostics = True
        self.load_rerank_records = []

        self.qatten_load_rerank_mode = os.getenv(
            "QATTEN_LOAD_RERANK_MODE", "margin_gated"
        )
        self.qatten_load_margin_threshold = float(
            os.getenv("QATTEN_LOAD_MARGIN_THRESHOLD", "1.0")
        )
        self.qatten_load_topk = int(os.getenv("QATTEN_LOAD_TOPK", "2"))
        self.qatten_load_penalty_weight = float(
            os.getenv("QATTEN_LOAD_PENALTY_WEIGHT", "0.1")
        )
