import os

from config_gnn_qatten import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.use_gnn_graph_embedding = False
        self.use_gnn_action_bias = True
        self.use_gnn = True
        self.gnn_aux_target_type = "si_aware"
        self.gnn_aux_loss_type = "pairwise_rank"
        self.gnn_aux_weight = float(os.getenv("ACTION_GNN_SI_AUX_WEIGHT", "0.1"))
        self.gnn_action_fusion_mode = os.getenv(
            "ACTION_GNN_FUSION_MODE", "margin_gated"
        ).strip().lower()
        self.gnn_margin_threshold = float(os.getenv("ACTION_GNN_MARGIN_THRESHOLD", "1.0"))
        self.gnn_rerank_weight = float(os.getenv("ACTION_GNN_RERANK_WEIGHT", "1.0"))
        self.gnn_use_load_penalty = False
        self.model_dir = os.path.join(os.path.dirname(__file__), "models_gnn_qatten_si_aware")
        self.model_tag = os.getenv("ACTION_GNN_MODEL_TAG", "latest")
        self.load_model = True
        self.save_frequency = int(os.getenv("ACTION_GNN_SAVE_FREQUENCY", "5000"))
