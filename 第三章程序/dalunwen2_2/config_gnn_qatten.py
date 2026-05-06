import os

from config import Config as BaseConfig


def _env_bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mixer = "qatten"
        self.gnn_hidden_dim = 32
        self.gnn_embed_dim = 8
        self.gnn_layers = 2
        self.use_gnn_graph_embedding = _env_bool("ACTION_GNN_USE_GRAPH_EMBEDDING", False)
        self.use_gnn_action_bias = _env_bool("ACTION_GNN_USE_ACTION_BIAS", False)
        self.gnn_action_bias_weight = float(os.getenv("ACTION_GNN_ACTION_BIAS_WEIGHT", "0.5"))
        self.gnn_action_fusion_mode = os.getenv(
            "ACTION_GNN_FUSION_MODE", "margin_gated"
        ).strip().lower()
        if self.gnn_action_fusion_mode not in ("add_bias", "margin_gated", "topk_rerank"):
            raise ValueError(
                "ACTION_GNN_FUSION_MODE must be one of add_bias, margin_gated, topk_rerank"
            )
        self.gnn_margin_threshold = float(os.getenv("ACTION_GNN_MARGIN_THRESHOLD", "0.5"))
        self.gnn_topk = int(os.getenv("ACTION_GNN_TOPK", "2"))
        self.gnn_rerank_weight = float(os.getenv("ACTION_GNN_RERANK_WEIGHT", "0.5"))
        self.gnn_use_load_penalty = _env_bool("ACTION_GNN_USE_LOAD_PENALTY", False)
        self.gnn_load_penalty_weight = float(os.getenv("ACTION_GNN_LOAD_PENALTY_WEIGHT", "0"))
        self.gnn_bias_norm_scope = os.getenv("ACTION_GNN_BIAS_NORM_SCOPE", "candidate")
        self.gnn_bias_clip = float(os.getenv("ACTION_GNN_BIAS_CLIP", "1.0"))
        self.gnn_aux_target_type = os.getenv("ACTION_GNN_AUX_TARGET_TYPE", "successor_time")
        self.gnn_aux_loss_type = os.getenv("ACTION_GNN_AUX_LOSS_TYPE", "pairwise_rank")
        self.gnn_aux_weight = float(os.getenv("ACTION_GNN_AUX_WEIGHT", "0"))
        if not _env_bool("ACTION_GNN_USE_AUX_LOSS", self.gnn_aux_weight > 0):
            self.gnn_aux_weight = 0.0
        self.gnn_bias_var_weight = float(os.getenv("ACTION_GNN_BIAS_VAR_WEIGHT", "0"))
        self.use_gnn = any([
            self.use_gnn_graph_embedding,
            self.use_gnn_action_bias,
            self.gnn_aux_weight > 0,
            self.gnn_bias_var_weight > 0,
        ])
        load_original_qatten = _env_bool("ACTION_GNN_LOAD_ORIGINAL_QATTEN", False)
        model_name = "models" if load_original_qatten else (
            "models_gnn_qatten_action_bias_only"
            if self.use_gnn
            else "models_gnn_qatten_parity"
        )
        model_dir_override = os.getenv("ACTION_GNN_MODEL_DIR")
        model_root = model_dir_override if model_dir_override else model_name
        if not os.path.isabs(model_root):
            model_root = os.path.join(os.path.dirname(__file__), model_root)
        self.model_dir = model_root
        self.load_model = True
        self.model_tag = os.getenv(
            "ACTION_GNN_MODEL_TAG", "5" if load_original_qatten else "latest"
        )
        self.save_frequency = 5000
        self.verbose = False
