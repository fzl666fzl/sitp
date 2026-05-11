import os

from config_gnn_qatten import Config as BaseConfig


def _env_bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(__file__)
        self.use_gnn = True
        self.use_gnn_graph_embedding = False
        self.use_gnn_action_bias = True
        self.gnn_aux_weight = 0.0
        self.gnn_bias_var_weight = 0.0
        self.gnn_action_fusion_mode = os.getenv(
            "ACTION_GNN_FUSION_MODE", "margin_gated"
        ).strip().lower()
        self.gnn_margin_threshold = float(os.getenv("ACTION_GNN_MARGIN_THRESHOLD", "1.0"))
        self.gnn_topk = int(os.getenv("ACTION_GNN_TOPK", "2"))
        self.gnn_rerank_weight = float(os.getenv("ACTION_GNN_RERANK_WEIGHT", "1.0"))
        self.use_si_predict_aux = True
        self.use_si_predict_load_features = True
        self.use_si_predict_rerank = _env_bool("SI_PREDICT_RERANK", True)
        self.si_predict_feature_dim = int(os.getenv("SI_PREDICT_FEATURE_DIM", "15"))
        self.si_predict_hidden_dim = int(os.getenv("SI_PREDICT_HIDDEN_DIM", "64"))
        self.si_predict_target_mode = "mean_centered"
        self.si_predict_output_dim = 5
        self.si_predict_loss_type = "smooth_l1"
        self.si_predict_time_scale = float(os.getenv("SI_PREDICT_TIME_SCALE", "700.0"))
        self.si_predict_deviation_scale = float(os.getenv("SI_PREDICT_DEVIATION_SCALE", "120.0"))
        self.si_predict_si_scale = float(os.getenv("SI_PREDICT_SI_SCALE", "30.0"))
        self.si_predict_aux_weight = 0.0
        self.si_predict_si_loss_weight = 1.0
        self.si_predict_penalty_weight = float(os.getenv("SI_PREDICT_PENALTY_WEIGHT", "0.1"))
        self.require_si_predictor_model = True
        self.si_predictor_model_path = os.getenv(
            "SI_PREDICT_MODEL_PATH",
            os.path.join(
                base_dir,
                "models_qatten_si_predict_load_v2",
                self.map_name,
                "latest_si_predictor_params.pkl",
            ),
        )
        self.model_dir = os.path.join(base_dir, "models_gnn_qatten_action_bias_aux010")
        self.model_tag = os.getenv("ACTION_GNN_MODEL_TAG", "latest")
        self.load_model = True
        self.record_gnn_diagnostics = _env_bool("ACTION_GNN_TRACE_DECISIONS", True)
