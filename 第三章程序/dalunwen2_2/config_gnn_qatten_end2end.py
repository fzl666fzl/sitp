import os

from config import Config as BaseConfig
from parameter import args_parser


def _env_bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        args = args_parser()

        self.action_mode = "combo_proc_team"
        self.combo_proc_count = int(args.pro_num)
        self.combo_team_count = int(args.team_num)
        self.n_agents = 1
        self.n_actions = self.combo_proc_count * self.combo_team_count
        self.episode_limit = int(os.getenv("END2END_EPISODE_LIMIT", "60"))

        self.mixer = "qatten"
        self.action_value_mode = os.getenv("END2END_ACTION_VALUE_MODE", "pair_scorer").strip().lower()
        if self.action_value_mode not in ("pair_scorer", "drqn_bias"):
            raise ValueError("END2END_ACTION_VALUE_MODE must be pair_scorer or drqn_bias")
        self.combo_pair_feature_dim = int(os.getenv("END2END_PAIR_FEATURE_DIM", "14"))
        self.combo_scorer_hidden_dim = int(os.getenv("END2END_COMBO_SCORER_HIDDEN_DIM", "128"))
        self.use_gnn = True
        self.use_gnn_graph_embedding = False
        self.use_gnn_action_bias = self.action_value_mode == "drqn_bias"
        self.gnn_action_bias_weight = float(os.getenv("END2END_GNN_BIAS_WEIGHT", "0.5"))
        self.gnn_action_fusion_mode = os.getenv("END2END_GNN_FUSION_MODE", "add_bias").strip().lower()
        if self.gnn_action_fusion_mode not in ("add_bias", "margin_gated", "topk_rerank"):
            raise ValueError("END2END_GNN_FUSION_MODE must be add_bias, margin_gated, or topk_rerank")
        self.gnn_margin_threshold = float(os.getenv("END2END_GNN_MARGIN_THRESHOLD", "0.5"))
        self.gnn_topk = int(os.getenv("END2END_GNN_TOPK", "2"))
        self.gnn_rerank_weight = float(os.getenv("END2END_GNN_RERANK_WEIGHT", "0.5"))
        self.gnn_bias_norm_scope = "candidate"
        self.gnn_bias_clip = float(os.getenv("END2END_GNN_BIAS_CLIP", "1.0"))
        self.gnn_aux_weight = 0.0
        self.gnn_bias_var_weight = 0.0
        self.gnn_use_load_penalty = False

        self.use_si_predict_aux = False
        self.use_si_predict_load_features = False
        self.use_si_predict_rerank = False
        self.use_qatten_load_rerank = False
        self.use_si_shaping_reward = False
        self.end2end_last_station_penalty_weight = float(
            os.getenv("END2END_LAST_STATION_PENALTY_WEIGHT", "1.0")
        )
        self.end2end_final_si_penalty_weight = float(
            os.getenv("END2END_FINAL_SI_PENALTY_WEIGHT", "0.0")
        )
        self.end2end_best_score_mode = os.getenv("END2END_BEST_SCORE_MODE", "pulse_plus_si")
        self.end2end_best_si_weight = float(os.getenv("END2END_BEST_SI_WEIGHT", "1.0"))
        self.end2end_validation_best_tag = os.getenv(
            "END2END_VALIDATION_BEST_TAG", "validation_best"
        ).strip() or "validation_best"
        self.end2end_online_pulse_best_tag = os.getenv(
            "END2END_ONLINE_PULSE_BEST_TAG", "online_pulse_best"
        ).strip() or "online_pulse_best"
        self.end2end_online_balanced_best_tag = os.getenv(
            "END2END_ONLINE_BALANCED_BEST_TAG", "online_balanced_best"
        ).strip() or "online_balanced_best"
        self.end2end_online_balanced_si_weight = float(
            os.getenv("END2END_ONLINE_BALANCED_SI_WEIGHT", str(self.end2end_best_si_weight))
        )
        self.enable_expert_distillation = _env_bool("END2END_ENABLE_EXPERT_DISTILLATION", False)
        self.expert_summary_path = os.getenv("END2END_EXPERT_SUMMARY_PATH", "").strip()
        self.expert_model_tag = os.getenv("END2END_EXPERT_MODEL_TAG", "expert_trace_best").strip() or "expert_trace_best"
        self.expert_init_pulse = int(os.getenv("END2END_EXPERT_INIT_PULSE", "0"))
        self.expert_seed = int(os.getenv("END2END_EXPERT_SEED", str(self.seed)))
        self.expert_disable_disturbance = _env_bool("END2END_EXPERT_DISABLE_DISTURBANCE", True)
        self.expert_pretrain_epochs = int(os.getenv("END2END_EXPERT_PRETRAIN_EPOCHS", "50"))
        self.expert_pretrain_imitation_weight = float(
            os.getenv("END2END_EXPERT_PRETRAIN_WEIGHT", "1.0")
        )
        self.expert_finetune_imitation_weight = float(
            os.getenv("END2END_EXPERT_FINETUNE_WEIGHT", "0.1")
        )
        self.end2end_validation_interval = int(os.getenv("END2END_VALIDATION_INTERVAL", "10"))
        self.expert_trace_interval = int(
            os.getenv("END2END_EXPERT_TRACE_INTERVAL", str(self.end2end_validation_interval))
        )
        self.end2end_validation_init_pulse = int(os.getenv("END2END_VALIDATION_INIT_PULSE", "608"))
        self.end2end_validation_seed = int(os.getenv("END2END_VALIDATION_SEED", str(self.seed)))
        self.end2end_epsilon_decay_fraction = float(
            os.getenv("END2END_EPSILON_DECAY_FRACTION", "0.70")
        )

        self.load_model = _env_bool("END2END_LOAD_MODEL", False)
        self.model_tag = os.getenv("END2END_MODEL_TAG", "latest")
        model_root = os.getenv("END2END_MODEL_DIR", "models_gnn_qatten_end2end")
        if not os.path.isabs(model_root):
            model_root = os.path.join(os.path.dirname(__file__), model_root)
        self.model_dir = model_root

        self.n_epochs = int(os.getenv("END2END_N_EPOCHS", str(self.n_epochs)))
        self.n_eposodes = int(os.getenv("END2END_N_EPISODES", str(self.n_eposodes)))
        self.train_steps = int(os.getenv("END2END_TRAIN_STEPS", str(self.train_steps)))
        self.save_frequency = int(os.getenv("END2END_SAVE_FREQUENCY", "5000"))
        self.verbose = _env_bool("END2END_VERBOSE", False)
