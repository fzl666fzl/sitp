import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from graph_utils import build_procedure_graph
from NN import DRQN, QMIXNET, ProcedureGraphEncoder, QattenMixer, StationTimePredictor
from parameter import args_parser

GNN_ACTION_WEIGHTS = [
    [1, 0, 0],
    [0.8, 0.2, 0],
    [0.8, 0, 0.2],
    [0.6, 0.2, 0.2],
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0, 0.8],
    [0.2, 0.3, 0.5],
    [0.2, 0.8, 0],
]


class QMIX:
    def __init__(self, conf):
        self.conf = conf
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_shape = self.conf.obs_shape
        self.mixer = self.conf.mixer.lower()
        self.use_gnn = bool(getattr(self.conf, "use_gnn", False))
        self.use_gnn_graph_embedding = bool(getattr(self.conf, "use_gnn_graph_embedding", self.use_gnn))
        self.use_gnn_action_bias = bool(getattr(self.conf, "use_gnn_action_bias", False))
        self.use_si_predict_aux = bool(getattr(self.conf, "use_si_predict_aux", False))
        self.use_si_predict_load_features = bool(
            getattr(self.conf, "use_si_predict_load_features", False)
        )
        self.si_predict_feature_dim = int(getattr(self.conf, "si_predict_feature_dim", 0))
        self.external_si_predictor_path = os.path.abspath(
            getattr(self.conf, "si_predictor_model_path", "") or ""
        )
        input_shape = self.obs_shape

        if self.conf.last_action:
            input_shape += self.n_actions
        if self.conf.reuse_network:
            input_shape += self.n_agents
        if self.use_gnn and self.use_gnn_graph_embedding:
            input_shape += self.conf.gnn_embed_dim

        self.eval_graph_encoder = None
        self.target_graph_encoder = None
        self.pro_id_to_idx = {}
        self.idx_to_pro_id = {}
        self.rule_features = None
        self.gnn_action_weights = None
        if self.use_gnn:
            adjacency, node_features, pro_ids, rule_features = build_procedure_graph(args_parser())
            self.pro_id_to_idx = {pro_id: idx for idx, pro_id in enumerate(pro_ids)}
            self.idx_to_pro_id = {idx: pro_id for pro_id, idx in self.pro_id_to_idx.items()}
            self.rule_features = rule_features.to(self.device)
            self.gnn_action_weights = torch.tensor(GNN_ACTION_WEIGHTS, dtype=torch.float32, device=self.device)
            self.eval_graph_encoder = ProcedureGraphEncoder(adjacency, node_features, self.conf).to(self.device)
            self.target_graph_encoder = ProcedureGraphEncoder(adjacency, node_features, self.conf).to(self.device)

        self.eval_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, self.conf).to(self.device)

        mixer_cls = QattenMixer if self.mixer == "qatten" else QMIXNET
        self.eval_mixer_net = mixer_cls(self.conf).to(self.device)
        self.target_mixer_net = mixer_cls(self.conf).to(self.device)
        self.si_predictor = None
        if self.use_si_predict_aux:
            si_predict_input_shape = self.state_shape + self.n_actions
            if self.use_si_predict_load_features:
                si_predict_input_shape += self.si_predict_feature_dim
            self.si_predictor = StationTimePredictor(si_predict_input_shape, self.conf).to(self.device)

        self.model_dir = self._model_roots()[0]
        self.conf.loaded_model_tag = None
        self.conf.loaded_drqn_path = None
        self.conf.loaded_mixer_path = None
        self.conf.loaded_gnn_path = None
        self.conf.loaded_si_predictor_path = None

        if self.conf.load_model:
            self._load_model_if_available()
            self._load_external_si_predictor_if_available()

        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())
        if self.use_gnn:
            self.target_graph_encoder.load_state_dict(self.eval_graph_encoder.state_dict())

        self.eval_parameters = list(self.eval_mixer_net.parameters()) + list(self.eval_drqn_net.parameters())
        if self.use_gnn:
            self.eval_parameters += list(self.eval_graph_encoder.parameters())
        if self.use_si_predict_aux:
            self.eval_parameters += list(self.si_predictor.parameters())
        if self.conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)

        self.eval_hidden = None
        self.target_hidden = None

        if getattr(self.conf, "verbose", False):
            print("init {} nets finished!".format(self.mixer))

    def _model_roots(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        configured_base = os.path.abspath(self.conf.model_dir)
        if os.path.basename(configured_base) == self.conf.map_name:
            configured_root = configured_base
        else:
            configured_root = os.path.abspath(os.path.join(configured_base, self.conf.map_name))
        default_root = os.path.abspath(os.path.join(base_dir, "models", self.conf.map_name))
        roots = [
            configured_root,
            default_root,
        ]
        deduped = []
        for root in roots:
            if root not in deduped:
                deduped.append(root)
        return deduped

    def _candidate_model_pairs(self):
        candidates = []
        seen_pairs = set()
        drqn_pattern = re.compile(r"^(\d+)_drqn_net_params\.pkl$")
        mixer_pattern = re.compile(rf"^(\d+)_{self.mixer}_mixer_params\.pkl$")
        gnn_pattern = re.compile(r"^(\d+)_gnn_encoder_params\.pkl$")
        si_predictor_pattern = re.compile(r"^(\d+)_si_predictor_params\.pkl$")
        legacy_qmix_pattern = re.compile(r"^(\d+)_qmix_net_params\.pkl$")
        requested_tag = str(getattr(self.conf, "model_tag", "latest") or "latest").strip()

        def add_candidate(tag, drqn_path, mixer_path, gnn_path=None, si_predictor_path=None):
            if not self.use_gnn:
                gnn_path = None
            if not self.use_si_predict_aux:
                si_predictor_path = None
            pair_key = (
                os.path.abspath(drqn_path),
                os.path.abspath(mixer_path),
                os.path.abspath(gnn_path) if gnn_path else "",
                os.path.abspath(si_predictor_path) if si_predictor_path else "",
            )
            if pair_key in seen_pairs:
                return
            has_required_gnn = (not self.use_gnn) or (gnn_path and os.path.exists(gnn_path))
            has_required_si_predictor = (
                not self.use_si_predict_aux
            ) or (
                self.external_si_predictor_path
                and os.path.exists(self.external_si_predictor_path)
            ) or (si_predictor_path and os.path.exists(si_predictor_path))
            if (
                os.path.exists(drqn_path)
                and os.path.exists(mixer_path)
                and has_required_gnn
                and has_required_si_predictor
            ):
                seen_pairs.add(pair_key)
                candidates.append((str(tag), drqn_path, mixer_path, gnn_path, si_predictor_path))

        for model_root in self._model_roots():
            if not os.path.isdir(model_root):
                continue

            latest_drqn = os.path.join(model_root, "latest_drqn_net_params.pkl")
            latest_mixer = os.path.join(model_root, f"latest_{self.mixer}_mixer_params.pkl")
            latest_gnn = os.path.join(model_root, "latest_gnn_encoder_params.pkl")
            latest_si_predictor = os.path.join(model_root, "latest_si_predictor_params.pkl")

            drqn_files = {}
            mixer_files = {}
            gnn_files = {}
            si_predictor_files = {}
            for filename in os.listdir(model_root):
                drqn_match = drqn_pattern.match(filename)
                if drqn_match:
                    drqn_files[drqn_match.group(1)] = os.path.join(model_root, filename)
                    continue

                gnn_match = gnn_pattern.match(filename)
                if gnn_match:
                    gnn_files[gnn_match.group(1)] = os.path.join(model_root, filename)
                    continue

                si_predictor_match = si_predictor_pattern.match(filename)
                if si_predictor_match:
                    si_predictor_files[si_predictor_match.group(1)] = os.path.join(model_root, filename)
                    continue

                mixer_match = mixer_pattern.match(filename)
                if mixer_match:
                    mixer_files[mixer_match.group(1)] = os.path.join(model_root, filename)
                    continue

                if self.mixer == "qmix":
                    legacy_match = legacy_qmix_pattern.match(filename)
                    if legacy_match:
                        mixer_files[legacy_match.group(1)] = os.path.join(model_root, filename)

            if requested_tag == "latest":
                add_candidate("latest", latest_drqn, latest_mixer, latest_gnn, latest_si_predictor)
            else:
                exact_drqn = drqn_files.get(requested_tag)
                exact_mixer = mixer_files.get(requested_tag)
                exact_gnn = gnn_files.get(requested_tag)
                exact_si_predictor = si_predictor_files.get(requested_tag)
                if exact_drqn and exact_mixer:
                    add_candidate(
                        requested_tag,
                        exact_drqn,
                        exact_mixer,
                        exact_gnn,
                        exact_si_predictor,
                    )
                add_candidate("latest", latest_drqn, latest_mixer, latest_gnn, latest_si_predictor)

            common_prefixes = set(drqn_files.keys()) & set(mixer_files.keys())
            if self.use_gnn:
                common_prefixes = common_prefixes & set(gnn_files.keys())
            if self.use_si_predict_aux:
                common_prefixes = common_prefixes & set(si_predictor_files.keys())
            common_prefixes = sorted(common_prefixes, key=lambda item: int(item), reverse=True)
            for prefix in common_prefixes:
                add_candidate(
                    prefix,
                    drqn_files[prefix],
                    mixer_files[prefix],
                    gnn_files.get(prefix),
                    si_predictor_files.get(prefix),
                )

        return candidates

    def _load_model_if_available(self):
        loaded = False
        for model_tag, drqn_path, mixer_path, gnn_path, si_predictor_path in self._candidate_model_pairs():
            if os.path.exists(drqn_path) and os.path.exists(mixer_path):
                self.eval_drqn_net.load_state_dict(self._load_state_dict(drqn_path))
                self.eval_mixer_net.load_state_dict(self._load_state_dict(mixer_path))
                if self.use_gnn and gnn_path:
                    self.eval_graph_encoder.load_state_dict(self._load_state_dict(gnn_path))
                if self.use_si_predict_aux and si_predictor_path and os.path.exists(si_predictor_path):
                    self.si_predictor.load_state_dict(self._load_state_dict(si_predictor_path))
                print("successfully load models:", drqn_path, mixer_path)
                self.conf.loaded_model_tag = model_tag
                self.conf.loaded_drqn_path = drqn_path
                self.conf.loaded_mixer_path = mixer_path
                self.conf.loaded_gnn_path = gnn_path
                self.conf.loaded_si_predictor_path = si_predictor_path
                loaded = True
                break
        if not loaded:
            suffix = " with GNN" if self.use_gnn else ""
            print("model files not found for {}{}, continue with random initialization.".format(self.mixer, suffix))

    def _load_external_si_predictor_if_available(self):
        if not (self.use_si_predict_aux and self.external_si_predictor_path):
            return
        if not os.path.exists(self.external_si_predictor_path):
            if getattr(self.conf, "require_si_predictor_model", False):
                raise FileNotFoundError(
                    "external SI predictor not found: {}".format(self.external_si_predictor_path)
                )
            return
        self.si_predictor.load_state_dict(self._load_state_dict(self.external_si_predictor_path))
        self.conf.loaded_si_predictor_path = self.external_si_predictor_path
        print("successfully load external si predictor:", self.external_si_predictor_path)

    def _load_state_dict(self, model_path):
        try:
            with open(model_path, "rb") as file_obj:
                return torch.load(file_obj, map_location=self.device, weights_only=True)
        except TypeError:
            with open(model_path, "rb") as file_obj:
                return torch.load(file_obj, map_location=self.device)

    def _save_state_dict(self, state_dict, model_path):
        with open(model_path, "wb") as file_obj:
            torch.save(state_dict, file_obj)

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s = batch['s'].to(self.device)
        s_ = batch['s_'].to(self.device)
        u = batch['u'].to(self.device)
        r = batch['r'].to(self.device)
        terminated = batch['terminated'].to(self.device)
        mask = (1 - batch['padded'].float()).to(self.device)

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        q_evals = self._add_action_bias(q_evals, batch, "order_mask", self.eval_graph_encoder)
        q_targets = self._add_action_bias(q_targets, batch, "order_mask_", self.target_graph_encoder)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]

        reward_mean = float(torch.max(r, dim=1)[0].mean().item())
        if getattr(self.conf, "verbose", False):
            print("reward mean", reward_mean)

        q_total_eval = self.eval_mixer_net(q_evals, s)
        q_total_target = self.target_mixer_net(q_targets, s_)

        targets = r + self.conf.gamma * q_total_target * (1 - terminated)
        td_error = q_total_eval - targets.detach()
        mask_td_error = mask * td_error
        loss = (mask_td_error ** 2).sum() / mask.sum()
        aux_loss = self._candidate_aux_ranking_loss(batch, max_episode_len)
        if aux_loss is not None:
            loss = loss + float(getattr(self.conf, "gnn_aux_weight", 0.0)) * aux_loss
        si_predict_loss = self._si_predict_aux_loss(batch, max_episode_len)
        if si_predict_loss is not None:
            loss = loss + float(getattr(self.conf, "si_predict_aux_weight", 0.0)) * si_predict_loss

        print("*******开始训练({})*********".format(self.mixer), loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())
            if self.use_gnn:
                self.target_graph_encoder.load_state_dict(self.eval_graph_encoder.state_dict())

    def _station_times_to_si(self, station_times):
        centered = station_times - station_times.mean(dim=-1, keepdim=True)
        return torch.sqrt(torch.mean(centered ** 2, dim=-1, keepdim=True) + 1e-6)

    def _decode_si_prediction(self, pred_raw):
        if getattr(self.conf, "si_predict_target_mode", "absolute") != "mean_centered":
            scale = float(getattr(self.conf, "si_predict_time_scale", 700.0))
            return pred_raw.view(*pred_raw.shape[:-1], 4) * scale

        time_scale = float(getattr(self.conf, "si_predict_time_scale", 700.0))
        deviation_scale = float(getattr(self.conf, "si_predict_deviation_scale", 120.0))
        mean_time = pred_raw[..., :1] * time_scale
        centered = pred_raw[..., 1:5] * deviation_scale
        centered = centered - centered.mean(dim=-1, keepdim=True)
        return torch.clamp(mean_time + centered, min=0.0)

    def _si_predict_inputs(self, states, actions, batch, max_episode_len):
        parts = [states, actions]
        if self.use_si_predict_load_features:
            features = batch.get("si_predict_features")
            if features is None:
                features = torch.zeros(
                    states.shape[0],
                    max_episode_len,
                    self.si_predict_feature_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                features = features[:, :max_episode_len].to(self.device)
            parts.append(features)
        return torch.cat(parts, dim=-1)

    def predict_si_values_for_action_features(self, state, action_features):
        if not (self.use_si_predict_aux and self.si_predictor is not None):
            return None
        if action_features is None:
            return None
        features = np.asarray(action_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != self.n_actions:
            return None
        if features.shape[1] < self.si_predict_feature_dim:
            pad_width = self.si_predict_feature_dim - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
        elif features.shape[1] > self.si_predict_feature_dim:
            features = features[:, : self.si_predict_feature_dim]

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape[0] != self.state_shape:
            return None
        states = torch.tensor(
            np.repeat(state_array.reshape(1, -1), self.n_actions, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.eye(self.n_actions, dtype=torch.float32, device=self.device)
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        inputs = torch.cat([states, actions, feature_tensor], dim=-1)
        with torch.no_grad():
            pred_raw = self.si_predictor(inputs)
            pred_station_times = self._decode_si_prediction(pred_raw)
            pred_si = self._station_times_to_si(pred_station_times)
        return pred_si.view(-1)

    def _si_predict_aux_loss(self, batch, max_episode_len):
        if not (
            self.use_si_predict_aux
            and getattr(self.conf, "si_predict_aux_weight", 0.0) > 0
            and "final_station_times" in batch
            and "final_si" in batch
        ):
            return None

        states = batch["s"][:, :max_episode_len].to(self.device)
        actions = batch["u_onehot"][:, :max_episode_len, 0, :].to(self.device)
        labels = batch["final_station_times"][:, :max_episode_len].to(self.device)
        final_si = batch["final_si"][:, :max_episode_len].to(self.device)
        valid_mask = (
            (1 - batch["padded"][:, :max_episode_len].float()).to(self.device).squeeze(-1) > 0
        )
        if not bool(valid_mask.any().item()):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        scale = float(getattr(self.conf, "si_predict_time_scale", 700.0))
        inputs = self._si_predict_inputs(states, actions, batch, max_episode_len)
        pred_norm = self.si_predictor(inputs.reshape(-1, inputs.shape[-1]))
        pred = self._decode_si_prediction(pred_norm).view(states.shape[0], max_episode_len, 4)
        station_error = (pred - labels) / scale
        if getattr(self.conf, "si_predict_loss_type", "mse") == "smooth_l1":
            station_loss = F.smooth_l1_loss(station_error, torch.zeros_like(station_error), reduction="none")
        else:
            station_loss = station_error ** 2
        station_mse = station_loss.mean(dim=-1, keepdim=True)
        pred_si = self._station_times_to_si(pred)
        si_scale = float(getattr(self.conf, "si_predict_si_scale", 30.0))
        si_error = (pred_si - final_si) / si_scale
        if getattr(self.conf, "si_predict_loss_type", "mse") == "smooth_l1":
            si_loss = F.smooth_l1_loss(si_error, torch.zeros_like(si_error), reduction="none")
        else:
            si_loss = si_error ** 2
        combined = station_mse + float(getattr(self.conf, "si_predict_si_loss_weight", 1.0)) * si_loss
        return combined.squeeze(-1)[valid_mask].mean()

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_ = self._get_inputs(batch, transition_idx)

            inputs = inputs.to(self.device)
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)
            q_eval, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_drqn_net(inputs_, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def evaluate_si_prediction_batch(self, batch, max_episode_len=None):
        if not self.use_si_predict_aux or self.si_predictor is None:
            return None
        if "final_station_times" not in batch or "final_si" not in batch:
            return None
        if max_episode_len is None:
            max_episode_len = batch["s"].shape[1]
        prepared = {}
        for key, value in batch.items():
            dtype = torch.long if key == "u" else torch.float32
            prepared[key] = torch.tensor(np.asarray(value), dtype=dtype)
        expected_dims = {
            "s": 3,
            "u_onehot": 4,
            "final_station_times": 3,
            "final_si": 3,
            "padded": 3,
            "si_predict_features": 3,
        }
        for key, expected_dim in expected_dims.items():
            if key in prepared and prepared[key].dim() == expected_dim - 1:
                prepared[key] = prepared[key].unsqueeze(0)

        states = prepared["s"][:, :max_episode_len].to(self.device)
        actions = prepared["u_onehot"][:, :max_episode_len, 0, :].to(self.device)
        labels = prepared["final_station_times"][:, :max_episode_len].to(self.device)
        final_si = prepared["final_si"][:, :max_episode_len].to(self.device)
        valid_mask = (
            (1 - prepared["padded"][:, :max_episode_len].float()).to(self.device).squeeze(-1) > 0
        )
        if not bool(valid_mask.any().item()):
            return None

        scale = float(getattr(self.conf, "si_predict_time_scale", 700.0))
        with torch.no_grad():
            inputs = self._si_predict_inputs(states, actions, prepared, max_episode_len)
            pred_norm = self.si_predictor(inputs.reshape(-1, inputs.shape[-1]))
            pred = self._decode_si_prediction(pred_norm).view(states.shape[0], max_episode_len, 4)
            pred_si = self._station_times_to_si(pred)

        pred_station = pred[valid_mask].detach().cpu().numpy()
        true_station = labels[valid_mask].detach().cpu().numpy()
        pred_si_values = pred_si[valid_mask].view(-1).detach().cpu().numpy()
        true_si_values = final_si[valid_mask].view(-1).detach().cpu().numpy()
        si_corr = None
        if pred_si_values.size > 1 and np.std(pred_si_values) > 1e-8 and np.std(true_si_values) > 1e-8:
            si_corr = float(np.corrcoef(pred_si_values, true_si_values)[0, 1])

        return {
            "station_time_mae": float(np.mean(np.abs(pred_station - true_station))),
            "si_mae": float(np.mean(np.abs(pred_si_values - true_si_values))),
            "si_corr": si_corr,
            "pred_station_times_mean": [float(item) for item in np.mean(pred_station, axis=0).tolist()],
            "true_station_times_mean": [float(item) for item in np.mean(true_station, axis=0).tolist()],
            "pred_si_mean": float(np.mean(pred_si_values)),
            "true_si_mean": float(np.mean(true_si_values)),
        }

    def get_eval_graph_embedding_numpy(self, force_zero=None):
        if not (self.use_gnn and self.use_gnn_graph_embedding):
            return None
        zero_graph = getattr(self.conf, "zero_gnn_embedding", False) if force_zero is None else force_zero
        if zero_graph:
            return torch.zeros(self.conf.gnn_embed_dim).cpu().numpy()
        with torch.no_grad():
            return self.eval_graph_encoder().detach().cpu().numpy()

    def get_order_mask_numpy(self, order_candidates):
        node_count = getattr(self.conf, "gnn_node_count", len(self.pro_id_to_idx))
        mask = torch.zeros(node_count, dtype=torch.float32)
        if not order_candidates:
            return mask.numpy()
        for order_id in order_candidates:
            idx = self.pro_id_to_idx.get(int(order_id))
            if idx is not None:
                mask[idx] = 1.0
        return mask.numpy()

    def get_eval_action_bias(self, order_candidates, agent_num, force_zero=None, weight_override=None):
        if not (self.use_gnn and self.use_gnn_action_bias) or agent_num != 0:
            return None
        zero_bias = getattr(self.conf, "zero_gnn_embedding", False) if force_zero is None else force_zero
        if zero_bias:
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        order_mask = torch.tensor(
            self.get_order_mask_numpy(order_candidates),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, -1)
        return self._action_bias_from_mask(
            order_mask, self.eval_graph_encoder, weight_override=weight_override
        )[0, 0, agent_num]

    def get_rule_node_ids(self, order_candidates):
        if not (self.use_gnn and order_candidates):
            return [-1] * self.n_actions
        order_mask = torch.tensor(
            self.get_order_mask_numpy(order_candidates),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, -1)
        selected_indices, valid_mask = self._candidate_rule_indices(order_mask)
        rule_node_ids = []
        for action_idx in range(self.n_actions):
            if not bool(valid_mask[0, 0, action_idx].item()):
                rule_node_ids.append(-1)
                continue
            node_idx = int(selected_indices[0, 0, action_idx].item())
            rule_node_ids.append(int(self.idx_to_pro_id.get(node_idx, -1)))
        return rule_node_ids

    def get_candidate_action_stats(self, order_candidates, action_bias=None):
        defaults = {
            "candidate_bias_std": None,
            "candidate_target_std": None,
            "candidate_target_bias_corr": None,
            "candidate_pairwise_acc": None,
        }
        if not (self.use_gnn and self.use_gnn_action_bias and order_candidates):
            return defaults

        rule_node_ids = self.get_rule_node_ids(order_candidates)
        valid_positions = [idx for idx, node_id in enumerate(rule_node_ids) if node_id != -1]
        if len(valid_positions) < 2:
            return defaults

        node_indices = torch.tensor(
            [self.pro_id_to_idx[rule_node_ids[idx]] for idx in valid_positions],
            dtype=torch.long,
            device=self.device,
        )
        targets = self._node_targets()[node_indices].detach().cpu().numpy()
        if action_bias is None:
            values = self.eval_graph_encoder.node_scores()[node_indices].detach().cpu().numpy()
        else:
            values = action_bias.detach().cpu().numpy()[valid_positions]

        bias_std = float(np.std(values))
        target_std = float(np.std(targets))
        corr = None
        if bias_std > 1e-8 and target_std > 1e-8:
            corr = float(np.corrcoef(targets, values)[0, 1])

        pair_total = 0
        pair_correct = 0
        for i in range(len(valid_positions)):
            for j in range(len(valid_positions)):
                if targets[i] <= targets[j]:
                    continue
                pair_total += 1
                pair_correct += int(values[i] > values[j])

        return {
            "candidate_bias_std": bias_std,
            "candidate_target_std": target_std,
            "candidate_target_bias_corr": corr,
            "candidate_pairwise_acc": round(pair_correct / pair_total, 6) if pair_total else None,
        }

    def get_rule_target_values(self, rule_node_ids):
        if not (self.use_gnn and self.rule_features is not None):
            return [None] * self.n_actions
        targets = self._node_targets().detach()
        values = []
        for node_id in rule_node_ids:
            node_idx = self.pro_id_to_idx.get(int(node_id)) if node_id != -1 else None
            if node_idx is None:
                values.append(None)
            else:
                values.append(float(targets[node_idx].item()))
        return values

    def _node_targets(self):
        target_type = getattr(self.conf, "gnn_aux_target_type", "successor_time")
        target_columns = {
            "successor_count": 0,
            "successor_time": 1,
            "critical_path": 2,
        }
        column_idx = target_columns.get(target_type, 1)
        return self.rule_features[:, column_idx]

    def _candidate_rule_indices(self, order_mask):
        batch_size, episode_len, _ = order_mask.shape
        has_candidate = order_mask.sum(dim=-1) > 0
        rule_priority = torch.matmul(self.rule_features, self.gnn_action_weights.t())
        masked_priority = rule_priority.view(1, 1, -1, self.n_actions).expand(
            batch_size, episode_len, -1, -1
        )
        masked_priority = masked_priority.masked_fill(order_mask.unsqueeze(-1) <= 0, -float("inf"))
        selected_indices = torch.argmax(masked_priority, dim=2)
        valid_mask = has_candidate.unsqueeze(-1).expand(batch_size, episode_len, self.n_actions)
        selected_indices = selected_indices.masked_fill(~valid_mask, 0)
        return selected_indices, valid_mask

    def _candidate_aux_ranking_loss(self, batch, max_episode_len):
        if not (
            self.use_gnn
            and self.use_gnn_action_bias
            and getattr(self.conf, "gnn_aux_weight", 0.0) > 0
            and getattr(self.conf, "gnn_aux_loss_type", "pairwise_rank") == "pairwise_rank"
            and "order_mask" in batch
        ):
            return None

        order_mask = batch["order_mask"][:, :max_episode_len].to(self.device)
        step_mask = (1 - batch["padded"][:, :max_episode_len].float()).to(self.device).squeeze(-1) > 0
        selected_indices, valid_mask = self._candidate_rule_indices(order_mask)
        node_scores = self.eval_graph_encoder.node_scores()
        selected_scores = node_scores[selected_indices]
        if getattr(self.conf, "gnn_aux_target_type", "successor_time") == "si_aware":
            if "order_si_target" not in batch:
                return None
            selected_targets = batch["order_si_target"][:, :max_episode_len].to(self.device)
        else:
            selected_targets = self._node_targets()[selected_indices]

        valid_mask = valid_mask & step_mask.unsqueeze(-1)
        pair_mask = (
            valid_mask.unsqueeze(-1)
            & valid_mask.unsqueeze(-2)
            & (selected_targets.unsqueeze(-1) > selected_targets.unsqueeze(-2) + 1e-6)
        )
        if not bool(pair_mask.any().item()):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        score_diff = selected_scores.unsqueeze(-1) - selected_scores.unsqueeze(-2)
        pair_losses = F.softplus(-score_diff)
        return pair_losses[pair_mask].mean()

    def _add_action_bias(self, q_values, batch, mask_key, encoder):
        if not (self.use_gnn and self.use_gnn_action_bias) or mask_key not in batch:
            return q_values
        return q_values + self._action_bias_from_mask(batch[mask_key].to(self.device), encoder)

    def _action_bias_from_mask(self, order_mask, encoder, weight_override=None):
        batch_size, episode_len, _ = order_mask.shape
        bias = torch.zeros(
            batch_size,
            episode_len,
            self.n_agents,
            self.n_actions,
            dtype=torch.float32,
            device=order_mask.device,
        )
        if getattr(self.conf, "zero_gnn_embedding", False):
            return bias

        node_scores = encoder.node_scores()
        selected_indices, valid_mask = self._candidate_rule_indices(order_mask)
        action_bias = node_scores[selected_indices].masked_fill(~valid_mask, 0.0)
        if getattr(self.conf, "gnn_bias_norm_scope", "candidate") == "candidate":
            denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = action_bias.sum(dim=-1, keepdim=True) / denom
            centered = (action_bias - mean).masked_fill(~valid_mask, 0.0)
            var = (centered ** 2).sum(dim=-1, keepdim=True) / denom
            action_bias = centered / torch.sqrt(var + 1e-6)
        clip_value = float(getattr(self.conf, "gnn_bias_clip", 1.0))
        if clip_value > 0:
            action_bias = torch.clamp(action_bias, -clip_value, clip_value)
        weight = (
            float(weight_override)
            if weight_override is not None
            else float(getattr(self.conf, "gnn_action_bias_weight", 0.5))
        )
        action_bias = action_bias * weight
        action_bias = action_bias.masked_fill(~valid_mask, 0.0)
        bias[:, :, 0, :] = action_bias
        return bias

    def _graph_embedding_for_batch(self, encoder, episode_num, device):
        if not (self.use_gnn and self.use_gnn_graph_embedding):
            return None
        graph_embedding = encoder().to(device)
        return graph_embedding.view(1, 1, -1).expand(episode_num, self.n_agents, -1)

    def _get_inputs(self, batch, transition_idx):
        o = batch['o'][:, transition_idx]
        o_ = batch['o_'][:, transition_idx]
        u_onehot = batch['u_onehot'][:]

        episode_num = o.shape[0]
        inputs, inputs_ = [], []
        inputs.append(o)
        inputs_.append(o_)

        if self.conf.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_.append(u_onehot[:, transition_idx])

        if self.conf.reuse_network:
            agent_ids = torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            inputs.append(agent_ids)
            inputs_.append(agent_ids)

        if self.use_gnn and self.use_gnn_graph_embedding:
            inputs.append(self._graph_embedding_for_batch(self.eval_graph_encoder, episode_num, o.device))
            inputs_.append(self._graph_embedding_for_batch(self.target_graph_encoder, episode_num, o_.device))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)
        return inputs, inputs_

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.conf.save_frequency)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print("save model: {} epoch.".format(num))
        drqn_path = os.path.join(self.model_dir, f"{num}_drqn_net_params.pkl")
        mixer_path = os.path.join(self.model_dir, f"{num}_{self.mixer}_mixer_params.pkl")
        latest_drqn_path = os.path.join(self.model_dir, "latest_drqn_net_params.pkl")
        latest_mixer_path = os.path.join(self.model_dir, f"latest_{self.mixer}_mixer_params.pkl")
        self._save_state_dict(self.eval_drqn_net.state_dict(), drqn_path)
        self._save_state_dict(self.eval_mixer_net.state_dict(), mixer_path)
        self._save_state_dict(self.eval_drqn_net.state_dict(), latest_drqn_path)
        self._save_state_dict(self.eval_mixer_net.state_dict(), latest_mixer_path)
        if self.use_gnn:
            gnn_path = os.path.join(self.model_dir, f"{num}_gnn_encoder_params.pkl")
            latest_gnn_path = os.path.join(self.model_dir, "latest_gnn_encoder_params.pkl")
            self._save_state_dict(self.eval_graph_encoder.state_dict(), gnn_path)
            self._save_state_dict(self.eval_graph_encoder.state_dict(), latest_gnn_path)
        if self.use_si_predict_aux:
            predictor_path = os.path.join(self.model_dir, f"{num}_si_predictor_params.pkl")
            latest_predictor_path = os.path.join(self.model_dir, "latest_si_predictor_params.pkl")
            self._save_state_dict(self.si_predictor.state_dict(), predictor_path)
            self._save_state_dict(self.si_predictor.state_dict(), latest_predictor_path)
