import numpy as np
import torch
from load_rerank import select_load_rerank_action
from policy import QMIX
from torch.distributions import Categorical

class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions 
        self.n_agents = conf.n_agents 
        self.state_shape = conf.state_shape 
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit

        self.policy = QMIX(conf)

        if getattr(self.conf, "verbose", False):
            print("Agents inited!")

    def _mask_unavailable_actions(self, q_value, availible_actions):
        if availible_actions.ndim != 1:
            return q_value
        invalid_mask = torch.tensor(
            availible_actions == 0.0, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        return q_value.masked_fill(invalid_mask, -float("inf"))

    def _load_penalty_tensor(self, load_penalty):
        if load_penalty is None or not getattr(self.conf, "gnn_use_load_penalty", False):
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        if getattr(self.conf, "zero_gnn_embedding", False):
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        penalty = torch.tensor(load_penalty, dtype=torch.float32, device=self.device).view(-1)
        if penalty.numel() != self.n_actions:
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        penalty = torch.where(torch.isfinite(penalty), penalty, torch.zeros_like(penalty))
        return penalty

    def _si_penalty_tensor(self, predicted_si):
        if predicted_si is None or not getattr(self.conf, "use_si_predict_rerank", False):
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        if getattr(self.conf, "zero_gnn_embedding", False):
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        values = predicted_si.to(self.device).view(-1)
        if values.numel() != self.n_actions:
            return torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        return values

    def _normalise_si_penalty(self, si_values, mask):
        penalty = torch.zeros_like(si_values)
        if not bool(mask.any().item()):
            return penalty
        valid = si_values[mask]
        if valid.numel() <= 1 or float(torch.std(valid).item()) <= 1e-8:
            return penalty
        penalty[mask] = (valid - torch.mean(valid)) / (torch.std(valid) + 1e-6)
        return penalty

    def _action_fusion(self, base_q, raw_action_bias, agent_num, load_penalty=None, predicted_si=None):
        mode = getattr(self.conf, "gnn_action_fusion_mode", "add_bias")
        load_penalty_values = self._load_penalty_tensor(load_penalty)
        load_penalty_weight = float(getattr(self.conf, "gnn_load_penalty_weight", 0.0))
        predicted_si_values = self._si_penalty_tensor(predicted_si)
        si_penalty_weight = float(getattr(self.conf, "si_predict_penalty_weight", 0.0))
        info = {
            "fusion_mode": mode,
            "rerank_candidate_size": None,
            "rerank_candidate_actions": None,
            "rerank_margin_threshold": None,
            "rerank_topk": None,
            "load_penalty_weight": load_penalty_weight,
            "load_penalty_values": self._q_list(load_penalty_values),
            "si_penalty_weight": si_penalty_weight,
            "predicted_si_by_action": self._q_list(predicted_si) if predicted_si is not None else [None] * self.n_actions,
            "si_penalty_by_action": [None] * self.n_actions,
        }
        if raw_action_bias is None or agent_num != 0:
            empty_bias = torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
            return base_q, empty_bias, info

        raw_action_bias = raw_action_bias.to(self.device)
        if mode == "add_bias":
            weight = float(getattr(self.conf, "gnn_action_bias_weight", 0.5))
        else:
            weight = float(getattr(self.conf, "gnn_rerank_weight", 0.5))
        action_bias = raw_action_bias * weight
        action_bias = action_bias - load_penalty_values * load_penalty_weight

        q_flat = base_q.view(-1)
        finite_mask = torch.isfinite(q_flat)
        if mode == "add_bias":
            si_penalty_values = self._normalise_si_penalty(predicted_si_values, finite_mask)
            action_bias = action_bias - si_penalty_values * si_penalty_weight
            info["si_penalty_by_action"] = self._q_list(si_penalty_values)

        if mode == "add_bias":
            candidate_actions = torch.nonzero(finite_mask, as_tuple=False).view(-1)
            info["rerank_candidate_size"] = int(candidate_actions.numel())
            info["rerank_candidate_actions"] = [int(item) for item in candidate_actions.cpu().tolist()]
            return base_q + action_bias.unsqueeze(0), action_bias, info

        if not bool(finite_mask.any().item()):
            return base_q, action_bias, info

        if mode == "margin_gated":
            threshold = float(getattr(self.conf, "gnn_margin_threshold", 0.5))
            top_q = torch.max(q_flat[finite_mask])
            candidate_mask = finite_mask & ((top_q - q_flat) <= threshold + 1e-8)
            info["rerank_margin_threshold"] = threshold
        elif mode == "topk_rerank":
            finite_count = int(finite_mask.sum().item())
            k = max(1, min(int(getattr(self.conf, "gnn_topk", 2)), finite_count))
            top_indices = torch.topk(q_flat.masked_fill(~finite_mask, -float("inf")), k=k).indices
            candidate_mask = torch.zeros_like(finite_mask, dtype=torch.bool)
            candidate_mask[top_indices] = True
            info["rerank_topk"] = k
        else:
            candidate_actions = torch.nonzero(finite_mask, as_tuple=False).view(-1)
            info["fusion_mode"] = "add_bias"
            info["rerank_candidate_size"] = int(candidate_actions.numel())
            info["rerank_candidate_actions"] = [int(item) for item in candidate_actions.cpu().tolist()]
            return base_q + action_bias.unsqueeze(0), action_bias, info

        fused_flat = torch.full_like(q_flat, -float("inf"))
        si_penalty_values = self._normalise_si_penalty(predicted_si_values, candidate_mask)
        candidate_bias = action_bias.clone()
        candidate_bias[candidate_mask] = (
            candidate_bias[candidate_mask] - si_penalty_values[candidate_mask] * si_penalty_weight
        )
        fused_flat[candidate_mask] = q_flat[candidate_mask] + candidate_bias[candidate_mask]
        candidate_actions = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
        info["rerank_candidate_size"] = int(candidate_actions.numel())
        info["rerank_candidate_actions"] = [int(item) for item in candidate_actions.cpu().tolist()]
        info["si_penalty_by_action"] = self._q_list(si_penalty_values)
        return fused_flat.view_as(base_q), candidate_bias, info

    def _q_list(self, values):
        result = []
        for item in values.view(-1).detach().cpu().tolist():
            result.append(round(float(item), 6) if np.isfinite(item) else None)
        return result

    def choose_action(
        self,
        obs,
        last_action,
        agent_num,
        availible_actions,
        epsilon,
        evaluate=False,
        order_candidates=None,
        decision_context=None,
    ):
        inputs = obs[:]
        # print(availible_actions)
        availible_actions = np.asarray(availible_actions)
        if availible_actions.ndim == 1:
            availible_actions_idx = np.nonzero(availible_actions)[0]
        else:
            availible_actions_idx = np.arange(self.n_actions)
        agents_id = np.zeros(self.n_agents)
        agents_id[agent_num] = 1.

        if self.conf.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.conf.reuse_network:
            inputs = np.hstack((inputs, agents_id))
        base_inputs = inputs.copy()
        graph_embedding = self.policy.get_eval_graph_embedding_numpy()
        if graph_embedding is not None:
            inputs = np.hstack((inputs, graph_embedding))
        hidden_state = self.policy.eval_hidden[:, agent_num, :].clone()
        load_penalty = None
        predicted_si = None
        combo_q_value = None
        if decision_context:
            load_penalty = decision_context.get("load_penalty_by_action")
            predicted_si = self.policy.predict_si_values_for_action_features(
                obs, decision_context.get("si_predict_features_by_action")
            )
            combo_q_value = self.policy.get_eval_combo_q_values(
                obs, decision_context.get("combo_pair_features_by_action")
            )

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (42,) -> (1,42)

        if combo_q_value is None:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn_net(inputs, hidden_state)
            q_value = self._mask_unavailable_actions(q_value, availible_actions)
            raw_action_bias = self.policy.get_eval_action_bias(
                order_candidates, agent_num, weight_override=1.0, avail_mask=availible_actions
            )
            q_value, action_bias, fusion_info = self._action_fusion(
                q_value,
                raw_action_bias,
                agent_num,
                load_penalty=load_penalty,
                predicted_si=predicted_si,
            )
        else:
            q_value = self._mask_unavailable_actions(combo_q_value, availible_actions)
            action_bias = torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
            fusion_info = {}
        load_rerank_action = None
        load_rerank_record = None
        if (
            getattr(self.conf, "use_qatten_load_rerank", False)
            and evaluate
            and load_penalty is not None
        ):
            avail_mask = availible_actions if availible_actions.ndim == 1 else None
            load_rerank_action, load_rerank_info = select_load_rerank_action(
                q_value.detach().cpu().numpy().reshape(-1),
                load_penalty=load_penalty,
                agent_num=agent_num,
                mode=getattr(self.conf, "qatten_load_rerank_mode", "margin_gated"),
                margin_threshold=getattr(self.conf, "qatten_load_margin_threshold", 0.5),
                topk=getattr(self.conf, "qatten_load_topk", 2),
                penalty_weight=getattr(self.conf, "qatten_load_penalty_weight", 0.0),
                avail_mask=avail_mask,
            )
            if (
                agent_num == 0
                and getattr(self.conf, "record_load_rerank_diagnostics", False)
            ):
                load_rerank_record = dict(load_rerank_info)
                load_rerank_record["agent_id"] = int(agent_num)
                load_rerank_record["penalty_weight"] = float(
                    getattr(self.conf, "qatten_load_penalty_weight", 0.0)
                )
                load_rerank_record["margin_threshold"] = float(
                    getattr(self.conf, "qatten_load_margin_threshold", 0.5)
                )
                load_rerank_record["topk"] = int(getattr(self.conf, "qatten_load_topk", 2))
                if decision_context:
                    load_rerank_record.update(decision_context)
        diagnostic_record = None
        if (
            getattr(self.conf, "record_gnn_diagnostics", False)
            and self.policy.use_gnn
            and not getattr(self.policy, "use_combo_scorer", False)
        ):
            normal_graph = self.policy.get_eval_graph_embedding_numpy(force_zero=False)
            zero_graph = self.policy.get_eval_graph_embedding_numpy(force_zero=True)
            normal_inputs = np.hstack((base_inputs, normal_graph)) if normal_graph is not None else base_inputs
            zero_inputs = np.hstack((base_inputs, zero_graph)) if zero_graph is not None else base_inputs
            normal_inputs = torch.tensor(normal_inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
            zero_inputs = torch.tensor(zero_inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_normal, _ = self.policy.eval_drqn_net(normal_inputs, hidden_state)
                q_zero, _ = self.policy.eval_drqn_net(zero_inputs, hidden_state)
                q_normal = self._mask_unavailable_actions(q_normal, availible_actions)
                q_zero = self._mask_unavailable_actions(q_zero, availible_actions)
                normal_raw_bias = self.policy.get_eval_action_bias(
                    order_candidates,
                    agent_num,
                    force_zero=False,
                    weight_override=1.0,
                    avail_mask=availible_actions,
                )
                q_normal, gnn_bias, diagnostic_fusion_info = self._action_fusion(
                    q_normal,
                    normal_raw_bias,
                    agent_num,
                    load_penalty=load_penalty,
                    predicted_si=predicted_si,
                )
                normal_action = int(torch.argmax(q_normal).item())
                zero_action = int(torch.argmax(q_zero).item())
                q_delta = torch.abs(q_normal - q_zero)
                finite_delta = q_delta[torch.isfinite(q_delta)]
                mean_abs_q_delta = (
                    float(torch.mean(finite_delta).item()) if finite_delta.numel() else 0.0
                )
                max_abs_q_delta = (
                    float(torch.max(finite_delta).item()) if finite_delta.numel() else 0.0
                )
                zero_top2 = torch.topk(q_zero.view(-1), k=min(2, self.n_actions)).values
                q_margin_zero = (
                    float((zero_top2[0] - zero_top2[1]).item()) if zero_top2.numel() > 1 else 0.0
                )
                q_delta_to_margin = max_abs_q_delta / max(abs(q_margin_zero), 1e-6)
                max_abs_bias = float(torch.max(torch.abs(gnn_bias)).item())
                rule_node_ids = self.policy.get_rule_node_ids(order_candidates)
                action_target_values = self.policy.get_rule_target_values(rule_node_ids)
                candidate_stats = self.policy.get_candidate_action_stats(order_candidates, gnn_bias)
                changed_action_q_gap = None
                changed_action_bias_gain = None
                changed_action_target_gain = None
                changed_action_load_penalty_gain = None
                changed_action_pred_si_gain = None
                selected_pred_si_lower_than_original = None
                selected_successor_time_higher = None
                if normal_action != zero_action:
                    q_values_zero = q_zero.view(-1)
                    changed_action_q_gap = float(
                        (q_values_zero[zero_action] - q_values_zero[normal_action]).item()
                    )
                    changed_action_bias_gain = float(
                        (gnn_bias[normal_action] - gnn_bias[zero_action]).item()
                    )
                    original_target = action_target_values[zero_action]
                    selected_target = action_target_values[normal_action]
                    if original_target is not None and selected_target is not None:
                        changed_action_target_gain = float(selected_target - original_target)
                        selected_successor_time_higher = bool(changed_action_target_gain > 0)
                    load_penalty_values = diagnostic_fusion_info.get("load_penalty_values") or []
                    if (
                        0 <= zero_action < len(load_penalty_values)
                        and 0 <= normal_action < len(load_penalty_values)
                        and load_penalty_values[zero_action] is not None
                        and load_penalty_values[normal_action] is not None
                    ):
                        changed_action_load_penalty_gain = float(
                            load_penalty_values[zero_action]
                            - load_penalty_values[normal_action]
                        )
                    predicted_si_values = diagnostic_fusion_info.get("predicted_si_by_action") or []
                    if (
                        0 <= zero_action < len(predicted_si_values)
                        and 0 <= normal_action < len(predicted_si_values)
                        and predicted_si_values[zero_action] is not None
                        and predicted_si_values[normal_action] is not None
                    ):
                        changed_action_pred_si_gain = float(
                            predicted_si_values[zero_action] - predicted_si_values[normal_action]
                        )
                        selected_pred_si_lower_than_original = bool(changed_action_pred_si_gain > 0)

                diagnostic_record = {
                    "agent_id": int(agent_num),
                    "mean_abs_q_delta": mean_abs_q_delta,
                    "max_abs_q_delta": max_abs_q_delta,
                    "q_changed": int(mean_abs_q_delta > 1e-8),
                    "action_changed": int(normal_action != zero_action),
                    "normal_action": normal_action,
                    "zero_action": zero_action,
                    "q_margin_zero": q_margin_zero,
                    "max_abs_bias": max_abs_bias,
                    "q_delta_to_margin": q_delta_to_margin,
                    "rule_node_ids": rule_node_ids,
                    "action_target_values": action_target_values,
                    "valid_order_count": len(order_candidates or []),
                    "q_zero": self._q_list(q_zero),
                    "gnn_bias": self._q_list(gnn_bias),
                    "q_normal": self._q_list(q_normal),
                    "original_action": zero_action,
                    "rerank_action": normal_action,
                    "changed_action_q_gap": changed_action_q_gap,
                    "changed_action_bias_gain": changed_action_bias_gain,
                    "changed_action_target_gain": changed_action_target_gain,
                    "changed_action_load_penalty_gain": changed_action_load_penalty_gain,
                    "changed_action_pred_si_gain": changed_action_pred_si_gain,
                    "selected_pred_si_lower_than_original": selected_pred_si_lower_than_original,
                    "selected_successor_time_higher_than_original": selected_successor_time_higher,
                }
                diagnostic_record.update(diagnostic_fusion_info)
                diagnostic_record.update(candidate_stats)
                if decision_context:
                    diagnostic_record.update(decision_context)
                records = getattr(self.conf, "gnn_diagnostic_records", None)
                if records is not None:
                    records.append(diagnostic_record)
        # choose action form q value

        if (not evaluate) and np.random.uniform() < epsilon:
            action = int(np.random.choice(availible_actions_idx))
        elif load_rerank_action is not None:
            action = int(load_rerank_action)
        else:
            action = int(torch.argmax(q_value).item())
        if load_rerank_record is not None:
            load_rerank_record["executed_action"] = int(action)
            load_rerank_record["executed_action_changed"] = int(
                action != load_rerank_record["original_action"]
            )
            records = getattr(self.conf, "load_rerank_records", None)
            if records is not None:
                records.append(load_rerank_record)
        if diagnostic_record is not None:
            diagnostic_record["executed_action"] = int(action)
            diagnostic_record["executed_action_changed"] = int(action != diagnostic_record["zero_action"])
            rule_node_ids = diagnostic_record.get("rule_node_ids") or []
            if 0 <= action < len(rule_node_ids):
                diagnostic_record["selected_rule_node_id"] = int(rule_node_ids[action])
            else:
                diagnostic_record["selected_rule_node_id"] = -1
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx+1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(
        self,
        batch,
        train_step,
        epsilon=None,
        expert_batch=None,
        td_weight=1.0,
        imitation_weight=0.0,
    ):
        # 不同的episode的数据长度不同，因此需要得到最大长度
        max_episode_len = self._get_max_episode_len(batch)
        # print("最大长度是",max_episode_len)
        if getattr(self.conf, "action_mode", "rule") != "combo_proc_team":
            max_episode_len = 4
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]

        if expert_batch is not None:
            expert_max_episode_len = self._get_max_episode_len(expert_batch)
            if getattr(self.conf, "action_mode", "rule") != "combo_proc_team":
                expert_max_episode_len = 4
            for key in expert_batch.keys():
                expert_batch[key] = expert_batch[key][:, :expert_max_episode_len]

        completed_train_step = train_step + 1
        self.policy.learn(
            batch,
            max_episode_len,
            completed_train_step,
            epsilon,
            expert_batch=expert_batch,
            td_weight=td_weight,
            imitation_weight=imitation_weight,
        )
        if completed_train_step % self.conf.save_frequency == 0:
            self.policy.save_model(completed_train_step)

 
