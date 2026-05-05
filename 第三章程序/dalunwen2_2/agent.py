import numpy as np
import torch
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

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (42,) -> (1,42)

        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn_net(inputs, hidden_state)
        action_bias = self.policy.get_eval_action_bias(order_candidates, agent_num)
        if action_bias is not None:
            q_value = q_value + action_bias.unsqueeze(0)
        diagnostic_record = None
        if getattr(self.conf, "record_gnn_diagnostics", False) and self.policy.use_gnn:
            normal_graph = self.policy.get_eval_graph_embedding_numpy(force_zero=False)
            zero_graph = self.policy.get_eval_graph_embedding_numpy(force_zero=True)
            normal_inputs = np.hstack((base_inputs, normal_graph))
            zero_inputs = np.hstack((base_inputs, zero_graph))
            normal_inputs = torch.tensor(normal_inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
            zero_inputs = torch.tensor(zero_inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_normal, _ = self.policy.eval_drqn_net(normal_inputs, hidden_state)
                q_zero, _ = self.policy.eval_drqn_net(zero_inputs, hidden_state)
                normal_bias = self.policy.get_eval_action_bias(order_candidates, agent_num, force_zero=False)
                if normal_bias is not None:
                    q_normal = q_normal + normal_bias.unsqueeze(0)
                zero_bias = self.policy.get_eval_action_bias(order_candidates, agent_num, force_zero=True)
                if zero_bias is not None:
                    q_zero = q_zero + zero_bias.unsqueeze(0)
                if availible_actions.ndim == 1:
                    invalid_mask = torch.tensor(availible_actions == 0.0, dtype=torch.bool, device=self.device).unsqueeze(0)
                    q_normal = q_normal.masked_fill(invalid_mask, -float("inf"))
                    q_zero = q_zero.masked_fill(invalid_mask, -float("inf"))
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
                gnn_bias = normal_bias if normal_bias is not None else torch.zeros(self.n_actions, device=self.device)
                max_abs_bias = float(torch.max(torch.abs(gnn_bias)).item())
                rule_node_ids = self.policy.get_rule_node_ids(order_candidates)

                def q_list(values):
                    result = []
                    for item in values.view(-1).detach().cpu().tolist():
                        result.append(round(float(item), 6) if np.isfinite(item) else None)
                    return result

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
                    "valid_order_count": len(order_candidates or []),
                    "q_zero": q_list(q_zero),
                    "gnn_bias": q_list(gnn_bias),
                    "q_normal": q_list(q_normal),
                }
                if decision_context:
                    diagnostic_record.update(decision_context)
                records = getattr(self.conf, "gnn_diagnostic_records", None)
                if records is not None:
                    records.append(diagnostic_record)
        # choose action form q value

        if availible_actions.ndim == 1:
            invalid_mask = torch.tensor(availible_actions == 0.0, dtype=torch.bool, device=self.device).unsqueeze(0)
            q_value = q_value.masked_fill(invalid_mask, -float("inf"))
        if (not evaluate) and np.random.uniform() < epsilon:
            action = int(np.random.choice(availible_actions_idx))
        else:
            action = int(torch.argmax(q_value).item())
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

    def train(self, batch, train_step, epsilon=None):
        # 不同的episode的数据长度不同，因此需要得到最大长度
        max_episode_len = self._get_max_episode_len(batch)
        # print("最大长度是",max_episode_len)
        max_episode_len = 4
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]

        completed_train_step = train_step + 1
        self.policy.learn(batch, max_episode_len, completed_train_step, epsilon)
        if completed_train_step % self.conf.save_frequency == 0:
            self.policy.save_model(completed_train_step)

 
