import os
import re
import torch
import torch.nn.functional as F
from NN import DRQN, QMIXNET, QattenMixer


class QMIX:
    def __init__(self, conf):
        self.conf = conf
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_shape = self.conf.obs_shape
        self.mixer = self.conf.mixer.lower()
        input_shape = self.obs_shape

        if self.conf.last_action:
            input_shape += self.n_actions
        if self.conf.reuse_network:
            input_shape += self.n_agents

        self.eval_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, self.conf).to(self.device)

        mixer_cls = QattenMixer if self.mixer == "qatten" else QMIXNET
        self.eval_mixer_net = mixer_cls(self.conf).to(self.device)
        self.target_mixer_net = mixer_cls(self.conf).to(self.device)

        self.model_dir = self._model_roots()[0]
        self.conf.loaded_model_tag = None
        self.conf.loaded_drqn_path = None
        self.conf.loaded_mixer_path = None

        if self.conf.load_model:
            self._load_model_if_available()

        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())

        self.eval_parameters = list(self.eval_mixer_net.parameters()) + list(self.eval_drqn_net.parameters())
        if self.conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)

        self.eval_hidden = None
        self.target_hidden = None

        if getattr(self.conf, "verbose", False):
            print("init {} nets finished!".format(self.mixer))

    def _model_roots(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        configured_root = os.path.abspath(os.path.join(self.conf.model_dir, self.conf.map_name))
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
        legacy_qmix_pattern = re.compile(r"^(\d+)_qmix_net_params\.pkl$")
        requested_tag = str(getattr(self.conf, "model_tag", "latest") or "latest").strip()

        def add_candidate(tag, drqn_path, mixer_path):
            pair_key = (os.path.abspath(drqn_path), os.path.abspath(mixer_path))
            if pair_key in seen_pairs:
                return
            if os.path.exists(drqn_path) and os.path.exists(mixer_path):
                seen_pairs.add(pair_key)
                candidates.append((str(tag), drqn_path, mixer_path))

        for model_root in self._model_roots():
            if not os.path.isdir(model_root):
                continue

            latest_drqn = os.path.join(model_root, "latest_drqn_net_params.pkl")
            latest_mixer = os.path.join(model_root, f"latest_{self.mixer}_mixer_params.pkl")

            drqn_files = {}
            mixer_files = {}
            for filename in os.listdir(model_root):
                drqn_match = drqn_pattern.match(filename)
                if drqn_match:
                    drqn_files[drqn_match.group(1)] = os.path.join(model_root, filename)
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
                add_candidate("latest", latest_drqn, latest_mixer)
            else:
                exact_drqn = drqn_files.get(requested_tag)
                exact_mixer = mixer_files.get(requested_tag)
                if exact_drqn and exact_mixer:
                    add_candidate(requested_tag, exact_drqn, exact_mixer)
                add_candidate("latest", latest_drqn, latest_mixer)

            common_prefixes = sorted(
                set(drqn_files.keys()) & set(mixer_files.keys()),
                key=lambda item: int(item),
                reverse=True,
            )
            for prefix in common_prefixes:
                add_candidate(prefix, drqn_files[prefix], mixer_files[prefix])

        return candidates

    def _load_model_if_available(self):
        loaded = False
        for model_tag, drqn_path, mixer_path in self._candidate_model_pairs():
            if os.path.exists(drqn_path) and os.path.exists(mixer_path):
                self.eval_drqn_net.load_state_dict(self._load_state_dict(drqn_path))
                self.eval_mixer_net.load_state_dict(self._load_state_dict(mixer_path))
                print("successfully load models:", drqn_path, mixer_path)
                self.conf.loaded_model_tag = model_tag
                self.conf.loaded_drqn_path = drqn_path
                self.conf.loaded_mixer_path = mixer_path
                loaded = True
                break
        if not loaded:
            print("model files not found for {}, continue with random initialization.".format(self.mixer))

    def _load_state_dict(self, model_path):
        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(model_path, map_location=self.device)

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

        q_eval_logits, q_targets = self.get_q_values(batch, max_episode_len)
        q_evals = torch.gather(q_eval_logits, dim=3, index=u).squeeze(3)
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

        expert_weight = getattr(self.conf, "expert_action_loss_weight", 0.0)
        expert_table = getattr(self.conf, "expert_action_table", None)
        if expert_weight and expert_table:
            expert_actions = torch.tensor(expert_table, dtype=torch.long, device=self.device)
            station_ids = torch.arange(max_episode_len, device=self.device).view(1, -1)
            station_ids = station_ids.expand(episode_num, -1).clamp(0, expert_actions.shape[0] - 1)
            expert_targets = expert_actions[station_ids]
            expert_loss = F.cross_entropy(
                q_eval_logits.reshape(-1, self.n_actions),
                expert_targets.reshape(-1),
                reduction="none",
            ).view(episode_num, max_episode_len, self.n_agents)
            expert_mask = mask.expand(-1, -1, self.n_agents)
            loss = loss + expert_weight * (expert_loss * expert_mask).sum() / expert_mask.sum()

        print("*******开始训练({})*********".format(self.mixer), loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())

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
        torch.save(self.eval_drqn_net.state_dict(), drqn_path)
        torch.save(self.eval_mixer_net.state_dict(), mixer_path)
        torch.save(self.eval_drqn_net.state_dict(), latest_drqn_path)
        torch.save(self.eval_mixer_net.state_dict(), latest_mixer_path)
