import torch.nn as nn
import torch
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_shape, conf):
        super(DRQN, self).__init__()
        self.conf = conf
        self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
        self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
        self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class ProcedureGraphEncoder(nn.Module):
    def __init__(self, adjacency, node_features, conf):
        super(ProcedureGraphEncoder, self).__init__()
        self.conf = conf
        hidden_dim = conf.gnn_hidden_dim
        embed_dim = conf.gnn_embed_dim
        self.layers = max(1, int(getattr(conf, "gnn_layers", 2)))

        self.register_buffer("adjacency", adjacency.float())
        self.register_buffer("node_features", node_features.float())
        self.input_proj = nn.Linear(self.node_features.size(1), hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.node_scorer = nn.Linear(embed_dim, 1)

    def node_embeddings(self):
        x = F.relu(self.input_proj(self.node_features))
        for _ in range(self.layers):
            x = torch.matmul(self.adjacency, x)
            x = F.relu(self.hidden_proj(x))
        return self.output_proj(x)

    def node_scores(self):
        return self.node_scorer(self.node_embeddings()).squeeze(-1)

    def forward(self):
        return self.node_embeddings().mean(dim=0)


class ComboActionScorer(nn.Module):
    def __init__(self, conf):
        super(ComboActionScorer, self).__init__()
        self.conf = conf
        embed_dim = conf.gnn_embed_dim
        hidden_dim = getattr(conf, "combo_scorer_hidden_dim", 128)
        self.n_actions = conf.n_actions
        self.team_count = int(getattr(conf, "combo_team_count", 1))
        self.pair_feature_dim = int(getattr(conf, "combo_pair_feature_dim", 14))
        self.team_embedding = nn.Embedding(self.team_count, embed_dim)
        self.state_proj = nn.Linear(conf.state_shape, embed_dim)
        input_dim = embed_dim * 4 + self.pair_feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        team_ids = torch.arange(self.n_actions, dtype=torch.long) % self.team_count
        self.register_buffer("combo_action_team_ids", team_ids)

    def forward(self, node_embeddings, states, pair_features, proc_action_indices):
        batch_size, episode_len, action_count, feature_dim = pair_features.shape
        if action_count != self.n_actions or feature_dim != self.pair_feature_dim:
            raise ValueError("invalid combo pair feature shape")

        proc_emb = node_embeddings[proc_action_indices].view(1, 1, action_count, -1)
        proc_emb = proc_emb.expand(batch_size, episode_len, -1, -1)
        team_emb = self.team_embedding(self.combo_action_team_ids).view(1, 1, action_count, -1)
        team_emb = team_emb.expand(batch_size, episode_len, -1, -1)
        global_emb = node_embeddings.mean(dim=0).view(1, 1, 1, -1)
        global_emb = global_emb.expand(batch_size, episode_len, action_count, -1)
        state_emb = F.relu(self.state_proj(states.reshape(-1, states.shape[-1])))
        state_emb = state_emb.view(batch_size, episode_len, 1, -1).expand(
            batch_size, episode_len, action_count, -1
        )
        scorer_input = torch.cat([proc_emb, team_emb, global_emb, state_emb, pair_features], dim=-1)
        scores = self.net(scorer_input).squeeze(-1)
        return scores.unsqueeze(2)


class StationTimePredictor(nn.Module):
    def __init__(self, input_shape, conf):
        super(StationTimePredictor, self).__init__()
        hidden_dim = getattr(conf, "si_predict_hidden_dim", 64)
        output_dim = getattr(conf, "si_predict_output_dim", 4)
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.softplus(self.net(x))


class QMIXNET(nn.Module):
    def __init__(self, conf):
        super(QMIXNET, self).__init__()
        """
        生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
        因此先生成一个（行*列）的向量，再reshape
        """
        # print(conf.state_shape)
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.conf.hyper_hidden_dim, self.conf.n_agents*self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents*self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim*1)
        
        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.conf.qmix_hidden_dim, 1))

    # input: (batch_size, n_agents, qmix_hidden_dim)
    # q_values: (episode_num, max_episode_len, n_agents)
    # states shape: (episode_num, max_episode_len, state_shape)
    def forward(self, q_values, states):
        # print(self.conf.state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        # print("整理后的q",q_values)###[128,1,2],1.5左右
        states = states.reshape(-1, self.conf.state_shape)
        # print(states)###【128,8】

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.relu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total


class QattenMixer(nn.Module):
    def __init__(self, conf):
        super(QattenMixer, self).__init__()
        self.conf = conf
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.n_attention_heads = conf.n_attention_heads
        self.qatten_hidden_dim = conf.qatten_hidden_dim

        self.state_proj = nn.Linear(self.state_shape, self.n_attention_heads * self.qatten_hidden_dim)
        self.agent_proj = nn.Embedding(self.n_agents, self.n_attention_heads * self.qatten_hidden_dim)
        self.attn_proj = nn.Linear(self.qatten_hidden_dim, 1)
        self.head_weight = nn.Sequential(
            nn.Linear(self.state_shape, self.qatten_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qatten_hidden_dim, self.n_attention_heads),
        )
        self.state_bias = nn.Sequential(
            nn.Linear(self.state_shape, self.qatten_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qatten_hidden_dim, 1)
        )
        self.register_buffer("agent_ids", torch.arange(self.n_agents, dtype=torch.long))

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        flat_q = q_values.reshape(-1, self.n_agents, 1)
        flat_states = states.reshape(-1, self.state_shape)

        state_context = self.state_proj(flat_states).view(
            -1, 1, self.n_attention_heads, self.qatten_hidden_dim
        )
        agent_context = self.agent_proj(self.agent_ids).view(
            1, self.n_agents, self.n_attention_heads, self.qatten_hidden_dim
        )

        attn_logits = self.attn_proj(torch.tanh(state_context + agent_context)).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)

        head_q = (attn_weights * flat_q.expand(-1, -1, self.n_attention_heads)).sum(dim=1)
        head_weight = F.softplus(self.head_weight(flat_states))
        q_total = (head_weight * head_q).sum(dim=1, keepdim=True) + self.state_bias(flat_states)
        return q_total.view(episode_num, -1, 1)
