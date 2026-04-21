import numpy as np
import torch
import os
from NN import DRQN, QMIXNET
# import wandb
# import sys
# sys.setrecursionlimit(100000) # 例如这里设置为十万

# wandb.init(project="qmix660")
class QMIX:
    def __init__(self, conf):
        self.conf = conf
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_shape = self.conf.obs_shape
        input_shape = self.obs_shape

        # print(self.device, self.n_actions, self.n_agents, self.state_shape, self.obs_shape, input_shape)

        # DRQN 鐨勫弬鏁?
        if self.conf.last_action:
            input_shape += self.n_actions
        if self.conf.reuse_network:
            input_shape += self.n_agents

        # NET
        self.eval_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, self.conf).to(self.device)



        self.eval_qmix_net = QMIXNET(self.conf).to(self.device)
        self.target_qmix_net = QMIXNET(self.conf).to(self.device)
        # wandb.watch(self.eval_qmix_net)

        self.model_dir = self.conf.model_dir + self.conf.map_name

        if self.conf.load_model:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_pairs = [
                (
                    os.path.join(base_dir, "models", "3m", "0_drqn_net_params.pkl"),
                    os.path.join(base_dir, "models", "3m", "0_qmix_net_params.pkl"),
                ),
                (
                    os.path.join(base_dir, "models", "3m", "02_drqn_net_params.pkl"),
                    os.path.join(base_dir, "models", "3m", "02_qmix_net_params.pkl"),
                ),
                (
                    os.path.join(base_dir, "models", "3m", "03_drqn_net_params.pkl"),
                    os.path.join(base_dir, "models", "3m", "03_qmix_net_params.pkl"),
                ),
            ]
            map_location = "cuda:2" if self.conf.cuda else "cpu"
            loaded = False
            for drqn_path, qmix_path in candidate_pairs:
                if os.path.exists(drqn_path) and os.path.exists(qmix_path):
                    self.eval_drqn_net.load_state_dict(torch.load(drqn_path, map_location=map_location))
                    self.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=map_location))
                    print("successfully load models:", drqn_path, qmix_path)
                    loaded = True
                    break
            if not loaded:
                print("model files not found, continue with random initialization.")

        # copy eval net params to target net
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_drqn_net.parameters())
        if self.conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)

        # 学习时，为每个agent维护一个eval_hidden, target_hidden
        self.eval_hidden = None
        self.target_hidden = None

        print("init qmix nets finished!")

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        """
        batch: train data, obs: (batch_size, episode_limit, n_agents, obs_shape),(64, -53- ,3,42)
        max_episode_len: max episode length
        train_step: step record for updating target network parameters 
        """
        episode_num = batch['o'].shape[0]
        print("***********",max_episode_len)
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_, u, r, terminated = batch['s'], batch['s_'], batch['u'], batch['r'], batch['terminated']

        mask = 1 - batch['padded'].float()  # 把填充经验的TD-error置0，防止影响学习，padded都是0



        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len, n_agents, n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        # print("当前的q值是",q_evals)
        # print("当前的q值是", q_targets)
        s = s.to(self.device)
        u = u.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)

        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        # print("q_evals1 shape: ", q_evals.size()) #[batch_size, max_episode_len, n_agents, n_actions]
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        # q_targets[avail_u_ == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]
        # print(q_evals.size())###[32,4,2]
        # print(q_targets.size())###[32,4,2]
        # print(torch.max(r,dim=1))###[32,4,1]
        # print("q_evals2 shape: ", q_evals.size()) # [batch_size, max_episode_len, n_agents]找到了最大的那个
        r1 = torch.max(r,dim=1)[0].numpy()
        # a = -min(r1)
        a = r1[3]
        print("r",r1)
        print("a",np.mean(r1))
        # MSELoss函数的具体使用方法如下所示，其中MSELoss函数的参数均为默认参数。
        losshi = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        losshi = losshi(q_evals, q_targets)
        # wandb.log({"losshi": losshi})
        # wandb.log({"rewards": a})


        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_)

        print(q_total_eval.size())###[32,4,1]


        targets = r + self.conf.gamma * q_total_target * (1 - terminated)
        # targets = r
        td_error = (q_total_eval-targets.detach())
        mask_td_error = mask * td_error

        loss = (mask_td_error**2).sum() / mask.sum()
        # loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        # loss = loss(q_total_eval, targets)
        # wandb.log({"rewards": torch.mean(a)})


        print("*******开始训练*********", loss)
        # wandb.log({"loss":loss})


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_ = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id

            inputs = inputs.to(self.device)  # [batch_size*n_agents, obs_shape+n_agents+n_actions]
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)
            q_eval, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden)  # (n_agents, n_actions)
            q_target, self.target_hidden = self.target_drqn_net(inputs_, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)  #(batch_size, n_agents, n_actions)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的维度是(episode个数, n_agents, n_actions)
        # 把该列表转化成(batch_size, max_episode_len, n_agents, n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_inputs(self, batch, transition_idx):
        o, o_, u_onehot = batch['o'][:, transition_idx], batch['o_'][:, transition_idx], batch[
            'u_onehot'][:]  # u_onehot取全部，要用上一条

        episode_num = o.shape[0]  # episode_num=batch_size=32

        inputs, inputs_ = [], []
        inputs.append(o)
        inputs_.append(o_)

        # 给obs添加上一个动作、agent编号
        if self.conf.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_.append(u_onehot[:, transition_idx])

        if self.conf.reuse_network:  #### 对所有的智能体用一个网络
            """
            因为当前的obs是三维数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应向量即可
            比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            """
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 把batch_size、n_agents个agent的obs拼起来
        # 因为这里所有agent共享一个神经网络，每条数据都带了自己的编号，所以仍然是自己的数据
        # (batch_size, n_agents, n_actions) -> (batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)
        # print("处理后的inputs为", inputs)
        return inputs, inputs_

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.conf.save_frequency)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print("save model: {} epoch.".format(num))
        torch.save(self.eval_drqn_net.state_dict(), self.model_dir + '/' + num + '3_drqn_net_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '3_qmix_net_params.pkl')
