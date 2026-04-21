import numpy as np
import torch
import os
from NN import DRQN, QMIXNET
# import wandb
# import sys
# sys.setrecursionlimit(100000) #渚嬪杩欓噷璁剧疆涓哄崄涓?

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
                    os.path.join(base_dir, "1_drqn_net_params.pkl"),
                    os.path.join(base_dir, "1_qmix_net_params.pkl"),
                ),
                (
                    os.path.join(base_dir, "0_drqn_net_params.pkl"),
                    os.path.join(base_dir, "0_qmix_net_params.pkl"),
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

        # 瀛︿範鏃讹紝涓烘瘡涓猘gent缁存姢涓€涓猠val_hidden, target_hidden
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

        mask = 1 - batch['padded'].float()  # 鎶婂～鍏呯粡楠岀殑TD-error缃?锛岄槻姝㈠奖鍝嶅涔?padded閮芥槸0



        # 寰楀埌姣忎釜agent瀵瑰簲鐨凲鍊硷紝缁村害涓?episode涓暟, max_episode_len锛?n_agents锛?n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        # print("褰撳墠鐨剄鍊兼槸",q_evals)
        # print("褰撳墠鐨剄鍊兼槸", q_targets)
        s = s.to(self.device)
        u = u.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)

        # 鍙栨瘡涓猘gent鍔ㄤ綔瀵瑰簲鐨凲鍊硷紝骞朵笖鎶婃渶鍚庝笉闇€瑕佺殑涓€缁村幓鎺夛紝鍥犱负鏈€鍚庝竴缁村彧鏈変竴涓€间簡
        # print("q_evals1 shape: ", q_evals.size()) #[batch_size, max_episode_len, n_agents, n_actions]
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        # q_targets[avail_u_ == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]
        # print(q_evals.size())###[32,4,2]
        # print(q_targets.size())###[32,4,2]
        # print(torch.max(r,dim=1))###[32,4,1]
        # print("q_evals2 shape: ", q_evals.size()) # [batch_size, max_episode_len, n_agents]鎵惧埌浜嗘渶澶х殑閭ｄ釜
        r1 = torch.max(r,dim=1)[0].numpy()
        # a = -min(r1)
        a = r1[3]
        print("r",r1)
        print("a",np.mean(r1))
        # MSELoss鍑芥暟鐨勫叿浣撲娇鐢ㄦ柟娉曞涓嬫墍绀猴紝鍏朵腑MSELoss鍑芥暟鐨勫弬鏁板潎涓洪粯璁ゅ弬鏁般€?
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


        print("*******寮€濮嬭缁?*********",loss)
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
            inputs, inputs_ = self._get_inputs(batch, transition_idx)  # 缁檕bs鍔爈ast_action銆乤gent_id

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

        # 寰楃殑q_eval鍜宷_target鏄竴涓垪琛紝鍒楄〃閲岃鐫€max_episode_len涓暟缁勶紝鏁扮粍鐨勭殑缁村害鏄?episode涓暟, n_agents锛宯_actions)
        # 鎶婅鍒楄〃杞寲鎴?batch_size, max_episode_len锛?n_agents锛宯_actions)鐨勬暟缁?
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_inputs(self, batch, transition_idx):
        o, o_, u_onehot = batch['o'][:, transition_idx], batch['o_'][:, transition_idx], batch[
            'u_onehot'][:]  # u_onehot鍙栧叏閮紝瑕佺敤涓婁竴鏉?

        episode_num = o.shape[0]  # episode_num=batch_size=32

        inputs, inputs_ = [], []
        inputs.append(o)
        inputs_.append(o_)

        # 缁檕bs娣诲姞涓婁竴涓姩浣溿€乤gent缂栧彿
        if self.conf.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_.append(u_onehot[:, transition_idx])

        if self.conf.reuse_network:####瀵规墍鏈夌殑鏅鸿兘浣撶敤涓€涓綉缁?
            """
            鍥犱负褰撳墠鐨刼bs涓夌淮鐨勬暟鎹紝姣忎竴缁村垎鍒唬琛?episode缂栧彿锛宎gent缂栧彿锛宱bs缁村害)锛岀洿鎺ュ湪dim_1涓婃坊鍔犲搴旂殑鍚戦噺
            鍗冲彲锛屾瘮濡傜粰agent_0鍚庨潰鍔?1, 0, 0, 0, 0)锛岃〃绀?涓猘gent涓殑0鍙枫€傝€宎gent_0鐨勬暟鎹濂藉湪绗?琛岋紝閭ｄ箞闇€瑕佸姞鐨?
            agent缂栧彿鎭板ソ灏辨槸涓€涓崟浣嶇煩闃碉紝鍗冲瑙掔嚎涓?锛屽叾浣欎负0
            """
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 鎶奲atch_size銆乶_agents涓猘gent鐨刼bs鎷艰捣鏉ワ紝
        # 鍥犱负杩欓噷鎵€鏈塧gent鍏变韩涓€涓缁忕綉缁滐紝姣忔潯鏁版嵁涓甫涓婁簡鑷繁鐨勭紪鍙凤紝鎵€浠ヨ繕鏄嚜宸辩殑鏁版嵁
        # (batch_size, n_agents, n_actions) -> (batch_size*n_agents, n_actions)
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)
        # print("澶勭悊鍚庣殑inputs涓?,inputs)
        return inputs, inputs_

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.conf.save_frequency)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print("save model: {} epoch.".format(num))
        torch.save(self.eval_drqn_net.state_dict(), self.model_dir + '/' + num + '_drqn_net_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
