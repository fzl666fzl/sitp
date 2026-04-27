import os
import torch


class Config:
    def __init__(self):
        self.train = True
        self.seed = 133
        self.cuda = False## 这里我没有gpu，所以cuda设置为False，如果有gpu可以设置为True，并且在下面的device中选择对应的gpu编号

        # train setting
        self.last_action = True  # 使用最新动作选择动作
        self.reuse_network = True  # 对所有智能体使用同一个网络
        self.n_epochs = 1000  # 20000
        self.evaluate_epoch = 2  # 20
        self.evaluate_per_epoch = 100  # 100
        self.batch_size = 64 # 32
        self.buffer_size = int(1e2)
        self.save_frequency = 5000
        self.n_eposodes = 1  # 每个epoch有多少episodes
        self.train_steps = 5  # 每个epoch有多少train steps
        self.gamma = 0.8
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.update_target_params = 100  # 200
        self.continuous_final_reward = True
        self.pulse_reward_target = 600.0
        self.pulse_reward_scale = 80.0
        self.smoothness_reward_weight = 1.5
        self.smoothness_reward_target = 30.0
        self.result_dir = './results/'
        self.verbose = False

        # test setting
        self.load_model = True
        self.model_tag = "5"

        # SC2 env setting
        self.map_name = '3m'
        self.step_mul = 8  # 多少步执行一次动作
        self.difficulty = '2'
        self.game_version = 'latest'
        self.replay_dir = './replay_buffer/'

        self.device = torch.device("cpu")

        # if self.cuda:
        #     self.device = torch.device("cuda: 3" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = torch.device("cpu")

        # model structure
        # drqn net
        self.drqn_hidden_dim = 64
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.mixer = "qatten"
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = True ##Flase
        self.hyper_hidden_dim = 64
        self.n_attention_heads = 4
        self.qatten_hidden_dim = 32
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.optimizer = "RMS"
        self.learning_rate = 1e-2

        # epsilon greedy
        self.start_epsilon = 0.4
        self.end_epsilon = 0.05
        self.anneal_steps = 1200  # 50000,800
        self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'

        self.n_actions = 9
        self.state_shape = 8 #8,14
        self.obs_shape = 8 #8,7
        self.n_agents = 2
        self.episode_limit = 4 ##4,7

        # self.n_actions = 45
        # self.state_shape = 8 #8,14
        # self.obs_shape = 8 #8,7
        # self.n_agents = 2
        # self.episode_limit = 4 ##4,7


    # def set_env_info(self, env_info):
    #     self.n_actions = env_info["n_actions"]
    #     self.state_shape = env_info["state_shape"]
    #     self.obs_shape = env_info["obs_shape"]
    #     self.n_agents = env_info["n_agents"]
    #     self.episode_limit = env_info["episode_limit"]
