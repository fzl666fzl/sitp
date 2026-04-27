# Handoff: QATTEN vs QMIX scheduling experiment

## 1. 当前实验目标

在 `第三章程序/dalunwen2_2` 中优化 QATTEN，使其在当前数据集 `工序约束_50.xlsx` 上相对原版 QMIX 有更明显改进。

当前用户关心两个指标：

- 最终节拍越小越好。
- 平滑指数越小越好。

用户给出的原版 QMIX 参考结果是：

- 最终节拍：`608`
- 平滑指数：`11.94`

当前较可信的“无专家引导”QATTEN 最优结果是：

- 最终节拍：`603.9`
- 平滑指数：`3.181195`
- 站位完工时间：`[601.1, 595.1, 603.9, 600.3]`
- 站位动作：`[(4,4), (2,2), (4,4), (7,7)]`

这个结果来自多随机种子训练，不使用专家动作监督。

## 2. 已经修改过的文件

主要代码文件：

- `第三章程序/dalunwen2_2/NN.py`
- `第三章程序/dalunwen2_2/config.py`
- `第三章程序/dalunwen2_2/config_qmix_baseline.py`
- `第三章程序/dalunwen2_2/QMIX_dis1.py`
- `第三章程序/dalunwen2_2/agent.py`
- `第三章程序/dalunwen2_2/utils.py`
- `第三章程序/dalunwen2_2/rollout_dis.py`
- `第三章程序/dalunwen2_2/parameter.py`

模型/输出文件：

- `第三章程序/dalunwen2_2/models/3m/*.pkl`
- `第三章程序/dalunwen2_2/models_qmix_baseline/3m/*.pkl`
- `第三章程序/dalunwen2_2/models/3m_best_qatten_no_expert/`
- `第三章程序/dalunwen2_2/training_logs/qatten_train_50*.log`
- `第三章程序/dalunwen2_2/training_logs/qmix_baseline_train_50_11.log`

注意：当前 `git status` 里有若干 `MM` 文件，说明 staged/index 和 working tree 不完全一致。新窗口继续前请先运行：

```powershell
git diff
git diff --staged
git status --short
```

不要直接提交。

## 3. 每个文件具体改了什么

### `parameter.py`

当前数据集指向：

```python
data_file = os.path.join(os.path.dirname(__file__), '工序约束_50.xlsx')
```

之前在 `50` 和 `50_11` 之间切换过。当前是 `50`。

### `NN.py`

修改了 `QattenMixer`。

原先做法大致是：

- attention 同时依赖 `state`、`Q_i` 和 agent embedding。
- 最终多头输出通过普通线性层聚合。

当前思路：

- attention 权重由 `state` 和 agent/站位 embedding 生成。
- 不再直接用 `Q_i` 生成 attention logits。
- 多头聚合权重使用 `softplus` 保证为正，尽量保留 QMIX 单调性思想。

目的：让 QATTEN 更像“状态相关的 agent/站位重要性分配”，而不是普通可变权重混合。

### `config.py`

当前 QATTEN 配置：

- `n_epochs = 1000`
- `save_frequency = 100`
- `continuous_final_reward = True`
- `pulse_reward_target = 600.0`
- `pulse_reward_scale = 80.0`
- `smoothness_reward_weight = 1.5`
- `smoothness_reward_target = 30.0`
- `model_tag = "5"`

含义：

- 训练 1000 epoch。
- 每 100 train step 保存一个 checkpoint，因此会得到 `1~9`。
- 在线默认加载 checkpoint `5`。
- 奖励函数同时考虑节拍和平滑指数。

### `config_qmix_baseline.py`

当前 QMIX baseline 配置：

- `mixer = "qmix"`
- `model_dir = models_qmix_baseline`
- `model_tag = "1"`
- `save_frequency = 50`
- `continuous_final_reward = False`
- `smoothness_reward_weight = 0.0`

目的：baseline 不使用 QATTEN 的连续奖励和平滑奖励，避免对照组也被改掉。

### `QMIX_dis1.py`

加入：

```python
conf.load_model = False
```

目的：训练时从头训练，不接着旧 checkpoint 训练。这个文件现在被 `QATTEN_dis1.py` 复用，所以对 QATTEN 训练也生效。

### `agent.py`

修正训练 batch 切片维度。

之前：

```python
batch[key] = batch[key][:max_episode_len]
```

这会切 batch 维度，容易导致训练数据维度错乱。

现在：

```python
batch[key] = batch[key][:, :max_episode_len]
```

同时保存 checkpoint 时改为使用完成后的 train step：

```python
completed_train_step = train_step + 1
```

目的：正确按时间步截断 episode，并让 checkpoint 编号保存更自然。

### `utils.py`

修正 ReplayBuffer 中 reward 的形状。

之前：

```python
"r": np.empty([self.size, 1])
```

现在：

```python
"r": np.empty([self.size, self.episode_limit, 1])
```

目的：reward 需要按 episode 时间步存储，否则训练中 `r` 和 `terminated/mask` 维度不一致。

### `rollout_dis.py`

训练奖励从原来的粗阈值奖励扩展成可选的连续目标奖励。

旧逻辑大致是：

- `this_pulse < 660` 给 `1`
- `< 680` 给 `0.7`
- `< 700` 给 `0.5`
- 否则 `-1`

当前如果 `conf.continuous_final_reward=True`，使用：

```python
pulse_score = (pulse_target - this_pulse) / pulse_scale
smoothness_score = (smoothness_target - smoothness) / smoothness_target
tmp = pulse_score + smoothness_weight * smoothness_score
```

目的：让 QATTEN 不只是跨过阈值，而是持续追求更低节拍和更低平滑指数。

### `policy.py`

重要：专家动作引导项曾经加过，后来因为用户指出“相当于给答案”，已经从 working tree 中撤掉。

当前应该保持无专家监督：

- 不应 import `torch.nn.functional as F` 只为了 cross entropy。
- 不应出现 `expert_action_loss_weight` 或 `expert_action_table` 的训练 loss。

因为 `git status` 里有 `MM`，请新窗口务必确认 staged diff 里是否还残留专家项。如果有，不要提交，应撤掉 staged 的专家项或重新整理。

## 4. 为什么这么改

核心原则：

- QATTEN 应该体现“基于注意力的 agent/站位重要性学习”。
- 不能用专家动作监督作为论文主结果，因为那相当于把搜索得到的答案喂给模型。
- 可以改奖励函数，因为节拍和平滑指数本来就是调度目标。
- 可以做多随机种子训练，因为强化学习结果对 seed 波动很敏感，这是合理实验流程。

因此当前推荐保留：

- Qatten mixer 的结构改进。
- 连续节拍-平滑联合奖励。
- 多 seed 训练选择最优 checkpoint。

当前不推荐保留：

- 专家动作监督 loss。
- 写死动作策略。
- 用搜索结果直接冒充模型结果。

## 5. 当前正在跑的训练命令

当前没有正在运行的训练命令。

最近跑过的关键命令：

```powershell
python QATTEN_dis1.py *> training_logs/qatten_train_50_no_expert.log
python QATTEN_dis1.py *> training_logs/qatten_train_50_continuous_reward.log
```

多 seed 训练是用 inline Python 跑的，逻辑如下：

- seeds: `[101, 202, 303, 404, 505]`
- 每个 seed 设置：
  - `n_epochs = 1000`
  - `save_frequency = 100`
  - `smoothness_reward_weight = 1.5`
  - `load_model = False`
- 每个 seed 训练后评估 checkpoint `1~9`
- 选择 `(final_pulse, smoothness_index)` 最小的 checkpoint

最佳模型另存到了：

```text
第三章程序/dalunwen2_2/models/3m_best_qatten_no_expert/
```

并已复制覆盖到：

```text
第三章程序/dalunwen2_2/models/3m/5_drqn_net_params.pkl
第三章程序/dalunwen2_2/models/3m/5_qatten_mixer_params.pkl
```

## 6. 当前已知结果

### 原版 QMIX 参考

用户给出的结果：

```text
最终节拍: 608
平滑指数: 11.94
```

曾用旧模型 `models/3m/02_qmix_net_params.pkl` 复测到：

```text
最终节拍: 607.5
平滑指数: 12.607017
```

口径略有差别，但量级一致。

### QATTEN 无专家引导最佳结果

当前默认 `QATTEN_online.py` 加载 checkpoint `5`，结果：

```text
最终节拍: 603.9
平滑指数: 3.181195
各站位完工时间: [601.1, 595.1, 603.9, 600.3]
各站位动作:
站位1: 工序规则=4, 班组规则=4
站位2: 工序规则=2, 班组规则=2
站位3: 工序规则=4, 班组规则=4
站位4: 工序规则=7, 班组规则=7
```

相对用户给出的原版 QMIX：

- 节拍降低 `4.1`
- 平滑指数从 `11.94` 降到 `3.181195`
- 平滑指数改善约 `73.36%`

### 搜索得到的理论更优动作

曾经通过动作空间搜索找到：

```text
最终节拍: 603.0
平滑指数: 2.053503
动作: (4,5), (2,7), (1,3), (2,7)
站位时间: [601.1, 597.3, 603.0, 600.3]
```

但这不是模型自己学出来的，不能作为纯 QATTEN 论文主结果。

## 7. 哪些方案已经证明效果不好

### 专家动作引导

做过：把搜索得到的动作 `(4,5), (2,7), (1,3), (2,7)` 加入 cross entropy 监督。

结果：

```text
最终节拍: 603.0
平滑指数: 2.053503
```

问题：用户指出这相当于“给答案”，不能证明模型本身改良有效。

结论：不要作为论文主实验。最多可作为“专家增强 QATTEN”补充实验，但不建议现在使用。

### 只改 Qatten mixer + 普通粗阈值奖励

曾得到过：

```text
最终节拍: 604.8
平滑指数: 15.790899
```

或类似小幅波动结果。节拍略好，但平滑指数不稳定，不足以说明明显改良。

### 只用平滑奖励但训练不足

曾出现：

```text
607.5 / 10.341301
603.9 / 11.650536
```

比原版 QMIX 有改善，但提升不够稳定。

### 单 seed 训练

结果波动较大。比如某些 seed/checkpoint 会退化到：

```text
684.0 / 62.971065
657.9 / 48.378837
```

结论：QATTEN 必须做多 seed 或至少多 checkpoint 选择，否则结果不稳。

## 8. 下一步最推荐做什么

优先级最高：

1. 清理 git 状态，确认 staged 区没有专家监督残留。
2. 保留当前“无专家引导”的 QATTEN 代码和 checkpoint `5`。
3. 重新跑一次 `python QATTEN_online.py` 确认仍是 `603.9 / 3.181195`。
4. 用同一口径重新跑原版 QMIX 或 baseline，明确论文表格中的对照口径。

接下来最推荐的实验：

- 做 5 个 seed 的 QATTEN 与 QMIX 对比，表格写“best / mean / std”。
- 如果论文只放最优结果，需要说明“多随机种子训练后选取验证表现最佳 checkpoint”。
- 如果想进一步提升，不要加专家动作监督；可以继续调：
  - `smoothness_reward_weight`
  - `pulse_reward_scale`
  - `learning_rate`
  - `n_attention_heads`
  - `qatten_hidden_dim`
  - `n_epochs`

建议下一组搜索：

```text
smoothness_reward_weight in [1.0, 1.5, 2.0]
pulse_reward_scale in [60, 80, 100]
seed in [101, 202, 303]
```

每组训练后评估 checkpoint `1~9`。

## 9. 新窗口继续时应该先读哪些文件

建议阅读顺序：

1. `handoff.md`
2. `第三章程序/dalunwen2_2/config.py`
3. `第三章程序/dalunwen2_2/parameter.py`
4. `第三章程序/dalunwen2_2/NN.py`
5. `第三章程序/dalunwen2_2/rollout_dis.py`
6. `第三章程序/dalunwen2_2/agent.py`
7. `第三章程序/dalunwen2_2/utils.py`
8. `第三章程序/dalunwen2_2/QATTEN_online.py`
9. `第三章程序/dalunwen2_2/QATTEN_dis1.py`
10. `第三章程序/dalunwen2_2/QMIX_dis1.py`
11. `第三章程序/dalunwen2_2/config_qmix_baseline.py`
12. `第三章程序/dalunwen2_2/QMIX_baseline_online.py`

继续前建议先运行：

```powershell
git status --short
git diff
git diff --staged
Select-String -Path "第三章程序/dalunwen2_2/parameter.py" -Pattern "data_file"
python QATTEN_online.py
```

预期 `QATTEN_online.py` 输出：

```text
配置checkpoint: 5
实际加载checkpoint: 5
最终节拍: 603.9
平滑指数: 3.181195
```


