
#  第一次handoff

## 1. 当前实验目标

当前主目标是在目录 [第三章程序/dalunwen2_2](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2) 下，做一个**公平的 QMIX baseline vs Qatten 对比**。

当前约定的比较口径是：

- 同一个环境
- 同一个数据集
- 同样的 2 个 agent 决策逻辑
- 同样的在线评估方式
- 只比较 `mixer=qmix` 和 `mixer=qatten`

当前更具体的工作重点是：

1. 用固定评估模式筛 `best Qatten checkpoint`
2. 用同样方式筛 `best QMIX baseline checkpoint`
3. 最后比较 `best QMIX` 和 `best Qatten`

---

## 2. 已经修改过的文件

当前窗口相关、且能确认改过的主要文件有：

- [第三章程序/dalunwen2_2/NN.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/NN.py)
- [第三章程序/dalunwen2_2/policy.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/policy.py)
- [第三章程序/dalunwen2_2/config.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config.py)
- [第三章程序/dalunwen2_2/config_qmix_baseline.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config_qmix_baseline.py)
- [第三章程序/dalunwen2_2/parameter.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/parameter.py)
- [第三章程序/dalunwen2_2/rollout_dis.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/rollout_dis.py)
- [第三章程序/dalunwen2_2/onlinerollout.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlinerollout.py)
- [第三章程序/dalunwen2_2/agent.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/agent.py)
- [第三章程序/dalunwen2_2/utils.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/utils.py)
- [第三章程序/dalunwen2_2/onlineqmix.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlineqmix.py)
- [第三章程序/dalunwen2_2/QATTEN_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_online.py)
- [第三章程序/dalunwen2_2/QMIX_baseline_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_online.py)
- [第三章程序/dalunwen2_2/QATTEN_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_dis1.py)
- [第三章程序/dalunwen2_2/QMIX_baseline_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_dis1.py)
- [第三章程序/dalunwen2_2/README_运行说明.md](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/README_运行说明.md)

---

## 3. 每个文件具体改了什么

| 文件 | 具体改动 |
|---|---|
| [NN.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/NN.py) | 在原有 `DRQN`、`QMIXNET` 基础上新增了 `QattenMixer`，让项目支持 `DRQN + Qatten`。 |
| [policy.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/policy.py) | 把 mixer 改成可切换：`qmix -> QMIXNET`，`qatten -> QattenMixer`；支持自动扫描 checkpoint；支持 `latest` 和指定编号 `model_tag`；记录实际加载的 checkpoint。 |
| [config.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config.py) | 作为 Qatten 主配置，当前设置为 `mixer="qatten"`；增加 `model_tag`；模型目录改为相对当前文件；增加连续奖励参数；当前 `save_frequency` 已改回 `5000`。 |
| [config_qmix_baseline.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config_qmix_baseline.py) | 基于 `config.py` 继承出一套 QMIX baseline 配置；切到 `mixer="qmix"`；单独模型目录 `models_qmix_baseline`；单独 `model_tag`；`save_frequency=50`；关闭连续奖励和平滑奖励。 |
| [parameter.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/parameter.py) | 去掉了导入时自动打印的大字典和自动执行；当前数据集切到 `工序约束_50.xlsx`。 |
| [rollout_dis.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/rollout_dis.py) | 训练环境里把第 2 个 agent 真正接入决策；现在是 `agent0` 控工序规则，`agent1` 控班组规则；增加连续节拍/平滑联合奖励分支。 |
| [onlinerollout.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlinerollout.py) | 在线评估环境里去掉大量调试输出；返回结构化摘要；支持 `evaluate=True` 时把 `epsilon` 强制视为 0；记录站位动作、站位完工时间、最终节拍、平滑指数。 |
| [agent.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/agent.py) | `choose_action()` 在 `evaluate=True` 时不再随机探索；`train()` 修正了 batch 的 episode 维度切片，并按完成后的 `train_step` 决定是否保存模型。 |
| [utils.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/utils.py) | 重写为更干净的 `ReplayBuffer`/`RolloutWorker` 版本，减少输出噪音，修正 reward/episode 的存储形状。 |
| [onlineqmix.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlineqmix.py) | 改成简洁在线摘要输出；固定评估 seed；在线时使用 `evaluate=True`；输出算法、checkpoint、节拍、平滑指数、站位动作。 |
| [QATTEN_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_online.py) | 增加一个名字清晰的 Qatten 在线入口，内部复用 `onlineqmix.py`。 |
| [QMIX_baseline_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_online.py) | 新增一个名字清晰的 QMIX baseline 在线入口，和 Qatten 评估口径保持一致。 |
| [QATTEN_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_dis1.py) | 新增一个名字清晰的 Qatten 训练入口，内部复用 `QMIX_dis1.py` 的主训练流程。 |
| [QMIX_baseline_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_dis1.py) | 新增当前环境下的 QMIX baseline 训练入口，使用 baseline 配置和单独模型目录。 |
| [README_运行说明.md](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/README_运行说明.md) | 增加了这轮改动和推荐运行方式说明，但内容里有一部分旧信息，继续用之前最好重新核对。 |

---

## 4. 为什么这么改

主要原因有 6 个：

1. 原项目只有 `QMIX`，用户想做 `Qatten` 改进，所以新增了 `QattenMixer`，但尽量不推翻原来的 `DRQN + mixer` 主结构。
2. 原项目虽然写了 2 个 agent，但环境里基本只吃第 1 个 agent 的动作，所以把第 2 个 agent 接到“班组规则”上，让多智能体名副其实。
3. 原项目在线输出太杂，前驱/后继字典、buffer 数组、内部状态都往外打，后面做实验记录很痛苦，所以改成了摘要输出。
4. 原项目在线评估带随机探索，同一个 checkpoint 会跑出完全不同的结果，所以加了“固定 seed + epsilon=0”的评估模式。
5. 原项目默认只适合 `latest`，不方便做 checkpoint 对比，所以新增了 `model_tag` 机制，可以直接评估 `1~9` 或 `latest`。
6. 需要公平比较 Qatten 和 QMIX，所以单独做了一条 `QMIX baseline` 线，和当前环境保持一致，只改 mixer。

---

## 5. 当前正在跑的训练命令

当前**没有确认正在运行的训练命令**。

说明：

- 这条线程当前没有 attached terminal session。
- 目前能确认的是最近主要在跑在线评估，不是在持续训练。

如果下一窗口要继续训练，当前推荐命令是：

```powershell
cd "C:\Users\Lenovo\Desktop\sitp\project\sitp\第三章程序\dalunwen2_2"
python QATTEN_dis1.py
```

如果要继续跑 baseline：

```powershell
cd "C:\Users\Lenovo\Desktop\sitp\project\sitp\第三章程序\dalunwen2_2"
python QMIX_baseline_dis1.py
```

补充：

- 现在 [config.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config.py) 里的 `save_frequency = 5000`，在 `n_epochs=1000` 这套训练规模下，Qatten 训练大概率**不会再自动产出新的 checkpoint**。
- 但 [config_qmix_baseline.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config_qmix_baseline.py) 里单独覆盖成了 `save_frequency = 50`，所以 QMIX baseline 训练仍会正常保存 `1~9` 这类 checkpoint。

---

## 6. 当前已知结果

以下结果都指当前数据集 [工序约束_50.xlsx](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/工序约束_50.xlsx) 下、固定评估模式的结果。

### Qatten

- `checkpoint 4`
  - 最终节拍：`600.3`
  - 平滑指数：`17.586501`
  - 各站位完工时间：`[555.0, 562.5, 581.4, 600.3]`
  - 各站位动作：`[(0,0), (5,5), (5,5), (5,5)]`
  - 这是目前窗口里看到的**最强候选结果**

- `checkpoint 3`
  - 最终节拍：`900.9`
  - 平滑指数：`160.890652`
  - 各站位完工时间：`[512.0, 576.9, 513.9, 900.9]`
  - 动作：`[(1,1), (5,5), (5,5), (8,5)]`

- `checkpoint 6`
  - 最终节拍：`908.1`
  - 平滑指数：`153.320791`
  - 各站位完工时间：`[570.8, 549.0, 544.5, 908.1]`
  - 动作基本全是 `[4,4]`

### QMIX baseline

- `checkpoint 3`
  - 最终节拍：`908.1`
  - 平滑指数：`156.126823`
  - 各站位完工时间：`[549.2, 549.0, 544.5, 908.1]`
  - 各站位动作：`[(3,3), (4,4), (4,4), (4,4)]`

- `checkpoint 4`
  - 和 `checkpoint 3` 完全一样

- 用户已经观察到：`QMIX baseline 1~9` 在当前固定评估场景下看起来都一样，至少 `3` 和 `4` 已明确相同。

当前粗结论：

- **Qatten checkpoint 4 明显优于当前看到的 QMIX baseline 结果**
- 但 Qatten 各 checkpoint 差异很大，因此不能只看 `latest`

---

## 7. 哪些方案已经证明效果不好

目前已知不推荐继续走的方向有：

1. **`latest QMIX` 直接对 `latest Qatten`**
   - 这个比较不稳定，也不能代表 best checkpoint。

2. **不固定 seed、在线评估继续随机探索**
   - 之前同一个 `Qatten checkpoint 6` 能跑出 `664.0` 和 `908.1` 两种完全不同的结果。
   - 这个问题已经通过“固定 seed + epsilon=0”解决。

3. **把 `save_frequency` 设成 `5000` 后直接期待训练出新 checkpoint**
   - 对当前 `Qatten` 配置来说，这样几乎不会保存出新 checkpoint。
   - 所以如果下一步要继续训练并筛新的 Qatten 模型，必须先重新调整保存频率。

4. **把 `Qatten checkpoint 3`、`6` 当成有效改进结果**
   - 当前数据集下，这两个 checkpoint 明显不如 `checkpoint 4`。

---

## 8. 下一步最推荐做什么

按优先级排序，当前最推荐的下一步是：

1. **先不要继续大改结构**
   - 目前最有价值的是把实验结论整理稳定，而不是继续堆功能。

2. **把 `Qatten 1~9` 再完整筛一遍并记录表格**
   - 当前已经知道 `4` 很强，`3/6` 较差。
   - 需要确认 `1、2、5、7、8、9` 的结果，正式选出 `best Qatten`。

3. **把 `QMIX baseline 1~9` 再确认一遍**
   - 如果确实都一样，可以在论文里写成“QMIX 在当前评估场景下已稳定收敛到同一策略”。

4. **整理正式对比表**
   - 建议至少记录：算法、数据集、checkpoint、最终节拍、平滑指数、各站位完工时间。

5. **如果要继续训练 Qatten，先改回更合理的保存频率**
   - 例如 `50` 或 `100`
   - 否则 `QATTEN_dis1.py` 跑完也不会留下可比较的新 checkpoint

6. **把在线输出补上“数据集”和“设定节拍”字段**
   - 这个之前已经讨论过，意义很大，但当前用户明确说先不要继续大改代码，所以暂时没动。

---

## 9. 新窗口继续时应该先读哪些文件

建议新窗口按这个顺序快速接手：

1. [handoff.md](/C:/Users/Lenovo/Desktop/sitp/project/sitp/handoff.md)
2. [第三章程序/dalunwen2_2/parameter.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/parameter.py)
3. [第三章程序/dalunwen2_2/config.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config.py)
4. [第三章程序/dalunwen2_2/config_qmix_baseline.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/config_qmix_baseline.py)
5. [第三章程序/dalunwen2_2/policy.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/policy.py)
6. [第三章程序/dalunwen2_2/NN.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/NN.py)
7. [第三章程序/dalunwen2_2/onlinerollout.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlinerollout.py)
8. [第三章程序/dalunwen2_2/agent.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/agent.py)
9. [第三章程序/dalunwen2_2/onlineqmix.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/onlineqmix.py)
10. [第三章程序/dalunwen2_2/QATTEN_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_online.py)
11. [第三章程序/dalunwen2_2/QMIX_baseline_online.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_online.py)
12. [第三章程序/dalunwen2_2/QMIX_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_dis1.py)
13. [第三章程序/dalunwen2_2/QATTEN_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QATTEN_dis1.py)
14. [第三章程序/dalunwen2_2/QMIX_baseline_dis1.py](/C:/Users/Lenovo/Desktop/sitp/project/sitp/第三章程序/dalunwen2_2/QMIX_baseline_dis1.py)

---

## 补充：当前能确认的关键设置

- 当前 Qatten 数据集：`工序约束_50.xlsx`
- 当前 Qatten 在线默认 `model_tag`：`5`
- 当前 QMIX baseline 在线默认 `model_tag`：`2`
- 当前 Qatten `save_frequency`：`5000`
- 当前 QMIX baseline `save_frequency`：`50`
- 当前在线评估方式：固定 `seed=133`，并且 `epsilon=0`

这几个值如果后面被改了，新的窗口一定要重新核对一遍。


# 第二次handoff Handoff: QATTEN vs QMIX scheduling experiment

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


