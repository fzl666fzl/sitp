# 第三章端到端 GNN-QATTEN 结果说明

本文档记录第三章端到端 GNN-QATTEN 实验线的当前冻结结果、复现命令、checkpoint 含义和论文写作口径。模型权重和运行日志不提交到 GitHub；代码、评估脚本和轻量结果说明提交到仓库。

## 1. 当前冻结主结果

当前端到端主结果使用：

```text
checkpoint = online_balanced_best
eval seed = 133
init_pulse = 608
score = final_pulse + SI
```

固定 seed 结果：

| 方法 | checkpoint | final_pulse | SI | station_times |
|---|---|---:|---:|---|
| End-to-End GNN-QATTEN | online_balanced_best | 585.0 | 4.8602 | [585.0, 573.1, 583.2, 576.4] |

对应甘特图：

```text
figures_end2end/gantt_online_balanced_best_seed133.png
figures_end2end/gantt_online_balanced_best_seed133_summary.json
```

本地 checkpoint 已额外压缩备份到忽略目录，避免后续训练覆盖后无法恢复。

## 2. Checkpoint 含义

| checkpoint | 选择标准 | 用途 |
|---|---|---|
| `validation_best` | 固定 `seed=133, init_pulse=608` 下 `final_pulse + SI` 最小 | 固定 online 验证 |
| `online_balanced_best` | 与 `validation_best` 同步，默认 online 加载目标 | 第三章端到端主结果 |
| `online_pulse_best` | 固定 online 下优先降低 `final_pulse`，同节拍再比较 SI | 节拍优先对照 |
| `best` | 训练采样 episode 的最优结果 | 训练过程诊断，不作为 online 主结果 |
| `latest` | 最后保存的训练状态 | 调试用，不作为论文结果 |

注意：`best` 中曾出现 `567.6 / 4.25`，但这是训练采样轨迹，不是固定 online greedy 策略结果，因此不作为主结果。

## 3. 多 seed 评估结果

端到端模型使用 `seeds=100..149`，`init_pulse=608`，`checkpoint=online_balanced_best`。

| 指标 | 10 seed: 100..109 | 50 seed: 100..149 |
|---|---:|---:|
| pulse mean | 590.135 | 591.604 |
| pulse median | 585.0 | 585.12 |
| pulse min | 585.0 | 585.0 |
| pulse max | 605.65 | 605.65 |
| SI mean | 6.7738 | 11.1182 |
| SI median | 4.8602 | 4.8950 |
| SI min | 4.8602 | 4.8602 |
| SI max | 12.6940 | 39.7148 |
| SI <= 5 | 7 / 10 | 26 / 50 |
| SI > 10 | 2 / 10 | 12 / 50 |
| finished 50 | 10 / 10 | 50 / 50 |

QATTEN checkpoint 5 使用相同 `seeds=100..149`，`init_pulse=608`。

| 指标 | QATTEN checkpoint 5 |
|---|---:|
| pulse mean | 614.988 |
| pulse median | 603.9 |
| pulse min | 603.9 |
| pulse max | 673.2 |
| SI mean | 8.9485 |
| SI median | 3.1812 |
| SI min | 3.1812 |
| SI max | 53.6803 |
| SI <= 5 | 42 / 50 |
| SI > 10 | 8 / 50 |

## 4. 与 QATTEN baseline 的对比

固定 seed 主结果：

| 方法 | 动作空间 | final_pulse | SI | 说明 |
|---|---|---:|---:|---|
| QATTEN checkpoint 5 | 两 agent 各选 9 个规则动作 | 603.9 | 3.1812 | 原主 baseline，负载更均衡 |
| 旧端到端 GNN-QATTEN | 150 个工序-班组组合动作 | 585.0 | 8.6488 | 节拍降低，但 SI 偏高 |
| 新端到端 GNN-QATTEN | 150 个工序-班组组合动作 | 585.0 | 4.8602 | 节拍保持，SI 明显改善 |

核心结论：

```text
端到端动作空间让 GNN 节点信息直接进入调度决策，显著降低 final_pulse；
但 QATTEN checkpoint 5 在 SI 中位数和 SI <= 5 比例上仍更稳定；
端到端模型在多 seed 下存在少数随机扰动造成的负载失衡尾部风险。
```

## 5. 论文第三章写作口径

建议按以下逻辑组织实验分析：

1. 原 9 规则动作空间下，GNN 只能影响规则优先级，不能直接选择具体工序，因此 GNN bias、SI predictor、SI shaping 都难以稳定超过 QATTEN。
2. 端到端动作空间将动作改为：

   ```text
   action = (工序, 班组)
   50 * 3 = 150 个组合动作
   ```

   这使 GNN 工序节点 embedding 和候选动作特征可以直接作用于调度决策。
3. 参考 Song 等关于候选动作直接打分的思想，引入 `ComboActionScorer`，对每个 `(工序, 班组)` 候选动作生成 Q 值。
4. Huang 等优先规则生成方法可作为对照背景：优先规则方法适合解释规则调度，但本项目为了让 GNN 直接控制工序选择，需要从规则动作转向组合动作。
5. 结果表明端到端模型在节拍指标上优于 QATTEN，但 SI 尾部风险仍存在，后续应使用多 seed validation 或风险惩罚，而不是继续普通长训。

## 6. 复现命令

固定 seed online：

```powershell
python GNN_QATTEN_END2END_online.py
```

端到端 checkpoint 对比：

```powershell
python GNN_QATTEN_END2END_eval_checkpoints.py
```

端到端多 seed：

```powershell
$env:END2END_EVAL_TAG='online_balanced_best'
$env:END2END_EVAL_SEEDS='100-149'
$env:END2END_EVAL_INIT_PULSE='608'
python GNN_QATTEN_END2END_eval_multiseed.py
Remove-Item Env:END2END_EVAL_TAG
Remove-Item Env:END2END_EVAL_SEEDS
Remove-Item Env:END2END_EVAL_INIT_PULSE
```

QATTEN checkpoint 5 同 seed 对比：

```powershell
$env:QATTEN_EVAL_SEEDS='100-149'
$env:QATTEN_EVAL_INIT_PULSE='608'
$env:QATTEN_EVAL_MODEL_TAG='5'
python QATTEN_eval_multiseed.py
Remove-Item Env:QATTEN_EVAL_SEEDS
Remove-Item Env:QATTEN_EVAL_INIT_PULSE
Remove-Item Env:QATTEN_EVAL_MODEL_TAG
```

甘特图：

```powershell
python GNN_QATTEN_END2END_plot_gantt.py
```

## 7. 后续优化建议

不建议继续普通长训。下一步如果继续优化，应改为多 seed 选择标准：

```text
score = pulse_median + SI_median + penalty * (SI > 10 ratio)
```

目标从“单 seed 更优”改为“降低 50 seed 尾部风险”。
