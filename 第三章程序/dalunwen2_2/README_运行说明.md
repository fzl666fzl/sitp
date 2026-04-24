# dalunwen2_2 运行说明

## 最近的修改

本次主要围绕 `Qatten` 方向，对当前实验代码做了整理，尽量不破坏原有运行方式。

1. 模型加载逻辑整理
- 修改了 `policy.py`
- 现在会自动扫描 `models/3m` 目录
- 优先加载最新编号的模型
- 同时支持固定文件名：
  - `latest_drqn_net_params.pkl`
  - `latest_qatten_mixer_params.pkl`
- 如果当前 `mixer=qmix`，也兼容旧的 `*_qmix_net_params.pkl`

2. 模型保存逻辑整理
- 修改了 `policy.py`
- 保存模型时，除了按编号保存，还会额外维护：
  - `latest_drqn_net_params.pkl`
  - `latest_qatten_mixer_params.pkl` 或 `latest_qmix_mixer_params.pkl`
- 这样后续在线推理时不需要再手改模型路径

3. 训练入口整理
- 保留旧入口：`QMIX_dis1.py`
- 新增清晰入口：`QATTEN_dis1.py`
- 现在推荐优先使用 `QATTEN_dis1.py` 进行 Qatten 训练

4. 在线推理入口整理
- 保留旧入口：`onlineqmix.py`
- 新增清晰入口：`QATTEN_online.py`
- 现在推荐优先使用 `QATTEN_online.py` 进行 Qatten 在线应用

5. 启动提示补充
- 在训练和在线入口中补了当前 `mixer` 的打印信息
- 运行时可以直接确认当前实际跑的是 `qmix` 还是 `qatten`


## 当前推荐运行方式

先进入目录：

```powershell
cd "C:\Users\Lenovo\Desktop\sitp\project\sitp\第三章程序\dalunwen2_2"
```

### 1. Qatten 训练

```powershell
python QATTEN_dis1.py
```

### 2. Qatten 在线应用 / 推理

```powershell
python QATTEN_online.py
```


## 旧入口说明

旧入口仍然保留，可继续使用：

### 1. 旧训练入口

```powershell
python QMIX_dis1.py
```

### 2. 旧在线入口

```powershell
python onlineqmix.py
```


## QMIX baseline 使用说明

为了和当前 `Qatten` 做公平对比，项目中已经补了一套“当前环境下的 QMIX baseline”入口。

这套 baseline 的特点是：

1. 保留当前环境不变
- 保留当前双 agent 决策逻辑
- 保留当前状态、动作空间和扰动设置
- 不回退到很早之前的旧 QMIX 代码

2. 只切换 mixer
- baseline 使用 `qmix`
- 改进方法使用 `qatten`
- 这样最后对比更公平

3. 模型目录单独保存
- QMIX baseline 模型保存在：
  - `models_qmix_baseline/3m`
- 不会和当前 `Qatten` 的模型混在一起

### 1. QMIX baseline 训练

```powershell
python QMIX_baseline_dis1.py
```

说明：
- 该入口使用 `config_qmix_baseline.py`
- 当前 `mixer = "qmix"`
- 默认训练时 `load_model = False`

### 2. QMIX baseline 在线应用 / 推理

```powershell
python QMIX_baseline_online.py
```

说明：
- 该入口会加载 `models_qmix_baseline/3m` 下的 baseline 模型
- 在线推理逻辑和 `QATTEN_online.py` 对应一致

### 3. 推荐对比方式

建议按下面两组方式做正式对比：

1. QMIX baseline
- 训练：`QMIX_baseline_dis1.py`
- 在线：`QMIX_baseline_online.py`

2. Qatten
- 训练：`QATTEN_dis1.py`
- 在线：`QATTEN_online.py`

建议保持以下条件一致：
- 同一个数据集
- 同一个扰动设置
- 同一个初始节拍
- 同样的训练轮数
- 同样的环境逻辑

这样论文里才能说明：
- 性能差异主要来自 `Qatten` 的注意力混合机制
- 而不是来自环境或脚本结构变化


## 当前配置说明

当前 `config.py` 中已经设置：

```python
self.mixer = "qatten"
```

因此当前默认推荐流程就是：

1. 训练：`QATTEN_dis1.py`
2. 在线推理：`QATTEN_online.py`


## 补充说明

1. 如果目录下还没有训练好的 `Qatten` 模型文件，程序会提示没有找到模型，并使用随机初始化继续运行。
2. 推荐训练完成后检查 `models/3m` 目录，确认是否已经生成：
- `latest_drqn_net_params.pkl`
- `latest_qatten_mixer_params.pkl`
3. 如果后续需要切回 `QMIX`，只需要在 `config.py` 中把 `mixer` 改回 `qmix`，原有兼容逻辑仍然保留。
