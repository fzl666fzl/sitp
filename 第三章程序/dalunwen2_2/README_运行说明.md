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
