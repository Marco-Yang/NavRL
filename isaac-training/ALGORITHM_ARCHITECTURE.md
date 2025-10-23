# NavRL 算法架构文档

## 概述

NavRL是一个基于**PPO (Proximal Policy Optimization)** 强化学习算法的无人机导航系统，使用NVIDIA Isaac Sim进行物理仿真，支持动态障碍物环境下的多机器人导航训练。

---

## 1. 整体架构

### 1.1 系统组件层次

```
训练系统
├── Isaac Simulation (物理仿真)
│   ├── Hummingbird 无人机模型
│   ├── 地形生成器 (静态障碍物)
│   └── 动态障碍物系统
│
├── 环境 (NavigationEnv)
│   ├── 观测空间 (LiDAR + 状态 + 动态障碍物信息)
│   ├── 动作空间 (3D速度控制)
│   ├── 奖励函数
│   └── 任务管理 (起点/目标点生成)
│
├── 控制器转换 (LeePositionController)
│   └── 速度指令 → 姿态控制指令
│
├── PPO算法
│   ├── 特征提取器 (CNN + MLP)
│   ├── Actor网络 (策略网络)
│   ├── Critic网络 (价值网络)
│   └── 训练循环 (GAE + PPO Loss)
│
└── 数据收集与监控
    ├── SyncDataCollector (同步数据采集)
    └── Wandb (训练监控)
```

---

## 2. 环境设计 (NavigationEnv)

### 2.1 观测空间 (Observation Space)

环境为每个无人机提供以下观测信息：

#### **1) LiDAR 点云数据**
- **维度**: `(36, 4)` = 水平36束 × 垂直4束
- **参数**:
  - 探测距离: 4.0米
  - 水平分辨率: 10° (360°/10° = 36束)
  - 垂直视场角: [-10°, 20°]
  - 垂直束数: 4束
- **数据处理**: 归一化距离值，提供周围障碍物的空间信息

#### **2) 无人机状态信息**
包含以下向量 (具体维度需查看`construct_input`函数):
- 当前位置
- 目标位置/方向
- 当前速度
- 姿态信息
- 与目标的相对关系

#### **3) 动态障碍物信息**
- **维度**: `(5, N)` - 追踪最近的5个动态障碍物
- **内容**: 相对位置、速度、大小等信息

### 2.2 动作空间 (Action Space)

- **类型**: 连续动作空间
- **维度**: 3D (x, y, z 速度分量)
- **范围**: [-2.0, 2.0] m/s (可配置)
- **坐标系转换**: 
  - 策略网络输出：机体坐标系下的速度
  - 自动转换为：世界坐标系速度指令
  - 通过Lee Position Controller转换为电机控制指令

### 2.3 奖励函数设计

奖励函数需要查看`env.py`中的详细实现，通常包含：
- ✅ **目标接近奖励**: 鼓励向目标移动
- ⚠️ **碰撞惩罚**: 与障碍物碰撞的负奖励
- 🎯 **成功到达奖励**: 到达目标的大额奖励
- ⏱️ **时间惩罚**: 鼓励快速完成任务
- 🚁 **平滑飞行奖励**: 惩罚过大的速度变化

### 2.4 终止条件

- **成功**: 到达目标点
- **失败**: 碰撞障碍物或地面
- **超时**: 达到最大步数 (2200步)

---

## 3. PPO算法架构

### 3.1 特征提取器 (Feature Extractor)

#### **LiDAR特征提取 (CNN)**
```python
Input: (36, 4) LiDAR数据
  ↓
Conv2d(4 channels, kernel=5×3) + ELU
  ↓
Conv2d(16 channels, kernel=5×3, stride=2×1) + ELU  # 降采样
  ↓
Conv2d(16 channels, kernel=5×3, stride=2×2) + ELU  # 降采样
  ↓
Flatten → Linear(128) + LayerNorm
  ↓
Output: 128维特征向量
```

#### **动态障碍物特征提取 (MLP)**
```python
Input: (5, N) 动态障碍物信息
  ↓
Flatten → MLP[128, 64]
  ↓
Output: 64维特征向量
```

#### **特征融合**
```python
Concatenate:
  - CNN特征 (128维)
  - 无人机状态 (state维度)
  - 动态障碍物特征 (64维)
  ↓
Total: ~192-256维
  ↓
MLP[256, 256]
  ↓
Output: 256维融合特征
```

### 3.2 Actor网络 (策略网络)

- **输入**: 256维特征向量
- **输出**: Beta分布参数 (α, β)
- **动作分布**: Independent Beta Distribution
  - 优势: 输出自然限制在[0,1]区间
  - 映射到动作空间: `action = 2 * normalized_action * limit - limit`
- **坐标转换**: 
  - 策略输出: 机体坐标系速度
  - 环境执行: 世界坐标系速度 (通过`vec_to_world`转换)

### 3.3 Critic网络 (价值网络)

- **输入**: 256维特征向量
- **输出**: 状态价值 V(s)
- **归一化**: 使用ValueNorm进行返回值归一化
  - 基于运行均值和方差
  - 稳定训练过程

### 3.4 PPO训练流程

#### **阶段1: 数据收集**
```python
for each training iteration:
    # 收集 num_envs × training_frame_num 步数据
    rollout = collect_data(
        num_envs=1024,        # 并行环境数
        frames_per_batch=1024 × 32 = 32768
    )
```

#### **阶段2: 优势估计 (GAE)**
```python
# Generalized Advantage Estimation
GAE参数:
  - γ (gamma) = 0.99      # 折扣因子
  - λ (lambda) = 0.95     # GAE平滑参数

计算过程:
  1. TD-error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  2. GAE: A_t = Σ(γλ)^i · δ_{t+i}
  3. Return: G_t = A_t + V(s_t)
```

#### **阶段3: 策略更新 (PPO)**

**超参数**:
- Training Epochs: 4
- Mini-batches: 16
- Total updates per iteration: 4 × 16 = 64

**损失函数**:

1. **Actor Loss (Clipped Surrogate Objective)**
```python
ratio = exp(log π_new(a|s) - log π_old(a|s))
L_actor = -min(
    ratio · A,
    clip(ratio, 1-ε, 1+ε) · A
)
其中 ε = 0.1 (clip_ratio)
```

2. **Critic Loss (Clipped Value Loss)**
```python
V_clipped = V_old + clip(V_new - V_old, -ε, +ε)
L_critic = max(
    HuberLoss(G, V_new),
    HuberLoss(G, V_clipped)
)
使用 HuberLoss (delta=10) 结合L1和L2损失
```

3. **Entropy Loss (探索鼓励)**
```python
L_entropy = -0.001 × mean(H(π))
鼓励策略保持一定的随机性
```

4. **总损失**
```python
L_total = L_actor + L_critic + L_entropy
```

#### **阶段4: 梯度更新**
```python
Optimizers:
  - Feature Extractor: Adam(lr=5e-4)
  - Actor: Adam(lr=5e-4)
  - Critic: Adam(lr=5e-4)

Gradient Clipping:
  - Actor: max_norm = 5.0
  - Critic: max_norm = 5.0
```

---

## 4. 训练配置

### 4.1 默认训练参数

```yaml
训练总帧数: 1.2e9 (12亿帧)
评估间隔: 每1000个训练步
保存间隔: 每1000个训练步

环境配置:
  - 并行环境数: 1024
  - 最大episode长度: 2200步
  - 环境间距: 8.0米
  - 静态障碍物数: 350
  - 动态障碍物数: 80
  - 动态障碍物速度: [0.5, 1.5] m/s

PPO配置:
  - 每次迭代收集帧数: 1024 × 32 = 32768
  - 训练epoch数: 4
  - Mini-batch数: 16
  - Actor学习率: 5e-4
  - Critic学习率: 5e-4
  - Clip ratio: 0.1
  - Entropy coefficient: 1e-3
```

### 4.2 训练命令解析

```bash
python training/scripts/train.py \
    headless=True \              # 无头模式(无渲染)
    env.num_envs=1024 \          # 1024个并行环境
    env.num_obstacles=350 \      # 350个静态障碍物
    env_dyn.num_obstacles=80 \   # 80个动态障碍物
    wandb.mode=online            # 在线监控
```

**训练资源需求**:
- GPU: 建议RTX 3090或更高 (24GB+ VRAM)
- 内存: 32GB+ RAM
- 存储: ~10GB (模型检查点)

---

## 5. 控制流程

### 5.1 完整的一步执行流程

```
1. 策略网络输出 (Policy Network)
   ↓
   Beta分布采样 → 归一化动作 [0,1]
   ↓
   映射到速度范围 [-2, 2] m/s (机体坐标系)
   ↓

2. 坐标转换 (Coordinate Transform)
   ↓
   vec_to_world() → 世界坐标系速度
   ↓

3. 低层控制器 (Lee Position Controller)
   ↓
   速度指令 → 姿态控制 → 电机推力
   ↓

4. 物理仿真 (Isaac Sim)
   ↓
   更新无人机状态
   ↓

5. 传感器更新
   ↓
   LiDAR扫描 + 状态观测
   ↓

6. 奖励计算与终止判断
   ↓
   返回 (observation, reward, done)
```

### 5.2 训练循环伪代码

```python
for iteration in range(total_iterations):
    # 1. 数据收集阶段
    rollout_data = []
    for step in range(frames_per_batch):
        action = policy.select_action(obs)
        next_obs, reward, done = env.step(action)
        rollout_data.append((obs, action, reward, done, value))
    
    # 2. GAE计算
    advantages, returns = compute_gae(rollout_data)
    
    # 3. PPO更新
    for epoch in range(4):
        for minibatch in split_into_minibatches(rollout_data, 16):
            # 计算三种损失
            actor_loss = compute_actor_loss(minibatch)
            critic_loss = compute_critic_loss(minibatch)
            entropy_loss = compute_entropy_loss(minibatch)
            
            # 梯度更新
            total_loss = actor_loss + critic_loss + entropy_loss
            total_loss.backward()
            optimizer.step()
    
    # 4. 评估与保存
    if iteration % eval_interval == 0:
        evaluate_policy()
    if iteration % save_interval == 0:
        save_checkpoint()
```

---

## 6. 关键技术特点

### 6.1 多模态感知融合
- **视觉**: CNN提取LiDAR空间特征
- **状态**: MLP处理无人机运动学状态
- **动态感知**: 专门网络处理动态障碍物信息
- **特征融合**: 多层MLP融合异构信息

### 6.2 坐标系管理
- **观测空间**: 机体坐标系 (便于学习)
- **动作输出**: 机体坐标系速度
- **执行空间**: 世界坐标系 (便于控制)
- **自动转换**: `vec_to_world()` 和 `vec_to_new_frame()`

### 6.3 训练稳定性技术
1. **Value Normalization**: 归一化返回值
2. **Gradient Clipping**: 限制梯度范数≤5.0
3. **PPO Clipping**: 限制策略更新幅度
4. **Huber Loss**: 结合L1和L2的鲁棒损失
5. **Layer Normalization**: 稳定特征提取

### 6.4 大规模并行训练
- 1024个并行环境
- GPU加速物理仿真
- 批量数据收集与处理
- 高效的经验利用 (4 epochs × 16 minibatches)

---

## 7. 性能指标

训练过程监控的关键指标：

```yaml
环境指标:
  - env_frames: 总训练帧数
  - rollout_fps: 数据收集速度 (帧/秒)
  - train/episode_reward: 平均episode奖励
  - train/success_rate: 训练成功率

训练指标:
  - actor_loss: Actor网络损失
  - critic_loss: Critic网络损失
  - entropy: 策略熵 (探索程度)
  - actor_grad_norm: Actor梯度范数
  - critic_grad_norm: Critic梯度范数
  - explained_var: 价值函数解释方差

评估指标:
  - eval/episode_reward: 评估平均奖励
  - eval/success_rate: 评估成功率
  - eval/avg_steps: 平均完成步数
```

---

## 8. 文件结构说明

```
isaac-training/
├── training/
│   ├── scripts/
│   │   ├── train.py          # 主训练脚本
│   │   ├── ppo.py            # PPO算法实现
│   │   ├── env.py            # 导航环境定义
│   │   └── utils.py          # 工具函数
│   │
│   └── cfg/
│       ├── train.yaml        # 训练总配置
│       ├── ppo.yaml          # PPO超参数
│       ├── drone.yaml        # 无人机与传感器配置
│       └── sim.yaml          # 仿真配置
│
├── third_party/
│   ├── OmniDrones/           # 无人机仿真库
│   ├── orbit/                # Isaac Orbit框架
│   └── rl/                   # TorchRL强化学习库
│
└── outputs/                  # 训练输出
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── checkpoint_*.pt    # 模型检查点
            └── logs/              # 训练日志
```

---

## 9. 使用示例

### 9.1 开始训练

```bash
# 基础训练 (小规模测试)
python training/scripts/train.py

# 大规模训练 (生产环境)
python training/scripts/train.py \
    headless=True \
    env.num_envs=1024 \
    env.num_obstacles=350 \
    env_dyn.num_obstacles=80 \
    wandb.mode=online

# 继续训练 (从检查点恢复)
python training/scripts/train.py \
    wandb.run_id=<your_run_id>
```

### 9.2 评估模型

```bash
# 评估训练好的模型
python training/scripts/eval.py \
    checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/checkpoint_final.pt \
    headless=False  # 可视化
```

---

## 10. 总结

NavRL采用**PPO算法**训练无人机在复杂动态环境中导航，核心特点包括：

✅ **多模态感知**: LiDAR + 状态 + 动态障碍物信息  
✅ **分层特征提取**: CNN + MLP 融合架构  
✅ **Beta分布策略**: 自然的动作边界约束  
✅ **GAE优势估计**: 减小方差，加速收敛  
✅ **PPO稳定训练**: Clip机制保证策略平滑更新  
✅ **大规模并行**: 1024环境 × 32步 = 32K样本/次  
✅ **坐标系管理**: 机体坐标学习 + 世界坐标执行  

该架构在Isaac Sim中实现了高效、稳定的端到端强化学习训练流程。

---

## 参考文献

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
2. Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
3. NVIDIA Isaac Sim Documentation
4. TorchRL Documentation
5. OmniDrones Framework

---

**文档版本**: 1.0  
**最后更新**: 2025-10-21  
**作者**: GitHub Copilot
