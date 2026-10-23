<div align="center">

# supermimic-A3

**SuperMimic 在 Agibot A3 + 青瞳乒乓球动捕数据上的复现：训练与评估代码**

用可微 world model 训练人形机器人全身动作跟踪策略。仿真只用来采数据，策略梯度全部从学到的 world model 反传。

![Python](https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%20cu126-EE4C2C?logo=pytorch&logoColor=white)
![MuJoCo Warp](https://img.shields.io/badge/MuJoCo%20Warp-3.8.1-orange)
![Robot](https://img.shields.io/badge/robot-Agibot%20A3%2031dof-2ea44f)
![Paper](https://img.shields.io/badge/arXiv-2610.28480-b31b1b)

</div>

## 目录

- [简介](#简介)
- [结果](#结果)
- [仓库结构](#仓库结构)
- [安装（全新服务器）](#安装全新服务器)
- [准备机器人模型和动作数据](#准备机器人模型和动作数据)
- [自检](#自检)
- [训练](#训练)
- [评估](#评估)
- [实现说明](#实现说明)
- [常见问题](#常见问题)

## 简介

复现论文 [*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480)（CoRL 2026）。论文没有开源代码，这里按论文正文、附录和项目页 demo 从零实现。

- **机器人**：Agibot A3 31dof，右手装球拍。PD 增益、力矩上限、默认姿态和部署配置一致。
- **数据**：青瞳乒乓球动捕，retarget 到 A3，并修正了拍面，按击球对齐重采样到 50 Hz。共 172 段、约 27 分钟、273 次击球。
- **仿真**：MuJoCo Warp，和论文一样。

每个 iteration 的训练循环（论文 Algorithm A.1）：

```text
1. 采数据    仿真里跑当前策略，动作加 N(0, 0.1²) 噪声。
             平均每 iteration 采 2.5 步 × 4096 env（UTD 0.4），写进 replay（每 env 1000 步）。
2. 拟合 WM   从 replay 采 8 步窗口，用记录的动作自回归 rollout。
             10 个 minibatch × 2048 个窗口（Eq. 1）。
3. 更新策略  从 replay 采 4096 个起点，在冻结的 WM 里想象 H=32 步。
             用 BPTT 求 ∇θ[Σγᵗ c_t r_t + γᴴ c_H V(s_H)]（Eq. 2）。
             再用想象轨迹上的 λ-return 训练 critic（Eq. 3，5 epoch × 4 minibatch）。
```

## 结果

单张 RTX 4090，在全部 172 段训练动作上按论文附录 F.1 的协议评估（见[评估](#评估)）：

| 运行 | 预设 | 训练时间 | 失败段数 | 归一化回报 | body 位置误差 | 击球帧拍心误差 |
| --- | --- | --- | --- | --- | --- | --- |
| paper60 / `model_4210` | `paper` | 1 h | 0 / 172 | 0.964 | 2.5 cm | 7.9 cm |
| racket_w1 / `model_10000` | `racket` | ≈2.3 h | 0 / 172 | 0.975 | 2.1 cm | 3.7 cm |
| racket_w1 / `model_25000` | `racket` | ≈5.9 h | 0 / 172 | 0.983 | 1.5 cm | 2.7 cm |
| racket_w1 / `model_50000` | `racket` | 11.7 h | 0 / 172 | **0.984** | **1.4 cm** | 2.9 cm |

- `racket` 预设在论文的奖励上加了球拍位置和姿态奖励。
- `model_50000` 各类技术的击球帧拍心误差：防守 2.2 cm、组合 2.6 cm、台内 3.8 cm、步法 4.0 cm、进攻 4.1 cm。

**速度**（`paper` 预设，4096 env，B=4096，H=32，TF32）：

- 每个 iteration 0.91 s：采数据 0.13 s + WM 0.32 s + 策略 0.46 s。
- 显存峰值 15.7 GB，进程约 18.6 GB。
- WM（3×2048 MLP）前向比 MuJoCo Warp 仿真快 5–7 倍。

## 仓库结构

```text
supermimic-A3/
├── setup_env.sh              从零建 .venv（uv 或 venv + pip），最后跑一次 GPU / torch.compile 自检
├── requirements.txt          torch 2.7.1、mujoco 3.8.1、mujoco-warp 3.8.1、warp-lang 1.14.0、numpy 1.26.4
├── supermimic/
│   ├── config.py             全部超参，默认值 = 论文 Table A.5；预设 paper / racket / 4090 / smoke
│   ├── robot.py              A3 关节表、PD 增益、MJCF 改造（电机 → PD 伺服、5 ms implicitfast、关闭自碰撞）
│   ├── kinematics.py         可微 FK（按层批量），输出位置、姿态和速度
│   ├── rotations.py          旋转工具（四元数、log/exp map、6D、yaw）
│   ├── state.py              广义状态 qpos/qvel；平移和 yaw 不变的网络特征；heading 系下的状态增量
│   ├── motion.py             参考动作库：有限差分速度、预计算 body 运动学、1 s bin
│   ├── task.py               观测（Table A.1）、K=10 未来帧目标、BeyondMimic 奖励（Table A.2）、硬终止和可微软终止
│   ├── networks.py           World model、Actor（goal encoder + 共享 decoder）、Critic
│   ├── sim.py                MuJoCo Warp 批量仿真（CUDA graph），按 env 随机化摩擦和躯干质心
│   ├── replay.py             GPU 环形 replay
│   ├── sampler.py            基于回报的重置采样（附录 D.1）
│   ├── agent.py              WM 拟合、想象 + BPTT 的 actor 更新、GAE critic、多卡梯度平均
│   ├── env.py                仿真采数据
│   ├── train.py              训练主循环（支持 torchrun 多卡）
│   └── evaluate.py           评估（附录 F.1 协议）
├── scripts/
│   ├── check_model.py        安装自检：FK 对 MuJoCo、状态增量往返、MuJoCo Warp 开环回放
│   └── train_eval.sh         训练，然后逐个 checkpoint 评估并汇总
├── tools/pack_assets.sh      把机器人模型和动作数据打成一个包，拷到新服务器
├── assets/                   机器人模型放这里（不进 git，见 assets/README.md）
└── data/                     动作数据放这里（不进 git，见 data/README.md）
```

## 安装（全新服务器）

**前置条件**

| 依赖 | 说明 |
| --- | --- |
| Linux x86_64 | glibc ≥ 2.28（Ubuntu 20.04 及以上） |
| NVIDIA GPU | `paper` 预设需要 24 GB 显存。显存小时用 `--preset racket,4090`，或减小 `--num_envs` / `--imag_batch` |
| NVIDIA 驱动 | `nvidia-smi` 显示的 CUDA Version ≥ 12.6（实测：驱动 580）。不需要另装 CUDA toolkit，torch 和 warp 都自带运行时 |
| gcc | `torch.compile`（Triton）运行时要编译一小段 C 代码 |
| Python | 推荐装 [uv](https://docs.astral.sh/uv/)，它会自动下载 Python 3.10。不用 uv 时需要系统的 `python3.10`、`python3.10-venv`、`python3.10-dev` |
| GitHub 访问 | 这是私有仓库：服务器上要配好 SSH key，或者先 `gh auth login` |

**步骤**

```bash
# 0. （推荐）装 uv：装到 ~/.local/bin，不需要 root
curl -LsSf https://astral.sh/uv/install.sh | sh        # 或者 pip install --user uv

# 1. 克隆
git clone git@github.com:Urutoramarin2000/supermimic-A3.git
cd supermimic-A3

# 2. 在仓库里建 .venv 并安装依赖，最后自检 GPU 和 torch.compile
bash setup_env.sh
```

最后一行打印 `setup ok: .../supermimic-A3/.venv` 就说明装好了。之后所有命令都用 `.venv/bin/python`，或者先 `source .venv/bin/activate`。

可以用环境变量调整安装方式：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh   # PyPI 镜像（Linux 上 PyPI 的 torch 2.7.1 就是 CUDA 12.6 版）
TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_env.sh       # 从 PyTorch 源装别的 CUDA 版本，例如 RTX 50 系 / B200 需要 cu128（未实测）
PYTHON=/usr/bin/python3.10 bash setup_env.sh                               # 指定解释器
```

## 准备机器人模型和动作数据

机器人模型（`a3_pingpong.xml` + meshes）和动作数据（172 个 npz）都不在 git 里。

**第 1 步：打包。** 在已经有这两样东西的机器上运行：

```bash
bash tools/pack_assets.sh /path/to/a3_pingpong.xml /path/to/a3_31dof_paddle_50hz
# -> supermimic_a3_assets.tar.gz
```

**第 2 步：拷到新服务器。**

```bash
scp supermimic_a3_assets.tar.gz <new-server>:~/supermimic-A3/
```

**第 3 步：在新服务器的仓库根目录解压。**

```bash
tar xzf supermimic_a3_assets.tar.gz
```

解压后的布局如下，正好是配置里的默认路径：

```text
assets/a3_racket/a3_pingpong.xml
assets/a3_racket/meshes/*.STL
data/a3_31dof_paddle_50hz/*.npz
```

- 配置里的相对路径都按仓库根目录解析，与当前工作目录无关。
- 放在别处时，加 `--xml /path/to/a3_pingpong.xml --motion_dir /path/to/motions`。
- npz 格式见 [`data/README.md`](data/README.md)。训练只需要 `qpos`（50 Hz，[T, 38]），`hit_frames` 用于评估拍心误差，所以换别的 A3 动作数据也能直接训练。

## 自检

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.check_model
```

自检做三件事：

1. 比较 torch FK 和 MuJoCo 的结果，误差应在 1e-6 量级。
2. 检查状态增量能否往返还原。
3. 用 256 个 MuJoCo Warp env 开环 PD 回放参考动作。

最后打印 `CHECKS PASSED`。

第一次运行要编译 MuJoCo Warp 的 kernel，需要几分钟，结果缓存在 `~/.cache/warp`，之后就快了。

## 训练

```bash
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

$PY -m supermimic.train --preset smoke  --out_dir runs/smoke       # 功能测试，约 4 分钟：训练 300 iteration 并保存
$PY -m supermimic.train --preset paper  --out_dir runs/paper       # 论文超参（Table A.5）
$PY -m supermimic.train --preset racket --out_dir runs/racket_w1   # 论文超参 + 球拍奖励，5 万 iteration，约 12 小时（上表的 racket_w1）
```

| 预设 | 内容 |
| --- | --- |
| `paper` | 论文 Table A.5 的单卡配置：4096 env、replay 1000、B=4096、H=32、TF32，学习率 2e-4 在 25 万 iteration 内衰减 |
| `racket` | `paper` + 球拍位置/姿态奖励（`--w_racket_pos 1 --w_racket_ori 1`），学习率 4 万 iteration 衰减，共 5 万 iteration，每 2500 存一次 |
| `4090` | bf16 网络 + replay 500，用来省显存，可以叠加：`--preset racket,4090` |
| `smoke` | 1024 env、300 iteration 的快速测试 |

**常用参数**

- 任何超参都能在命令行覆盖，字段见 `supermimic/config.py`。例如 `--horizon 16 --num_envs 2048 --precision bf16 --compile false`。
- `--max_minutes 60`：到时间就保存并停止。
- `--resume runs/x/model_20000.pt`：从 checkpoint 恢复网络、优化器和重置采样器，iteration 接着数。replay 会重新预热。

**多卡**（数据并行，梯度平均，学习率 ×√N）：

```bash
.venv/bin/torchrun --nproc_per_node 4 -m supermimic.train --preset paper --out_dir runs/paper_x4
```

每张卡各跑 `num_envs` 个 env。目前只用 gloo 在单卡上起两个 rank 验证过梯度平均和学习率缩放（`SM_DIST_BACKEND=gloo`），真正多卡默认走 nccl。

**输出**（`--out_dir` 下）：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本次训练的完整配置 |
| `log.jsonl` | 每 `log_every` 个 iteration 一行，同时打印到终端 |
| `model_N.pt` | 每 `save_every` 个 iteration 存一次，结束时再存一次。包含网络、优化器、采样器状态和配置 |

`log.jsonl` 里值得看的字段：

- `ep_norm_return`、`terminated`：仿真里的跟踪质量；
- `wm_err1`、`wm_errk`：WM 单步和多步预测误差；
- `imag_reward`：想象轨迹的奖励；
- `it_per_s`、`t_collect` / `t_wm` / `t_policy`：速度；
- `mem_gb`：显存峰值。

## 评估

评估按论文附录 F.1 的协议，在训练动作上逐段测试：

- 每段从第 0 帧开始播放一次；
- 摩擦 1.0，不做域随机化，观测和动作都不加噪声；
- 触发训练时的终止条件就算失败：根（骨盆）高度和参考相差 0.25 m 以上，或者根的竖直朝向分量（旋转矩阵的 zz 元素）和参考相差 0.8 以上。

```bash
# 评估一个 checkpoint -> runs/racket_w1/model_50000_eval.json（汇总）和 .csv（逐段）
$PY -m supermimic.evaluate --ckpt runs/racket_w1/model_50000.pt

# 训练完，自动评估每个 checkpoint 并打印汇总表
bash scripts/train_eval.sh runs/racket_w1 --preset racket
GPU=1 bash scripts/train_eval.sh runs/smoke --preset smoke
```

评估默认使用 checkpoint 里保存的机器人模型和动作路径。用本仓库训练的 checkpoint 存的是相对路径，换机器也能直接评估。如果 checkpoint 存的是绝对路径（比如在别的机器上用绝对路径训练），加 `--xml` 和 `--motion_dir` 指向本机的文件。

| 指标 | 含义 |
| --- | --- |
| `failed` | 失败段数 |
| `norm_return_mean` | 每段的回报除以“每步最大回报 × 步数”，再对所有段取平均；1 表示每一步都完美跟踪 |
| `mean_body_pos_err_m` | 14 个跟踪 body 的位置误差（每步取 RMS），先在段内对存活步取平均，再对所有段取平均，单位米 |
| `racket_hit_err_m` | 在标注的击球帧上，机器人拍心与参考拍心的距离，对全部击球取平均，单位米 |

## 实现说明

论文没写清楚的地方，下面是这里的选择：

- **WM 状态** = MuJoCo 的 qpos/qvel。论文 demo 的 worker 里，网络状态和仿真状态互相拷贝的正是 nq+nv 维。
  - 网络输入用平移和 yaw 不变的特征。
  - 输出是当前 heading 系下的状态增量（根位移、根旋转向量、关节角、速度），再积分回 SE(3)×Rⁿ。
  - 观测、奖励、终止都由状态解析计算（论文说每个 head 可以学，也可以解析计算），所以 WM 的损失里只有状态项，误差按固定的 nominal 尺度归一化。
- **软终止**：论文学一个终止概率 head（BCE）。这里把论文的终止条件直接用 sigmoid 平滑成可微的 d̂。
- **目标表示**：K=10 个未来帧（论文只写了“K=10 future reference frames”）。每帧包含：
  - 参考关节角（29）和关节速度（29）；
  - 机器人 heading 系下的参考根位置（3）；
  - 参考根的相对姿态（6D）。
- **A3 适配**：
  - 头部 2 个关节不进策略，和部署一致：策略输出 29 维，观测 93 维。
  - 增益默认用部署值（`--gains deploy`，action scale 0.25）。`--gains beyondmimic` 是论文 / BeyondMimic 的低增益：按 armature 计算，10 Hz、ζ=2。
- **球拍奖励**（`racket` 预设，论文没有）：对参考动作做 FK 得到球拍 site，用它的位置和姿态做奖励。
- **工程优化**（论文没有）：
  - 用 `torch.compile` 融合想象一步和 WM 的窗口损失；3×3 旋转乘法写成逐元素乘加。在 4090 上，一次策略更新从 1.37 s 降到 0.37 s。
  - 可选 bf16 和 activation checkpoint（`--checkpoint_imagination true`），用来省显存。
- **没有实现**：
  - 速度指令等下游 encoder（附录 B/G）。Actor 已经拆成 encoder 和 decoder，留了位置。
  - 论文的 PPO 基线。
  - 关节速度限制。

## 常见问题

**第一次训练要等一会儿才开始**

要编译两样东西，之后都会读缓存：

- MuJoCo Warp 的 kernel，缓存在 `~/.cache/warp`；
- `torch.compile` 的想象步和 WM 损失，缓存在 `/tmp/torchinductor_$USER`。

**`torch.compile` 报错**（找不到 `gcc` 或 `Python.h`）

装上 `gcc` 和 `python3.10-dev`；用 uv 装的 Python 自带头文件。也可以加 `--compile false`，训练本身不受影响，但策略更新会慢约 3.7 倍。

**显存不够**

依次尝试：

1. `--preset racket,4090`（bf16 + replay 500）；
2. `--checkpoint_imagination true`；
3. 减小 `--num_envs`、`--imag_batch`、`--wm_batch`。

**下载慢或连不上**

- PyPI：设置 `PIP_INDEX_URL`，指向镜像。
- uv 下载 Python 慢：设置 `UV_PYTHON_INSTALL_MIRROR`，或者用 `PYTHON=/usr/bin/python3.10` 指定系统里的 Python。

**选 GPU**

用 `CUDA_VISIBLE_DEVICES=<id>`。代码里默认的设备是 `cuda:0`，也就是可见设备里的第一张卡。

> 本仓库只有训练和评估代码，没有附开源许可证。机器人模型和动作数据都不在仓库里。
