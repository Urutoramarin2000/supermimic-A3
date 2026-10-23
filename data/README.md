# data/

参考动作不进 git，按下面的布局放在这里（`tools/pack_assets.sh` 打的包解压到仓库根目录后就是这个布局）：

```text
data/
└── a3_31dof_paddle_50hz/
    ├── 3_attack_Take_003_p01_a3_31dof_paddle.npz
    └── ...                  青瞳乒乓球动捕 → A3 retarget → 拍面修正 → 击球对齐 50 Hz，172 段，约 27 分钟
```

用别的位置时，训练和自检加 `--motion_dir /path/to/dir`；只用其中一部分时加 `--motion_glob '3_attack_*.npz'`。

## npz 格式

训练只读下面三个键，所以任何 A3 31dof 的 50 Hz 动作都可以直接用：

| 键 | 形状 | 内容 |
| --- | --- | --- |
| `qpos` | `[T, 38]` | 根位置 xyz（米，Z 朝上）+ 根四元数 wxyz + 31 个关节角（顺序同 `supermimic/robot.py` 的 `JOINT_NAMES`） |
| `fps` | `[1]`，可选 | 必须是 50；没有这个键时按 50 Hz 处理 |
| `hit_frames` | `[N]`，可选 | 击球帧号。评估时统计这些帧上的拍心误差；没有这个键时，这一段不参与拍心误差统计 |

动作名取文件名里 `_a3_31dof` 之前的部分，评估 csv 里按这个名字逐段列出结果。
