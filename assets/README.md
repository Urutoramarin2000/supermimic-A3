# assets/

机器人模型不进 git，按下面的布局放在这里（`tools/pack_assets.sh` 打的包解压到仓库根目录后就是这个布局）：

```text
assets/
└── a3_racket/
    ├── a3_pingpong.xml      Agibot A3 31dof + 右手球拍的 MJCF（含 right_racket site）
    └── meshes/              XML 里 <compiler meshdir="meshes"> 指向的 STL
```

用别的位置时，训练和自检加 `--xml /path/to/a3_pingpong.xml`。

代码对模型的要求（见 `supermimic/robot.py`）：

- 关节名和顺序与 `JOINT_NAMES` 一致；
- 有 `pelvis_link`、`torso_Link` 以及 `TRACKED_BODIES` 里的各个 body；
- 有 `right_racket` site；
- 有 `class="collision"` 的 default。

原有的 actuator 和 sensor 会被换成 PD 伺服。
