# TTO scaffold

本目录是清理后的最小 TTO 骨架。唯一入口为 `python -m tto.run`，
唯一配置协议为 `tto-scaffold-v1`（`configs/tto.yaml`）。
当前实现原生观测预检、冻结原模型的精确初始化、身份检查和新状态保存/恢复。
空间配准和优化算法尚未实现，调用这些阶段会明确失败，退出码为 2。

## 目录与职责

```text
tto/
├── run.py / config.py          唯一 CLI 与严格配置
├── pipeline.py / state.py      唯一阶段控制器与患者状态合同
├── dataset.py / prior.py       真实观测合同与原 NeuralFFDNet 桥接
├── io.py / gpu.py              原子保存、身份检查、GPU 资源工具
├── geometry/
│   ├── pose_math.py            SO(3)、刚体变换；无局部搜索
│   ├── guard.py                原三维/4D 几何审核
│   └── motion_metrics.py       独立运动指标；无冻结场传播或准入门控
├── supervision/
│   ├── slicing.py              复用 utils.slice_supervision 原切片核
│   ├── observations.py         原生二维证据选择、物理时间映射
│   ├── losses.py               无 GT 的原生 Dice/BCE/Boundary
│   └── evaluation.py           原生独立评价
├── alignment/
│   ├── global_alignment.py     患者整体 SE(3) 配准合同（待实现）
│   └── chamber_alignment.py    腔室初始候选合同（待实现）
├── optimization/
│   ├── shape.py                原 64D Shape 搜索合同（待实现）
│   ├── motion.py               原 32D Motion 搜索合同（待实现）
│   └── joint.py                Shape/Motion/整体位姿联合合同（待实现）
├── evaluator.py                独立原生渲染与汇总底层
├── synthetic_oracle.py         隔离隐藏 GT、保存终点后的独立评价
├── visualization.py            新状态的原生叠图与 4D 网格预览
├── experiments.py              阶段能力注册表，无第二运行入口
└── validate.py                 新状态合同检查，无第二运行入口
```

建议的后续阶段顺序为初始化 → 整体配准 → 腔室初始候选 → Shape →
Motion → 联合优化。这里定义职责，不假设这些算法已经完成。

## 已确定的空间语义

2026-10-17 确认：**逐腔室位移只用于生成初始对齐候选**。
候选必须回到原始 Decoder 的 Shape64 / Motion32 表达，并接受四腔解剖审核。
最终 Mesh 不附加腔室静态位移，状态合同也没有这类参数。

`global_transform` 是唯一的整体刚体变换，以列向量约定表示
`x_patient = R @ x_canonical + center_mm`，跨四腔和 24 帧共享。
`trajectory` 保存原始 Decoder 的 canonical mm 输出，
`patient_trajectory()` 才施加整体变换。
`observation_pose` 表示患者坐标中的三切面位姿，未经配准时为
`None`；初始化的单位整体变换不能被解释为已完成配准。
原生评价和叠图需要显式提供合法切面位姿。

## 数据、模型与监督边界

- 保留真实 Mask/SDF 合同、24 个模型相位、各视图原生尺寸和物理像素间距。
  原生 Mask 不缩放；只有编码器输入经过原 MaskTransform。
- 真实监督使用选定 RR 内的原始帧，每条原生观测只使用一次；
  根据真实时间插值相邻模型网格。二值监督不来自 GT Mesh。
- 标注来源、信任权重、时间误差和未参与优化的单元均保留。
  切片不存在时报告缺失；不计算质心/平面距离捕获力。
- `prior.py` 保留冻结 NeuralFFDNet、全部已训练空间条件、
  原始递归 Motion Decoder，以及初始化与原 forward 完全一致的检查。
- Shape64 和 Motion32 的职责独立定义；不继承初始 latent 的小范围搜索、
  Frozen-Field Motion-Lock、旧运动保护阈值或统一回退。
- `geometry/guard.py` 保留三维/4D 几何审核；后端不可用时为
  `UNVERIFIED`。初始化不运行几何准入，保存结果明确为
  `geometry_status=NOT_EVALUATED`。
- 隐藏 GT 由 `hidden_patient()` 单独返回，不进入观测或优化接口；
  `evaluate_hidden()` 只读取已保存的新状态，并核对样本身份。
  新的合成恢复测试编排与准入门控尚未实现。
- `visualization.py` 已适配新状态，保留原生轮廓叠图和 24 相位 Mesh
  预览；旧求解器历史曲线已移除。

## 运行

当前服务器已有依赖的解释器：

```bash
export PATH="/home/yilong/.conda/envs/4dheart/bin:$PATH"
python -m tto.run --describe
python -m tto.run --patients C01 --stage preflight
python -m tto.run --patients C01 --stage initialize
python -m tto.run --patients C01 --stage initialize --resume
python -m tto.run --stage validate --state outputs/tto_scaffold_v1/patients/C01/initialized.pth
```

可使用 `--config`、`--data-root`、`--output-dir` 指定配置和输出。
默认不加载原始影像；导出叠图前需要启用 `data.load_images`。
预检不会加载模型，初始化会核对 checkpoint SHA256。
当前 CLI 的初始化只支持 CPU；原 `gpu.select_gpu()` 的实测峰值准入工具
保留独立可测，GPU 工作负载探测尚未接入新状态机。

状态使用 `tto-state-v1`，身份包含患者、输入、checkpoint、配置和 TTO
源码摘要。恢复会拒绝身份不匹配的状态；旧协议状态不会隐式迁移。
已有初始化需显式 `--resume`，不覆盖旧结果。
初始化成功只证明工程合同，不代表重建、配准、几何或临床验证成功。

## 本地清理记录

清理基于服务器本地文件，而非远端清单。检查时
`tto/inverse_v2/` 没有 `.py` 源码，仅有未跟踪的字节码缓存。
缓存已从活跃包移除；原文件及逐文件 SHA256 清单保存于
`outputs/tto_cleanup_20261017/inverse_v2_local.zip` 和
`outputs/tto_cleanup_20261017/local_inventory.json`。
这些缓存不能视为已审阅的 v2 源码，也未被移植进新架构。

旧两套 pipeline、Shape/Pose/Joint 优化器、Motion-Lock、
重复 synthetic prior、专用损失与多套 YAML 已移除。
八个依赖旧状态/配置/优化器的脚本一并退役：
`analyze_real_tto_joint_hidden.py`、`analyze_real_tto_joint_pose.py`、
`audit_real_tto_joint.py`、`check_real_tto_joint_batch.py`、
`diagnose_real_tto_dice.py`、`profile_real_tto_joint_original.py`、
`report_real_tto_joint_progress.py`、`run_real_tto_coverage_rescue.py`。
仍可独立读取历史产物的审核脚本保留。旧实现通过 Git 历史查阅；
仓库根目录两份旧 TTO 实施报告已标注为历史协议。
`configs/real_tto_prior_calibration.json` 作为历史测量保留，
新入口不读取其中的 Pose、Shape 搜索范围或运动阈值。
既有实验输出和 checkpoint 保留。原 `net/`、`data/`、`utils/`
及训练实现不属于本次 TTO 清理的修改范围。

## 验证

```bash
python -m unittest discover -s tests -p 'test_real_tto*.py' -v
python -m unittest discover -s tests -p test_tto_scaffold.py -v
```

测试覆盖原生矩形网格与物理映射、独立渲染、24 帧梯度、
原始 Dice/BCE/Boundary 数值一致性、分数时刻插值与患者梯度隔离、
GPU 资源工具、无捕获缺失切片、新状态身份隔离，以及真实 checkpoint
的原 forward 一致性与 Shape/Motion 梯度。checkpoint 不可用时，
真实模型测试会明确跳过。

2026-10-17 本地验收：完整 unittest 套件 **102 项通过，零失败、零跳过**。
C01 已通过正式 CLI 的原生预检、224 网格原模型初始化、恢复及新状态检查；
四项原 forward 差异均为 0。状态仍明确标为未配准、未审核几何、未执行优化。
日志、缓存清单和汇总位于 `outputs/tto_cleanup_20261017/`，
汇总文件为 `verification.json`。
