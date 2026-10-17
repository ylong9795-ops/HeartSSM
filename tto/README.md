# 真实全周期 Shape–Pose Joint TTO

正式协议为 `real-tto-joint-v1`，入口是 `python -m tto.run`。原 NeuralFFDNet 的权重、64D Shape / 32D Motion 和24帧固定拓扑保留；主阶段优化患者独立的 Shape latent 与三视图共享周期 Pose。完整实现和已执行证据见仓库根目录 [REAL_TTO_JOINT_IMPLEMENTATION.md](../REAL_TTO_JOINT_IMPLEMENTATION.md)。

## 输入目录

`RealTTODataset` 支持两种输入：

1. 仓库已有的真实数据导出目录：患者原始DICOM、每帧标签、物理标定和手工ED/ES记录。读取器核对DICOM厘米标定与导出mm间距，选择包含ED且完整覆盖的一个RR。
2. 每位患者一个 `tto_input.json` 的可移植合同。患者ID就是数据根目录下的单层文件夹名；ID不能重复，不需要写入Python源码。

建议为新数据使用可移植合同：

```text
data_root/
  NEW01/
    tto_input.json
    A2C.npz
    A3C.npz
    A4C.npz
    A2C_images.npy
    A3C_images.npy
    A4C_images.npy
```

清单中的相对资源路径相对于患者目录。CLI的相对配置、数据根目录和输出路径按仓库根目录解析；命令从仓库根目录执行。

## 清单字段

顶层字段：

| 字段 | 类型与含义 |
|---|---|
| `version` | 固定为 `real-tto-input-v1` |
| `patient_id` | 与患者目录名一致 |
| `cycle_start_s` | 所选完整RR起点，秒 |
| `cycle_duration_s` | 正的RR时长，秒 |
| `anchor_time_s` | 模型参考相位0的估计时间，必须落在所选RR内 |
| `views` | 顺序固定为 A2C、A3C、A4C 的3个记录 |
| `acquisition_status` | 真实采集/同步证据的描述；不代表真实三维标定已经验证 |
| `initial_pose` | 可选，含 `R` `(3,3,3)` 和 `center_mm` `(3,3)`，依次为三个视图的估计旋转与中心 |

时间应有限，三个视图必须使用同一时间基准。ED锚定是估计，不能直接等同于合成frame0的生理含义。内部字符串 `synthetic source phase mapping verified` 是隐藏GT合成适配器的专用标记，不能用于声明真实患者已经具备源合成时相真值。

未提供 `initial_pose` 时使用合成TRAIN校准文件的共同初值。若有更好的初步估计，可以在清单中提供三个有限SO(3)旋转矩阵和与模型Mesh同一物理坐标系的mm中心；不传入逐帧Pose。它只作为未知Pose的搜索初值，后续仍执行多起点初始化和持续Shape–Pose联合优化，不是 `pose_gt`。

每个视图的字段：

| 字段 | 类型与含义 |
|---|---|
| `name` | A2C / A3C / A4C |
| `masks_npz` | NPZ路径，其中 `masks` 的形状为 `(T,4,H,W)` |
| `mask_files` | 与 `masks_npz` 二选一；T个NPZ，每个 `masks` 为 `(4,H,W)` |
| `channels` | Mask通道名称，必须恰为 LV、LA、RV、RA；推荐按此固定顺序导出 |
| `times_s` | T个严格递增的原始观测时间，秒 |
| `spacing_yx_mm` | `[行间距, 列间距]`，两个正有限数，mm |
| `annotated` | `(T,4)` 布尔值，固定按 LV、LA、RV、RA 声明是否有标签 |
| `sources` | `(T,4)` 来源字符串，顺序与 `annotated` 一致 |
| `frame_ids` | 可选，T个原始帧标识；默认0至T−1，建议显式保存原始标识 |
| `images_npy` | 无DICOM输入且 `data.load_images: true` 时必需；`(T,H,W)` 或 `(T,H,W,3)` 原图 |

Mask必须是0/1的独立通道，不能传入一张0/1/2/3/4互斥类别图代替四通道。数组禁止依赖pickle对象。一个视图内H/W固定，不同视图、不同患者可以不同；原图必须与该视图Mask的时间和像素网格完全对应。`annotated=false` 的右心通道不是空腔负标签；已有数据的A2C/A3C默认仅LV/LA有效。

当前读取器只根据 `channels` 重排Mask，不重排 `annotated` 和 `sources`；后两者必须采用上述固定腔室顺序。为避免歧义，建议所有数组及 `channels` 一致使用 LV、LA、RV、RA。

图像方向与Mask方向必须一致。物理网格使用当前原生图像中心 `(W/2,H/2)`，保持原生spacing；不按心腔bounding box重新缩放。SDF由读取器以mm计算，内负外正，不能将已有归一化SDF当成原生Mask传入。

原生时间范围必须包住全部24个目标输入时刻。读取器不会在已提供的时间范围之外外推。允许保留所选RR之外的原始帧来支持输入时刻插值，但临床Joint监督只使用 `cycle_start_s <= time < cycle_start_s + cycle_duration_s` 内的每个原生标注Cell一次。

## 从已有数组生成清单

以下示例只写输入适配清单。各NPZ/NPY必须先由真实数据导出，代码不会生成标签或假设新的时间、间距和来源。

先在患者目录保存 `acquisition.json`，包含上表的 `cycle_start_s`、`cycle_duration_s`、`anchor_time_s` 和真实 `acquisition_status`。每个视图的NPZ包含 `masks`、`times_s`、`spacing_yx_mm`、`annotated`、`sources`、`frame_ids` 数组，四通道按LV/LA/RV/RA排列。

```python
import json
from pathlib import Path
import numpy as np

folder = Path("path/to/data_root/NEW01")
timing = json.loads((folder / "acquisition.json").read_text())
contract = {**timing, "version": "real-tto-input-v1",
            "patient_id": folder.name, "views": []}
for view in ("A2C", "A3C", "A4C"):
    with np.load(folder / f"{view}.npz", allow_pickle=False) as packed:
        contract["views"].append({
            "name": view,
            "masks_npz": f"{view}.npz",
            "images_npy": f"{view}_images.npy",
            "channels": ["LV", "LA", "RV", "RA"],
            "times_s": packed["times_s"].tolist(),
            "spacing_yx_mm": packed["spacing_yx_mm"].tolist(),
            "annotated": packed["annotated"].tolist(),
            "sources": packed["sources"].astype(str).tolist(),
            "frame_ids": packed["frame_ids"].tolist(),
        })
(folder / "tto_input.json").write_text(
    json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
```

另一个可选路径是在顶层指定 `dicom`，并为每个视图提供 `region` 的 `x0/y0/width/height`。当前DICOM图像读取分支按观测数组索引取DICOM帧；该路径要求数组第j行对应DICOM第j帧。对于各视图独立采集、稀疏帧或非连续原始帧ID，应使用已经逐帧对应的 `images_npy` 路径。`frame_ids` 保留来源标识，不会在DICOM读取分支中重新映射帧。

## 标签来源与时间可信度

默认配置的来源权重：

| 来源 | 权重 |
|---|---:|
| `manual_ed_es`、`manual_review` | 1.0 |
| `patient_adapted_model` | 0.25 |
| `temporal_fallback` | 0.1 |
| `not_applicable` | 0 |

`synthetic_observed` 的1.0权重用于合成观察，不应把真实自动标签改名为该来源。未知来源必须在新版本配置的 `data.source_weights` 中明确声明，不能按当前Dice决定来源、删除困难Cell或调整可信度。这里的来源权重不等于最终置信度：当前真实相位因子为0.25，因此人工标签最终为0.25、自动标签为0.0625，均低于高可信阈值0.5。共同相位因子在归一化均值中抵消，不代表整体二维监督相对于先验被削弱。

## 对新患者运行

需要训练模型环境及原有几何依赖。当前服务器使用 `/home/yilong/.conda/envs/4dheart/bin/python`，几何扩展通过 `PYTHONPATH=outputs/real_tto_motion_locked_20261016/_geometry_site` 加载。若使用其他环境，应安装相同依赖；版本见输出的 `model_identity.json` 和 `tto/requirements.txt`。

同时保留仓库的 `synthetic_bundle_20261009` 源几何包及其 `configs/geometry/source_support.json`。正式适配器直接复用其中的 `heart4d.geometry` 谓词和固定房室帽支持，并核对源资产身份。

```bash
python -m tto.run --help
python -m tto.run --config configs/real_tto_joint_v1.yaml --patients NEW01 --data-root path/to/data_root --output outputs/new_patient_joint_v1 --stage preflight --device auto
python -m tto.run --config configs/real_tto_joint_v1.yaml --patients NEW01 --data-root path/to/data_root --output outputs/new_patient_joint_v1 --stage all --device auto --resume
```

使用新版本输出目录。缺少匹配的合成门控时，`--stage all` 会先执行正式隐藏GT验证，不会直接跳过门控进入真实优化。`preflight` 包括实际GPU工作负载与几何依赖检查，不只是读取JSON；资源不足返回 `WAITING_FOR_GPU_RESOURCES`。`--batch-size` 控制mini-batch，`--device auto` 在建立主CUDA上下文前选择设备。

首次门控还需要 `synthetic.data_config` 指向的合成TRAIN资产及与checkpoint匹配的 `prior.calibration`，不能仅凭checkpoint文件名跳过这些依赖。可通过 `synthetic.validation_gate` 引用已有正式门控，但必须通过科学配置、源码和种子身份核验；改变数据根目录等科学配置后，不应把旧门控复制成新协议的通过证据。

正式Joint阶段为 `initial → pose_init → joint → evaluate → export`；也可通过 `--stage` 单独恢复相应阶段。`--resume` 恢复Shape、Pose、Adam、最优合法候选、冻结Motion场及随机状态，输入或科学协议身份不同会拒绝复用。`--arm` 支持七种正式对照；比较结果要求共同初始化和预声明预算，不能把已优化端点作为另一臂的初始化。

## 时间映射与输出的解释

编码器使用24个模型槽，其Mask由原生SDF的时间插值得到。临床Joint损失使用原始二值标签，在对应的相邻Mesh时刻插值切片，包含23→0接缝；二者是不同用途。原始观察较稀疏时不保证每个模型槽都有直接标签，但仍解码和审核全部24帧。

每位患者在 `patients/<ID>/states/`、`export/`、`evaluation/` 保存状态、Mesh、Pose、latent、冻结场、逐Cell Native Dice和几何/Motion记录；检查点位于 `checkpoints/<ID>/`，可视化位于 `visualizations/<ID>/`。`export/completed.json` 与 `states/export.pth` 是提交端点的共同标记，单有中途检查点不代表完成。

参考/保存Mesh、跨腔和连续线性轨迹的验收与完整空间FFD场证书分开报告，后者当前为UNVERIFIED。Motion保持是相对于初始预测，不能证明初始生理运动正确。真实数据无3D Mesh/Pose GT，Native Dice改善不能单独证明真实三维Shape或绝对Pose更准确。当前队列和消融状态以版本输出目录的中文报告及独立审核为准。
