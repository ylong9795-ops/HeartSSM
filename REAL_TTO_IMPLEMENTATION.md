> Historical protocol (retired 2026-10-17). Commands and acceptance rules below
> describe the old implementation. Current entry and contracts: [tto/README.md](tto/README.md).

# HeartSSM 正式真实数据 TTO 实现

统一入口是 `python -m tto.run`。这是共享冻结合成先验、逐患者独立优化的 TTO；没有训练 PopulationAdapter，也没有修改原始模型、FFD 或几何准入阈值。

## 源码与调用关系

正式源码位于 Git 可见的 `tto/`，不依赖 `outputs/.../source` 中的临时 Python 脚本。配置为 `configs/real_tto_v1.yaml`，短程工程验收配置为 `configs/real_tto_validation.yaml`，合成 TRAIN16 的派生统计为 `configs/real_tto_prior_calibration.json`。测试在 `tests/test_real_tto.py`。

| 文件/类 | 输入 → 输出及职责 |
|---|---|
| `dataset.py / RealTTODataset` | 患者目录或 `tto_input.json` → `PatientObservations`，包含原始 Native Mask、原图、腔室可用性/来源/信任权重、原生 spacing、ED/ES、RR、24相位索引及 model-to-native 映射。每个视图独立物理变换，跨患者只堆叠模型网格。 |
| `synthetic_prior.py / FrozenSyntheticPrior` | 指定 checkpoint → 原 `train_neural_ffd.load_checkpoint` 严格加载的 NeuralFFDNet。`encode_real_observations`、`decode_initial_shape`、`decode_initial_motion`、`export_motion_fields`、`reconstruct_4d`。全部网络参数 eval/frozen；Shape 解码仍保留对 latent 的梯度。 |
| `pose_optimizer.py / MultiViewPoseOptimizer` | 固定参考 Mesh、原生多视图观察、初始 Pose → 每患者独立三个 SO(3)/mm 切面、多起点候选、粗到细轨迹、选中 Pose。随机起点只依赖患者 ID 和固定 seed，不依赖批次成员。 |
| `slice_loss.py / NativeSliceLoss` | 可变 Native 网格、Mesh、估计 Pose → 每患者二维 loss。调用现有 `batched_plane_segments` 与 `batched_signed_segment_distance`；切面缺失显式 capture，padding 不参与 loss。 |
| `shape_optimizer.py / LatentShapeOptimizer` | 固定 Pose、64D latent、合成标准差、初始三维 Shape 与运动场 → 最优合法 latent/24帧 Mesh、完整候选轨迹。约束未观察区域漂移、全顶点漂移与标准化 latent；每候选执行 Motion 和完整几何审核，失败同时回滚参数和 Adam。 |
| `motion_lock.py / FrozenMotionPropagator` | 初始 fine control fields、更新后的参考 Shape → 原 `model._deform` 递归传播的24帧。fine0 属于 Shape，不作为增量重复应用。比较各腔位移、方向、幅度、容积函数、速度/加速度/jerk 与23→0。 |
| `geometry_guard.py / GeometryGuard` | 实际四腔表面 → 分开的 Shape、24保存相位、跨腔、连续周期审核。复用 `SourceAVCapPolicy`、原 CGAL/ipctk/VTK；缺失或异常是 `UNVERIFIED`。没有将 GeometryHeartFFDNet 的控制证书套到旧 NeuralFFDNet。 |
| `joint_optimizer.py / GuardedJointOptimizer` | 已完成的独立 Pose/Shape/Motion-Lock → 小步 Joint。只优化 Shape latent/Pose，候选重新生成24帧，重新审核 Motion/三维/连续周期；无合法基准时明确不准入。 |
| `evaluator.py / ExactNativeEvaluator` |24帧 Mesh、固定估计 Pose、原始二值标签 →192个有效 Cell（完整常见合同），每 Cell 保留患者/视图/腔室/相位、原生帧、时间误差和标注来源，汇总 mean/min/P01/P05、≥.95/.98 数量、Shape前后变化。矩形网格保持物理 pixel0，用原 `section_mask` pad/crop 语义。 |
| `pipeline.py / RealTTOPipeline` | 配置、任意合规患者 ID → `load_patient → load_frozen_prior → initialize_mesh → optimize_pose → optimize_shape → apply_motion_lock → guarded_joint → evaluate → export`。各阶段有独立状态、原子检查点、身份核验和失败记录。 |
| `run.py` | 正式 CLI，支持 all/preflight/initial/pose/shape/motion_lock/joint/evaluate/export、GPU、Batch、checkpoint、输出目录和恢复。 |
| `io.py / config.py` | 原子 JSON/torch state、SHA256、配置版本校验、路径解析。恢复校验患者输入、科学配置、seed、源码、校准和模型身份。 |
| `validate.py` | 比较正式 CLI 已产生的 Batch/单独运行所有阶段状态、冻结场、梯度和患者身份；不产生替代实验。 |

## 执行与依赖

当前主机 Python 为 `/home/yilong/.conda/envs/4dheart/bin/python`。依赖清单为 `tto/requirements.txt`，包括 PyTorch、numpy、scipy、opencv、trimesh、pydicom、PyYAML、matplotlib，以及仓库原 geometry 后端 `cgal==6.0.1.post202410241521`、`ipctk==1.6.0` 和 VTK。几何后端未安装时不能把未验证结果当合法结果。当前主机已有隔离的几何依赖目录，可通过 `PYTHONPATH=outputs/real_tto_motion_locked_20261016/_geometry_site` 使用；正式包不导入该目录内的旧实验源码。在新环境按原 bundle geometry 依赖安装相同版本。

```bash
python -m tto.run --config configs/real_tto_v1.yaml --patients C01 C02 --stage preflight
python -m tto.run --config configs/real_tto_v1.yaml --patients C01 C02 --stage all
python -m tto.run --config configs/real_tto_v1.yaml --patients C01 C02 --stage shape --resume
python -m tto.run --config configs/real_tto_v1.yaml --patients NEW_ID --data-root /path/to/input --gpu 4 --batch-size 1 --output-dir outputs/new_tto --stage all
```

阶段命令会依次满足前置阶段。已有状态必须使用 `--resume`，否则拒绝覆盖。中途停止后重新使用同一科学配置和源码、相同输出目录及 `--resume`，会读取最近一次原子保存的优化器、当前/最佳 latent 或 Pose 以及完整轨迹。批次大小或 GPU 改变不改变科学身份。旧实验源码或科学配置改变时不会静默复用旧状态。

默认完整预算为 Pose 8起点/60更新（8→4 pixel stride）、Shape100更新、Joint30更新。短程验证为2起点/2次Pose、2次Shape、1次Joint，只用于工程 Gate，不代表完整拟合收敛。

## 新患者输入合同

原生仓库布局直接自动发现三个视图、DICOM、导出校准、逐帧索引和人工 ED/ES。没有硬编码队列名单或特定患者分支。其它数据提供 `/path/to/input/NEW_ID/tto_input.json`：

```json
{
  "version": "real-tto-input-v1",
  "patient_id": "NEW_ID",
  "cycle_start_s": 0.0,
  "cycle_duration_s": 1.0,
  "anchor_time_s": 0.0,
  "ED_frame": 0,
  "ES_frame": 8,
  "acquisition_status": "declared estimated geometry; absolute3D unverified",
  "views": [
    {
      "name": "A2C",
      "masks_npz": "A2C_masks.npz",
      "images_npy": "A2C_images.npy",
      "channels": ["LV", "LA", "RV", "RA"],
      "spacing_yx_mm": [0.8, 0.9],
      "times_s": [0.0, 0.5, 1.0],
      "annotated": [[true,true,false,false],[true,true,false,false],[true,true,false,false]],
      "sources": [["manual_ed_es","manual_ed_es","not_applicable","not_applicable"],["patient_adapted_model","patient_adapted_model","not_applicable","not_applicable"],["manual_ed_es","manual_ed_es","not_applicable","not_applicable"]]
    }
  ]
}
```

上例展示一个视图的字段；实际必须依次提供 A2C/A3C/A4C 三项。`masks` NPZ 数组为 `[N,4,H,W]` 独立二值通道，图像为 `[N,H,W]` 或 `[N,H,W,3]`，每视图允许不同原生 H/W、spacing、采样时间和来源。时间必须严格递增并覆盖目标24相位。`sources`、`annotated` 都按 `[N,4]` 的模型腔室顺序填写，未标注腔室不得作为空负标签。未知标签来源必须先在 YAML 中声明信任权重。

可提供 `initial_pose: {"R": [3,3,3], "center_mm": [3,3]}` 的实际数值数组。R 列为平面 x/y/normal，右手 SO(3)，坐标单位 mm，图像中心为 W/2,H/2。无初始 Pose 时采用同一冻结合成 TRAIN Pose 估计并明确标为未验证。原始 DICOM 显示时间同步不等于硬件绝对3D采集几何已恢复。

编码在固定 FOV 上对原生 SDF 作时间插值；独立评价比较最近的原始二值 Mask，逐 Cell 报告时间误差。自动标签权重 .25、时序补全 .1、人工标签1，评价按来源分层，不能称所有帧均为人工 GT。

## 结果、Gate 与局限

结果写入 `outputs/real_tto_v1/`（或指定目录）：每患者 `states/*.pth`、`checkpoints/*.pth`、原始输入映射、阶段状态、`evaluation/cells.csv`、`metrics.json` 和 `export/`（24帧固定拓扑 PLY、NPZ、Pose、latent、冻结控制场、几何证书、原图切面叠图及容积曲线）。不合法病例会保留，状态不能当作科学验收 PASS。

本次旧队列先验证45个可加载检查点再停止，记录在 `outputs/real_tto_v1/legacy_task_preservation.json`。旧结果没有删除或覆盖。新框架的数据变换修正了旧跨视图 padding 引入的物理中心偏移，因此不能预先宣称新旧结果逐位一致；将分别比较编码输入、初始 Mesh 与评价范围。

实际测试与患者 Gate 结果将由本轮执行证据更新。目前已实跑并通过7项 CPU 测试（`python -m unittest discover -s tests -p test_real_tto.py -v`）；C01/C02 已通过正式 CLI 数据/模型预检，确认原图加载、零可训练网络参数、几何后端可用。完整端到端、Batch/单独一致性、阶段/中途恢复与新患者验证仍在执行，尚不宣称 Gate2–4 或17例完成。

科学边界：二维稀疏观察仍欠定，合成先验不等于真实3D GT；运动锁定保证控制场不变并限制相对预测先验的退化，不证明真实生理运动正确；连续审核覆盖保存相位间顶点线性轨迹及周期接缝，不证明任意连续神经解码或未实现的全域 FFD 单射证书。Shape 扩展只允许64D latent，没有自由顶点形变。若初始4D不合法，Shape/Joint明确拒绝准入，保留诊断输出。
