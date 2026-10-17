# 全周期 Shape–Pose Joint TTO

新协议为 `real-tto-joint-v1`。旧 `real-tto-v1` 的参考帧 Shape-only 和末端小范围 Joint 保留为历史基线。进程 3729062 已按用户要求终止，不会由新入口恢复。

## 源码与调用

- `tto/prior.py`: `FrozenNeuralFFDPrior` 用原 checkpoint loader 严格加载；保留原编码器条件；独立比较原 `model.forward` 和 TTO 初始化。
- `tto/motion.py`: `MotionPropagation` 区分原始 Decoder 与初始控制场冻结的原物理 FFD 递归；后者是主实验。
- `tto/losses.py`: `RealTTOLoss` 复用原可微求交和 `MeshSliceSupervisor._terms` 的 Dice/BCE/Boundary，不调用需要 GT 的损失。
- `tto/pose.py`: `MultiViewPoseInitializer` 用全周期观察做多起点搜索，采用 `R_initial Exp(omega)`、mm 平移，各视图跨时间共享 Pose。
- `tto/shape_pose_joint.py`: `JointShapePoseOptimizer` 在一个全周期目标下持续更新 Shape/Pose；患者独立 Adam；非法候选恢复两个参数组和 Adam。单独记录预声明的拒绝后学习率退避策略并支持恢复。
- `tto/geometry_guard.py`: `TTOGeometryGuard` 复用已有几何审核，缓存完全相同轨迹的已验证结果；阈值与数学内核不变。
- `tto/evaluation.py`: `RealTTOEvaluator` 分别报告全部标签、可信标签和实际监督集合；失败 Cell 不删除。
- `tto/joint_pipeline.py`: 正式 `RealTTOPipeline`。`tto.pipeline.RealTTOPipeline` 按配置版本分派到该实现。
- `tto/gpu.py`: CUDA 上下文建立前查询可见设备、UUID、显存、利用率和进程；用独立子进程测量实际前向/反向、多起点切片和几何审核，再按峰值与安全余量准入。
- `tto/synthetic_oracle.py`: 固定 TRAIN manifest 选择与校准病例不重叠的样本，移除优化输入的 Mesh/Pose GT。通过相同 `process()` 运行，提交端点后由独立 evaluator 读取隐藏真值。
- `tto/visualization.py`: 原生轮廓叠加、优化与容积曲线、固定相机 24 帧 Mesh 动画。
- `tto/experiments.py`: 顺序执行 E 主队列后再执行 A/B/C/D/F/G；仅复用来源一致的新协议初始化/Pose 搜索端点，禁止复用已优化 Shape/Joint 端点。

## 时间与监督合同

模型仍输出 24 个独立时相。真实优化使用所选完整 RR 内的每个原始标注帧一次，在实际时间对应的相邻模型 Mesh 间做线性插值，包含 23→0 接缝。原始二值 Mask 不插值。对应连续几何证书明确覆盖这些线性轨迹。

旧的 24 槽最近邻标签评价通过原 `ExactNativeEvaluator` 单独保存为 `legacy_model24_nearest_compatibility`，不改写历史结果。自动标签、人工标签和相位证据分别记录。真实 ED 与源 frame0 不保证等价，目前真实相位对齐采用估计并显式降权，不能将其称为高可信相位真值。C02 等 ES 落在所选 RR 外的情况记录在 `manual_phase_alignment`，不会混入另一心动周期。

主切片权重为原 checkpoint 的 Dice=1、BCE=0.1、Boundary=0.1、sigma=1 mm。失去切面时的 capture 是单独命名的先验；latent 分布、相对初始化偏移、未观测结构、FFD 控制、Pose 和运动保护均单独命名。初始预测不是真实 GT。

指定 checkpoint 的 `saved_config.slice` 未显式声明 `target_mode`。按本次归档原训练入口和 `MeshSliceSupervisor` 的默认参数解释，它采用 `binary_native_v1`；不能将该 checkpoint 标为已使用 paired-soft 训练。真实 TTO 显式设置 `binary_native_v1`，直接使用原生二值 Mask 和 mm SDF。代码另支持从 `vertices_gt` 构造连续软目标的 `paired_soft_geometry_v1`，但该分支不进入真实 TTO。若与这种 paired-soft 目标比较，必须注明目标定义变化，不能仅因 Dice/BCE/Boundary 分项名称相同就宣称目标等价。配置及本次源码的证据记录于 `independent_audit/slice_target_semantics.json`，不据此推断未归档的历史训练源码。

## 入口

需使用包含 Torch/CUDA 与已有 CGAL/ipctk/VTK 几何依赖的环境。当前服务器为 `/home/yilong/.conda/envs/4dheart/bin/python`，几何扩展安装目录为 `outputs/real_tto_motion_locked_20261016/_geometry_site`，通过 `PYTHONPATH` 加载。原 `tto/requirements.txt` 记录现有运行依赖。

```bash
python -m tto.run --config configs/real_tto_joint_v1.yaml --stage preflight --device auto
python -m tto.run --config configs/real_tto_joint_v1.yaml --stage synthetic_oracle --device auto --resume
python -m tto.run --config configs/real_tto_joint_v1.yaml --patients C01 C02 --stage all --device auto --resume
python -m tto.experiments --config configs/real_tto_joint_v1.yaml --resume
```

新患者放入符合 `tto/dataset.py` 合同的目录，通过 `--patients <ID> --data-root <目录> --output <新版本目录>` 指定。没有充分 GPU 显存时返回 `WAITING_FOR_GPU_RESOURCES`，不默认长时间使用 CPU。设备物理编号、UUID 和映射后的 CUDA 索引分别记录。

可移植 `tto_input.json` 的字段、NPZ/NPY维度、原生物理坐标、真实时间覆盖、标签来源和启动示例见 [tto/README.md](tto/README.md)。该说明也记录当前DICOM分支按数组索引取帧的限制；非连续原始帧应使用逐帧对应的 `images_npy`。

`--stage all` 在缺少合成门控时先执行隐藏 GT 验证。门控失败则不会继续真实优化；短 Pilot 不满足至少 16 例的正式门控。

## 当前证据与尚未完成部分

- `outputs/real_tto_joint_v1_20261016/legacy_inventory.json` 登记旧 68 个 checkpoint 的 SHA、旧源码和完成情况。
- `source_contract_precheck.json`：一个独立 TRAIN 样本的原输入与 TTO 输入前向逐项误差为 0；实际 checkpoint 启用 regional Shape，未启用 reference/dynamic spatial conditioning。
- `final_source_unit_tests.log`：当前源码 22 项全部通过，包括原损失数值、全周期梯度、患者隔离、真实 checkpoint、完整回滚、精确断点恢复、原生帧唯一使用和分数时刻梯度。
- `engineering_preflight_v2/`：C01/C02 Batch=2，304 个原生观察 Cell；两例全部 24 个 Mesh 时相的切片梯度非零。去除参考时相观察后，Shape、平移和旋转的纯切片损失梯度仍非零。实际 GPU allocator 峰值 2334 MiB，另有约 490 MiB CUDA context；设备为自动选择，非固定编号。
- `synthetic_pilot_v1/`：两个固定 TRAIN 病例各完成 2 次 Joint 更新，仅作历史工程 Pilot。两例几何/Motion 通过，Pose 误差总体下降；参考 Shape RMSE 中位数约 0.374483→0.375108 mm，不能据此宣称 Shape 精度改善。该 Pilot 自身不满足正式16例门控。
- `synthetic_hidden_gt/`：正式16例全部完成20次 Joint，共320次候选更新；独立重渲染、冻结场、参数回滚与256个必需可视化文件检查通过，旧68个检查点重新哈希全部未变。`independent_audit/synthetic_gate_recomputed.json` 从独立端点指标重算并核对正式门控，病例数、阈值、源码/配置、种子和中位数均一致。Pose 旋转误差中位数8.870681→0.982420°，平移2.294288→0.584410 mm；参考 Shape RMSE 中位数0.396715→0.398369 mm，略有上升。门控的 Shape 条件是保持在预声明合理误差范围，不能解释为精度改善。
- 正式合成 Native Dice≥0.98 Cell 从 Pose 初始化后的988/3072增至 Joint 后的1042/3072（33.92%），全部 Cell 达标病例为0/16。该 TRAIN 闭环结果不证明未见患者泛化，也不证明真实患者三维精度。

正式16例门控已通过并完成独立复核。C01/C02 Batch=2 已完成8起点、60步 Pose 初始化、20次全周期 Shape–Pose Joint 和最终导出，官方 Smoke 门控通过。`independent_audit/clinical_smoke_independent.json` 独立重渲染全部304个原生 Cell，与保存指标一致；最终 Mean=0.763785、Minimum=0.567857，≥0.98为0/304，全部Cell达标患者为0/2。两例相对初始先验的 Motion 与声明范围内几何验收通过，完整空间FFD场证书仍为未验证。真实高可信Cell为0，不能把可用低权重观察称为已证实同步的高可信监督。

`smoke_stage_resume.json` 证明两例共10个完成阶段状态在正式CLI恢复前后SHA逐一不变。首次运行归档runtime为3879.286秒，allocator实际reserved峰值2360 MiB；完成阶段恢复runtime为25.471秒。恢复会更新根summary，不能将其短耗时当成首次优化。中文两病例结果与失败范围见 `independent_audit/CLINICAL_SMOKE_REPORT.md`。

协调进程已自动启动完整17例主队列，复用C01/C02正式端点，其余患者、A/B/C/D/F/G匹配消融和最终全队列中文报告尚未完成。正式Joint预算保持20次；工程与Smoke门控通过不等于≥0.98 Dice目标达成，也不能证明真实三维Shape或绝对Pose精度改善。C02 A4C的Pose初始化达到20°、15 mm搜索边界，继续保留为搜索受限诊断。
## 独立结果审计

### 原生前景采样失败与独立补充协议

主队列C03在Pose初始化构造监督器时失败：A3C左房原生第82帧的自动标签有32个前景像素（约21.761893 mm²，置信度0.0625），stride=8粗采样未命中。其合法初始化及失败记录保留，患者仍在17例分母中；这是固定采样分辨率的算法限制，不能据此宣布患者数据本身不可处理。C04继续执行原协议。

`independent_audit/native_stride_coverage.json` 对全部17例可用标签做输入覆盖审计：stride=8仅上述一个Cell漏前景，stride=4/2/1均无漏采。没有使用TTO预测、Dice或隐藏Mesh/Pose GT筛选患者。实际checkpoint模型与C03数据的监督器构造检查见 `coverage_rescue_constructor_check.json`：原stride=8重现失败，stride=4覆盖8起点共1344个Cell，未执行模型前向或参数更新。

独立配置 `configs/real_tto_joint_pose_stride4_rescue_v1.yaml` 将Pose粗层stride从8改为4，仍为8起点、60步Pose及20次Joint，主损失stride=4、原Dice/BCE/Boundary、模型和全部几何/Motion阈值保持原值。其队列由输入覆盖规则选出C03，输出为独立版本目录。`coverage_rescue_predeclaration.json` 保存改动及SHA；补充结果不替代原协议失败。

`scripts/run_real_tto_coverage_rescue.py` 已排队等待原协调进程结束且全部七臂处理完成，再通过正式 `tto.experiments` 执行该补充协议自己的16例隐藏GT门控、C01/C02 Smoke/恢复和C03七臂匹配实验。等待器不修改或终止原实验。当前仅输入/监督器工程检查通过，补充重建尚未执行；进程与状态分别见 `coverage_rescue_launch.json`、`coverage_rescue_queue.json`。这部分完成前，C03修复仍是待验收工作。

`scripts/audit_real_tto_joint.py` 在优化包之外核对结果，不修改运行中的 TTO 源码和 checkpoint：

```bash
python scripts/audit_real_tto_joint.py --output outputs/real_tto_joint_v1_20261016 --verify-legacy --verify-renderer --render-overlays
```

审计会验证来源/输入身份、导出完成标记、实际执行预算、参数回滚、冻结 Motion 场、latent/Pose 信任域、NPZ/state 一致性、原生观察唯一性、CSV 与汇总统计一致性，并可从最终保存 Mesh 独立重渲染 Native Dice。它也从实际 Joint 使用的原生 Cell 计算初始化评分，逐一比较获准候选，核对 best 的覆盖率/尾部排序、Shape/Pose、当前保留状态和各实验臂的参数更新组；ED-only 按其实际 Joint 子集核对。另检查每例必需图像可解码、原生预览尺寸正确且动画包含 24 帧；这些文件检查不代替几何证书或真实三维精度评价。尚未完成的病例明确保留 PENDING/IN_PROGRESS。

`independent_audit/candidate_selection_verification.json` 记录新增排序检查在16例正式合成端点上的实际执行，并对内存副本中错误的 best 分数、Shape、Pose、当前状态和参数组做负对照。此项不重新渲染候选，不代表尚未完成的真实端点或其他消融臂已通过；独立 Native 重渲染和旧检查点哈希的早期证据保留各自时间戳。

独立审计还为每个已提交端点生成 `joint_parameter_trajectory.csv/.png/.pdf/.json`，位于对应的 `independent_audit/<arm>/<patient>/` 或 `independent_audit/synthetic/<patient>/`。曲线区分提议与回滚后保留的状态，显示相对 Joint 起点的 Shape latent 标准化位移、三视图 mm 平移与角度增量，并标出最终选中的更新。Native Dice 曲线表示提议候选，不能当作保留状态或最终端点的 Dice。`parameter_trajectory_artifact_check.json` 核对正式合成16例均包含初始化加20次更新的21行 CSV、可解码 PNG、PDF 及对应状态/源码身份。

`scripts/report_real_tto_joint_progress.py` 从实际提交端点、逐Cell CSV和保存的失败记录更新根中文进展报告，保留全部17例分母与缺失值，并核对CSV可用监督均值与正式汇总一致。它不会更新优化参数，也不作目标完成声明：

```bash
python scripts/report_real_tto_joint_progress.py --config configs/real_tto_joint_v1.yaml --output outputs/real_tto_joint_v1_20261016
```

报告逐例显示实际获准/尝试的Joint更新次数。审核器的 `optimization_budget` 还保存拒绝原因、各参数组的尝试/获准次数和最终学习率退避因子；次数不等于每次都有非零数值变化，也不等于最终选中端点的改善。

`scripts/analyze_real_tto_joint_pose.py` 检查已保存的8起点候选、预声明排序及相对初始Pose的搜索范围，不重选候选或修改参数：

```bash
python scripts/analyze_real_tto_joint_pose.py --config configs/real_tto_joint_v1.yaml --output outputs/real_tto_joint_v1_20261016
```

其JSON/中文报告为 `independent_audit/pose_initialization_diagnostics.json` 和 `POSE_INITIALIZATION_DIAGNOSTICS.md`。截至C07完成时检查的6例Pose初始化中，5例至少一个视图达到平移限值的99%，3例接近旋转限值，4例的尾部排序与最高Mean候选存在权衡；6例全部8个保存候选均无0.98达标Cell。接近边界不证明真实Pose在范围之外；保存候选的事后比较不证明换一个起点后的Joint效果，也不是独立候选重渲染。

C06已完成20次Joint提议，10次获准、10次因RA周期衔接误差超过固定Motion阈值而拒绝；最终最优合法状态来自第19步。全部152个原生Cell在每次更新中参与监督，Shape/平移/旋转梯度均为正。独立最终Native重渲染与保存结果一致：Mean=0.677243、Minimum=0.455666，≥0.98为0/152。证据见 `clinical_C06_independent.json` 和 `joint/C06/rollback_final_checkpoint.json`。末态Adam两组计数均为10、退避因子为0.001；中间Adam矩未逐次归档，不能把末态计数称为每次完整矩回滚的独立重建证据。

C06的10次早期运动拒绝没有Native或几何评价，相关值保持缺失。曲线为实际已有的Dice加点标记，使第12、15、19步等被缺失值隔开的孤立评价仍可见；不会跨缺失值连线或补零。原图和对应源码保存在 `joint/C06/plot_history/` 与 `audit_code/`，绘图修正不重跑优化或改写原端点审核的源码身份。

C07已完成20次Joint提议且全部获准，最优合法端点为第20步。`independent_audit/clinical_C07_independent.json` 独立重渲染152个原生Cell并核对保存状态、候选排序、原始Motion场、20步联合梯度和末态Adam计数；Mean=0.735541、Minimum=0.465263，≥0.98为0/152。其原生时刻插值支持全部24个模型槽，但这不等于逐槽独立真值或逐槽梯度认证；真实高可信Cell仍为0。保存的生产几何证书通过，独立审核没有重新运行几何内核。主实验随后进入C08，完整17例和匹配消融仍待完成。

C08同样完成并获准20次Joint提议，最终选择第20步。`independent_audit/clinical_C08_independent.json` 的128个原生Cell独立重渲染与保存结果一致：Mean=0.669519、Minimum=0.377469，≥0.98为0/128；20步联合梯度、冻结场、末态Adam计数和候选选择检查通过，插值支持全部24槽。完整空间场、真实3D精度和中间Adam矩的验证范围仍按各审计文件限定。主实验已进入C09。

C09完成20次提议，11次获准、9次因LV周期衔接误差超过固定0.224479924 mm阈值而拒绝；最终选中第11次提议，而末态保留参数来自第17次提议（该次获准）。`independent_audit/clinical_C09_independent.json` 核对136个原生Cell，Mean=0.539810、Minimum=0.191939，≥0.98为0/136。9次拒绝的Shape/Pose回滚均通过，末态Adam两组计数为11，学习率因子0.001953125。拒绝候选的Native和几何评价未计算，曲线保留缺失值；第11、14、17步的孤立评价点仍可见。几何/Motion结论限于保存的生产证书及相对先验保护，独立审核未重跑几何内核或重建中间Adam矩。

C10完成20次提议且全部获准，按预声明排序最终选择第9次提议。`independent_audit/clinical_C10_independent.json` 对168个原生Cell的独立重渲染与保存结果一致：Mean=0.644050、Minimum=0.397633，≥0.98为0/168。20步Shape/Pose梯度、冻结场、实际监督覆盖和末态Adam计数检查通过，插值支持全部24槽。所有提议获准不等于最后一次提议是最优，也不证明真实三维精度改善；仍需完整队列及匹配消融的实际结果。

C11完成20次Joint提议且全部获准，预声明尾部排序最终选择第0步（Joint起点），参考Shape及Pose端点逐位保持原起点。`independent_audit/clinical_C11_independent.json` 的232个原生Cell独立重渲染通过：Mean=0.737012、Minimum=0.516719，≥0.98为0/232；全部20步同时使用232个观察，Shape/Pose梯度为正，冻结场及末态Adam计数检查通过。原生时间插值支持24槽。这表明合法更新未带来被最终排序接受的端点收益，不能把20次获准更新解释为Joint改善。主实验继续C13，匹配消融尚未执行。

辅助Pose角度报告已修正float32近似旋转矩阵经acos(trace)产生的相同矩阵非零角度问题。报告副本使用float64最近SO(3)旋转及atan2，相同保存矩阵严格为0；不修改原Pose、优化器、候选排序或任何几何/Motion阈值。数值检查及此前26个已提交端点的修正字段见 `independent_audit/pose_angle_reporting_correction.json`，该补充记录仅替代早期审计中的Pose角度字段，保留原Native重渲染审计与源码身份。

C13完成20次Joint提议且全部获准，最终选择第16次提议。`independent_audit/clinical_C13_independent.json` 的112个原生Cell独立重渲染通过：Mean从Pose初始化后的0.630064升至0.645185，Minimum从0.444844升至0.455553；≥0.98仍为0/112。全部20步同时使用112个观察，Shape/Pose梯度、冻结场、候选选择及末态Adam计数检查通过。原生CSV独立核对插值支持23/24槽，第23槽无直接二维监督，仍保留24帧重建与安全审核，不能将全周期描述为逐槽独立标注。支持范围证据见 `independent_audit/joint/C13/native_phase_support_check.json`。主实验随后进入C14，完整队列、匹配消融、独立补救和最终逐条审核仍待完成。

C14完成20次Joint提议且全部获准，最终选择第20次提议。`independent_audit/clinical_C14_independent.json` 的152个原生Cell独立重渲染通过：Mean从Pose初始化后的0.617630升至0.635622，Minimum从0.247460升至0.302792；≥0.98仍为0/152。全部20步使用相同152个观察，Shape/Pose梯度、冻结场、候选选择及末态Adam计数检查通过。原生CSV与保存历史均支持全部24槽，支持范围不等于逐槽独立标注或梯度幅值证明。证据见 `independent_audit/joint/C14/native_phase_support_check.json`。主实验随后进入C15；这些二维改善不证明真实3D/Pose精度，完整队列、匹配消融、补救和最终逐条审核仍待完成。

C15完成20次Joint提议且全部获准，最终尾部排序选择第0步，即Joint起点。`independent_audit/clinical_C15_independent.json` 的200个原生Cell独立重渲染通过：最终Mean=0.676040、Minimum=0.418019，≥0.98为0/200，与Pose初始化端点一致。`independent_audit/joint/C15/baseline_endpoint_identity.json` 直接比较正式Pose初始化与导出状态，确认Shape latent、Pose旋转/中心及完整4D Mesh原始字节均一致；不能将20次获准提议解释为端点改善。全部20步的200个观察、联合梯度、冻结场、末态Adam计数和候选选择检查通过，原生CSV支持全部24槽。主实验随后进入C16，匹配消融、独立补救和最终完整逐条审核尚未完成。

C15保存候选的事后比较见 `independent_audit/joint/C15/candidate_tail_tradeoff.json`：第20步Mean=0.691243，高于起点0.676040，但Minimum=0.409188，低于起点0.418019。该步Shape/Pose提议确实发生变化；按既定覆盖率/尾部排序仍选择第0步。这是保存评分与参数历史的比较，没有独立重渲染候选或证明真实3D精度，不能把候选Mean改善替代最终选择的端点收益。

C16完成20次Joint提议且全部获准，最终尾部排序选择第0步，即Joint起点。`independent_audit/clinical_C16_independent.json` 的160个原生Cell独立重渲染通过：最终Mean=0.699026、Minimum=0.316360，≥0.98为0/160，与Pose初始化端点一致。正式Pose初始化与导出状态中的Shape latent、Pose旋转/中心及完整4D Mesh原始字节均一致，证据见 `independent_audit/joint/C16/baseline_endpoint_identity.json`。全部20步的监督覆盖、联合梯度、冻结场、末态Adam和候选选择检查通过；原生CSV支持全部24槽，支持范围不等于独立标注或逐槽梯度幅值验证。主实验随后进入C17，完整队列、匹配消融、补救及最终逐条审核尚未完成。

C17完成20次Joint提议，16次获准，第12、16、17、19次因Minimum低于固定下限0.339803而拒绝；Pose初始化Minimum=0.359803，允许下降0.02的规则未改动。最终选择第6次提议，160个原生Cell独立重渲染通过：Mean从0.668853升至0.675231，Minimum从0.359803升至0.372293，≥0.98仍为0/160。末态Adam两组计数均为16，独立步长因子0.0625；20次联合梯度、监督覆盖、冻结场、参数回滚与候选选择检查通过，原生CSV支持24槽。见 `independent_audit/clinical_C17_independent.json`。

C17新增实际完整优化器回滚证据：归档相邻检查点对15→16、16→17、18→19，逐一核对Shape/Pose原始字节、完整Adam状态（矩、计数和参数组）、最佳候选及冻结场恢复。三对全部通过，快照SHA已复核；对真实快照内存副本篡改一阶矩、二阶矩、计数和学习率的负对照均被检出。证据分别为 `independent_audit/joint/C17/full_optimizer_rollback_observation.json` 和 `full_optimizer_rollback_negative_controls.json`。第12次拒绝前的Adam矩未留档，不能将这三对证据扩大为所有临床拒绝的逐位验证；全范围参数回滚及源代码单元/恢复测试保留原作用范围。Native尾部拒绝已有实际Native指标，曲线保留全部21点，参数回滚平台与最终选择第6步已核对，见 `parameter_plot_rejected_native_check.json`。这些检查未重跑几何内核或重渲染中间候选。主实验继续C20，匹配消融、独立补救及最终完整逐条审核仍待完成。

`scripts/audit_real_tto_geometry_identity.py` 补充核对37个协议源码清单之外、由原始源包提供的几何模块身份：

```bash
python scripts/audit_real_tto_geometry_identity.py --output outputs/real_tto_joint_v1_20261016
```

截至C08完成，16个合成及7个真实提交端点中的全部保存身份记录一致，4个原始几何Python模块和固定 `source_support.json` 与记录SHA匹配。结果见 `independent_audit/geometry_backend_source_identity.json`，来源副本位于 `geometry_source_snapshot/`，仅用于归档，不作为新的内核实现。该检查不重新计算几何、不核对未列出的传递模块或二进制库，也不将尚无端点的实验臂视为已完成。

C05的实际80个原生Cell经时间插值支持19个模型槽，另5槽没有直接切片标签，模型仍输出和审核全部24帧；不能把完整周期优化描述成每个模型槽都有独立标注。具体支持索引及20步联合梯度见 `independent_audit/clinical_C05_independent.json`。C04第20步因Motion越界在Native评价前拒绝，其轨迹CSV中的Dice为空、图中曲线止于第19步，未填零；可视化与缺失值核查见 `independent_audit/joint/C04/parameter_plot_missing_metric_check.json`。

审计输出补充最终未观测区域漂移、Pose/跨视图变化、失败 Cell 的可观测原因及实际 Joint 阶段的监督覆盖率。`optimization_observations` 在原评价文件中表示可用观察集合；ED-only 的实际 Joint 使用集合以 `sampling.joint_loss_coverage` 为准。G 组共享完整周期 Encoder 和共同 Pose 初始化，仅 Joint 损失与候选选择是 ED-only，不能解释为整条流程只见过 ED。

`--render-overlays` 输出按真实原生帧时刻插值的切片叠加及 `index.json` 时间凭据。原流水线的固定模型槽预览使用最近原生标签，不能作为精确时刻一致性的证据。旧 C01 端点仅在 `independent_audit/legacy_overlay_validation/` 中用于测试此导出器，未被计入新 Joint 队列。

`scripts/profile_real_tto_joint_original.py` 在自动选择的 GPU 上调用正式 Original-Motion Joint 前向/反向，在第一次 Adam 更新之前停止。当前实测最大原生渲染负载的 C11/C15、432 Cell、Batch=2，allocator 峰值 2242 MiB，context 约 490 MiB；未执行真实患者参数更新。结果在 `independent_audit/original_motion_memory_preflight.json`。

`scripts/check_real_tto_joint_batch.py` 补充实际 checkpoint 与 C01/C02 原生数据的 CPU 只读批量一致性检查。`independent_audit/actual_batch_parity.json` 记录 304 个 Cell、36 项比较全部通过：批量与单独编码/解码的完整4D顶点最大差约 0.000050 mm；在相同冻结初始化下，切片损失最大差约 5.96e-8，Shape/Pose 梯度最大差约 2.31e-8，两个患者的交叉梯度严格为零。该检查执行 0 次优化器更新，不包含几何审核，也不代替真实端到端 TTO、优化后 Batch/单例比较或端点恢复试验。

```bash
python scripts/check_real_tto_joint_batch.py --patients C01 C02
```

阶段性证据：正式 SYN_000 完成 20 次 Joint，192 个 Cell 的独立重渲染与已存指标逐项一致，14 次参数回滚通过。≥0.98 Cell 在 Pose 初始化后为 82/192，Joint 后为 85/192；Shape RMSE 0.362945→0.365187 mm，未改善。原始 Pose 旋转误差约 10–12°，最终约 0.386–1.819°。这是一例 TRAIN 未知 Pose 功能验证，不能作为未见患者泛化或真实三维精度的证明。

真实相位置信度当前为共同估计因子。它在归一化加权均值中会抵消，能标记时相不确定性，但不能声称它降低了整体观察相对于三维先验的权重。Motion-Lock 相对初始预测通过，也不证明该预测的生理运动本来正确。

`scripts/analyze_real_tto_joint_hidden.py` 只处理已经提交 `export.pth` 和 `completed.json` 的合成病例，分别输出初始化、Pose 搜索后、Joint 后的 GT 误差与 Native Dice，并保存隐藏 GT/初始预测/Joint 的容积曲线 PNG/PDF。

```bash
python scripts/analyze_real_tto_joint_hidden.py --output outputs/real_tto_joint_v1_20261016
```

这里的 GT Pose 仅用于端点后的独立诊断，不进入优化器、候选选择或预算调整。SYN_000 的 GT Mesh+GT Pose 在全部 192 个原生 Cell 上 Dice=1，验证源标签与原生渲染一致；初始网络 Mesh 在 GT Pose 下有 144/192 Cell 达到 0.98。实际估计 Pose 的结果较低，不能将全部差距归因于 Shape 先验容量。

独立分析器同时生成 `independent_audit/SYNTHETIC_RECOVERY_REPORT.md`、`hidden_gt_stage_attribution.json` 和 `synthetic_gate_recomputed.json`，按初始、Pose 初始化、Joint 三阶段汇总已提交病例，并逐项复核已生成的正式门控；门控尚未生成时明确保留待完成状态。GT 误差按病例汇总，Native Dice 按实际 Cell 数加权，不以某个已完成病例代替全队列。

正式 SYN_000–SYN_015 共16例现已完成，逐病例完整表格与三阶段比较见 `independent_audit/SYNTHETIC_RECOVERY_REPORT.md`。共320次 Joint 候选中108次获准、212次回滚；SYN_006、SYN_008、SYN_012 最终选择的最优合法端点与 Pose 初始化端点逐位相同。获准更新次数不能代替最终端点改善的证据。

Shape RMSE 相对 Pose 初始化在11例上升、3例不变、2例下降，其中一例下降仅约0.000005 mm。完整合成闭环支持 Pose 恢复和预声明范围内的 Shape 保持，未证明稳定的 Shape 精度收益。SYN_002 最终三视图旋转误差为3.927°、8.609°、3.015°，最低 Native Dice 为0.890036；不能用整体中位数掩盖该病例的不足。

全部16例保存端点的参考 Mesh、24个保存时相、跨腔关系、连续线性轨迹及相对初始先验的 Motion 检查通过；完整空间 FFD 场证书仍为未验证。真实队列、匹配消融及最终研究结论继续保留待验收状态，不由合成门控通过推定完成。

`scripts/summarize_real_tto_synthetic_surfaces.py` 以当前正式协议中的合成病例数为准，核对每个已提交端点/隐藏真值的SHA与独立三阶段RMSE，汇总正式评价器保存的参考ASSD、HD95及完整周期误差。它不加载模型或向优化器提供GT，也不重新计算表面距离。ASSD/HD95的具体定义为双向顶点采样到对侧三角形表面的距离；不能称为精确面积积分或对应顶点RMSE。

```bash
python scripts/summarize_real_tto_synthetic_surfaces.py --output outputs/real_tto_joint_v1_20261016
```

实际16例汇总见 `independent_audit/SYNTHETIC_SURFACE_REPORT.md`、`synthetic_surface_summary.json`、`synthetic_surface_cases.csv`。参考ASSD病例中位数0.183357→0.187328 mm，12例上升、3例不变、1例下降；HD95中位数0.453262→0.475972 mm，10例上升、3例不变、3例下降。完整周期顶点RMSE中位数1.120464→1.121409 mm；相对运动RMSE中位数1.076531→1.075754 mm。各指标的病例配对变化与中位数之差分别记录，二者不能互相替代。该合成TRAIN闭环未显示稳定的Shape精度收益，不能据此宣称真实三维精度改善。
