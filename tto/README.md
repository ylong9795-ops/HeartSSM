# TTO scaffold

本目录是清理后的最小 TTO 骨架。唯一入口为 `python -m tto.run`，
唯一配置协议为 `tto-scaffold-v1`（`configs/tto.yaml`）。
当前实现原生观测预检、冻结原模型的精确初始化、身份检查和新状态保存/恢复。
S1 已实现冻结 Mesh 的中心诊断及观测切面 Pose 候选搜索。S2 的
`chamber_candidates` 已实现逐腔室相对提示与原 Regional8 候选回拟合；
通用64维 Shape、Motion、Joint 优化尚未实现，调用这些阶段会明确失败。

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
│   ├── global_alignment.py     固定 canonical 坐标的静态切面候选搜索
│   ├── centres.py / config.py  二值中心/面积、鲁棒聚合与共享搜索策略
│   ├── chamber_alignment.py    可观测相对位移提示；禁止成为最终Mesh偏移
│   ├── regional_decoder.py     原Regional8解码路径；缓存冻结Global32
│   └── s2_config.py            S2共享策略和预算
├── optimization/
│   ├── regional_shape.py       S2 Regional8回拟合、原Decoder端点与几何准入
│   ├── shape.py                通用64D Shape搜索合同（待实现）
│   ├── motion.py               原 32D Motion 搜索合同（待实现）
│   └── joint.py                Shape/Motion/整体位姿联合合同（待实现）
├── evaluator.py                独立原生渲染与汇总底层
├── synthetic_oracle.py         隔离隐藏 GT、保存终点后的独立评价
├── visualization.py            新状态的原生叠图与 4D 网格预览
├── experiments.py              阶段能力注册表，无第二运行入口
└── validate.py                 新状态合同检查，无第二运行入口
```

当前可执行路径为初始化 → S1 观测切面 Pose 候选搜索。后续共享模型
训练应独立落实，不把旧 Shape/Motion 搜索接口视为患者微调执行计划。

用户后续明确授权 S2 的 Regional Shape latent 候选优化。该授权只适用于
本节的区域候选算法；网络权重继续冻结，不训练患者专属网络，也不启动
正式 Motion 或 Joint 训练。S2 与“所有latent永不调整”的旧约束不同，
以用户最新给定的 Global32/Regional8/Motion32 分工为准。

## 已确定的空间语义

2026-10-17 确认：**逐腔室位移只用于生成初始对齐候选**。
候选必须回到原始 Decoder 的 Shape64 / Motion32 表达，并接受四腔解剖审核。
最终 Mesh 不附加腔室静态位移，状态合同也没有这类参数。

历史状态的 `global_transform` 以列向量约定表示
`x_patient = R @ x_canonical + center_mm`，跨四腔和 24 帧共享。
`trajectory` 保存原始 Decoder 的 canonical mm 输出，
`patient_trajectory()` 才施加整体变换。
`observation_pose` 表示患者坐标中的三切面位姿，未经配准时为
`None`；初始化的单位整体变换不能被解释为已完成配准。
原生评价和叠图需要显式提供合法切面位姿。

S1 固定 NeuralFFDNet canonical 坐标，将 `global_transform` 设为严格单位
变换，只搜索三个视图相对于同一冻结 Mesh 的静态切面 Pose。已有患者
坐标 Pose 先通过全局逆刚体变换转换至 canonical；两组变量不同时自由
搜索。带 `pose_gauge=canonical_identity_global` 的状态会拒绝非单位全局
变换。S1 是用户授权的观测几何配准，不更新患者网络、latent 或控制量。

## 数据、模型与监督边界

- 保留真实 Mask/SDF 合同、24 个模型相位、各视图原生尺寸和物理像素间距。
  原生 Mask 不缩放；只有编码器输入经过原 MaskTransform。
- 默认 `data.input_mode: aligned24`：读取已选好的24帧，模型索引与原生
  索引一一对应，不读取 RR、不插值、不滚动相位，也不重新选帧。
  真实监督按同一索引取原生标签，二值监督不来自 GT Mesh。
  仅显式 `raw_rr` 模式保留旧的 RR 选帧和真实时间到相邻模型网格的映射。
- 标注来源、信任权重、时间误差和未参与优化的单元均保留。
  切片不存在时报告缺失；梯度 Loss 不虚构质心/平面距离捕获力。
  S1 通过显式独立几何候选作无梯度捕获，之后重新原生渲染评价。
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
默认数据根目录为 `outputs/patient_24frames_masks_reports_20261017`，
原 `real_data` 只能在明确选择 `data.input_mode: raw_rr` 后使用。
默认不加载原始影像；导出叠图前需要启用 `data.load_images`。
预检不会加载模型，初始化会核对 checkpoint SHA256。
当前 CLI 的初始化只支持 CPU；原 `gpu.select_gpu()` 的实测峰值准入工具
保留独立可测，GPU 工作负载探测尚未接入新状态机。

状态使用 `tto-state-v1`，身份包含患者、输入、checkpoint、配置和 TTO
源码摘要。恢复会拒绝身份不匹配的状态；旧协议状态不会隐式迁移。
已有初始化需显式 `--resume`，不覆盖旧结果。
初始化成功只证明工程合同，不代表重建、配准、几何或临床验证成功。

## S0 aligned24 输入合同

`aligned24` 的批次固定为 `masks/sdf [B,24,3,4,224,224]`、
`view_present [B,24,3]`、`annotated [B,24,3,4]`、`phases [B,24]`。
视图按名字映射至 A2C/A3C/A4C；所有掩膜、来源与 annotated 通道一起按
名字映射至 LV/LA/RV/RA，拒绝重名和缺名。SDF 是负内正外、截断15mm的
毫米距离；保留原 Encoder 内部除以15的行为，数据侧不再归一化。

现有24帧包由 `aligned24.py` 直接读取并校验包内 SHA256、帧号、时间、
通道、来源及物理标定。只有编码器输入执行与合成数据相同的空间
MaskTransform：224网格、固定各视图FOV、物理中心 W/2,H/2，不允许裁掉
前景。原生尺寸的掩膜与 spacing 保留在 `PatientObservations.views`，
`metadata.model_to_native [24,3,3,3]` 为齐次 x/y 像素索引变换，
`metadata.model_spacing_yx_mm [24,3,2]` 保留模型网格的毫米间距。

自定义 `tto_input.json` 使用 `version: real-tto-input-v1` 与
`input_mode: aligned24`，包含 patient_id、phases、views 与 phase_alignment。
每个 view 声明 name、channels、masks_npz、times_s、spacing_yx_mm、
annotated、sources，可选 frame_ids 和逐帧 view_present。annotated/sources
与 masks 一样使用该 view 声明的 channels 顺序。phases 必须为 i/24，
不含重复的闭合端点。times_s 只作为来源记录，绝不触发重采样。

若已有模型网格的 Mask/SDF，可通过 `encoder_npz` 提供 masks、sdf、
annotated、view_present、phases、views、channels、sdf_units='mm'、
model_to_native 与 pixel_spacing_yx_mm。各数组不含批次维；读取后只按
名字重排，核验与保留原生网格上的预训练空间预处理一致，再直接使用。
单位、符号、中心、间距或数值不一致均报错，不自动修补输入。

`phase_alignment` 必须声明 status、phase_grid_status（VERIFIED 或
UNVERIFIED）、evidence、phase_grid_evidence、synthetic_reference。
synthetic_reference 的定义为
`synthetic source frame 0; not independently selected clinical ED`。
人工 ED 或24帧数量本身不能证明与该参考相位一致；VERIFIED 应有明确
外部证据。S0 不会推断 ED=phase0，不改已有帧序。

现有包没有提供合成参考时相等价证据，C11/C15 还有非均匀实际采样间隔。
读取该包时 i/24 **仅作为原模型可运行的诊断索引**，phase0/grid 均记录
UNVERIFIED；不得将该诊断索引解释为已证明的均匀生理相位。
`diagnostics.s0_input_audit.ready_for_shape_tto` 因而为 false，
`s0_status=ENGINEERING_PASS_PHASE_ALIGNMENT_UNVERIFIED`。
只有参考时相及网格语义都经证据审核，才记录 S0 PASS。

初始化还会核对该 checkpoint 指定的合成数据配置与真实输入的尺寸、
FOV、SDF处理和通道合同。旧 checkpoint 未内嵌数据配置的历史内容哈希，
报告明确记录本次读取文件的 SHA256，不将它冒充历史来源证明。
`initialized.pth` 保存 Shape64、Motion32、全部24帧 canonical mm Mesh、
参考 Mesh 及全部训练条件；`initialization.json` 保存同一模型原 forward
与 Encoder→原 Decoder 的逐项差异，`preflight.json` 保存完整输入审核。

后续研究目标是适配新患者的共享模型及大规模真实队列训练。该输入合同
与 `collate_patients` 不含患者专属参数，可用于多患者批次；逐例初始化
文件只是统一先验的基线存档，不是17个患者模型或训练结果。正式训练应
使用固定的患者级 train/validation/test 划分与共享网络参数。现有合成
训练器依赖 `vertices_gt/pose_gt`，不能直接当作真实数据训练器；真实
监督、可用位姿和相位语义需在真实训练协议中落实。

用户进一步明确：**不进行单患者微调**。后续训练只更新跨患者共享的
网络参数；新患者走前向推理，不单独优化其网络权重、Shape/Motion latent
或患者专属控制量。当前没有额外的大规模真实队列目录，这一要求是模型
训练和推理方式的约束，并非本轮启动大规模真实训练。本文历史 TTO 搜索
阶段属于保留的未实现接口，不作为后续实施方向或执行授权。

## S1：中心诊断与静态切面候选

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tto.run --stage global_alignment --workers 4 \
  --initialization-root outputs/s0_aligned24_20261017/cohort \
  --output-dir outputs/s1_native_alignment_20261017/cohort
```

`--initialization-root` 显式导入原 S0 输出：核对患者、输入与 checkpoint
身份，重新运行原 Decoder 并要求完整 Mesh 逐位相等，再记录源状态 SHA256
和旧身份；新阶段绑定本次 config/source 身份，不伪装成旧 S0 的原地恢复。
不指定时使用本次输出目录的初始化。已有结果通过 `--resume` 严格核对
身份，不覆盖旧结果。

先用原生二值 Mask 分别记录真实/预测 centroid、预测减真实的毫米中心
偏差、面积比与 Dice；预测严格来自冻结的原 Decoder Mesh 经当前物理
切面渲染。空预测的 centroid/error 为 null，面积比为0；观察为空时
面积比分母不存在，记录 null。渲染异常独立保留并计为失败。

每视图每腔室优先使用 source_weight>=0.5 的可信标注，多时相用带来源/
相位置信度的 Huber 聚合，再对腔室等权 Huber 聚合，避免面积/时相数量
使 LV 支配其它腔室。若该腔室没有可信来源，显式标记为弱来源回退；
S0 的时相不确定性不会被配准抹去。正号修正
`center += R[:,:2] @ (predicted_centroid-observed_centroid)_mm`
已经过倾斜切面、各向异性间距和矩形图像的已知平移测试。

之后使用同一共享策略作无梯度粗到细搜索：默认深度±40mm/10mm步长，
倾角±25°，面内旋转±20°；两层细搜为8mm/8°及3mm/3°。另以 Mesh 沿
切面法向的分位深度和“Mesh 投影均值—观察质心”构造独立捕获种子，
不依赖已有预测非空。几何投影种子仅供捕获，不能冒充预测二值质心。
全部候选均由原 `native_exact` 重新评价，不改 spacing/FOV/Mask/帧序。

粗搜每腔取两个可信时相；最终较大候选束按完整24帧重新评价，使用
逐腔室等权、可信时相加权的 native Dice 排序，同时报告全部时相表现。
初始和中心修正候选始终参加最终比较。每视图保留3个空间上有差异的
候选；`observation_pose` 只是各视图排名第一候选的暂定组合，不表示
存在已证实的唯一三维真实 Pose，也未证明三视图独立最优组合最一致。

输出 `aligned.pth`、`alignment.json` 及每视图的 `search.json`、
`candidates.json`、baseline/centre_only/candidate_N 的逐 Cell CSV。
状态明确记录 `POSE_CANDIDATES_UNVERIFIED`。预算耗尽记录
`BUDGET_EXHAUSTED`，不伪报完整搜索；缺失视图保留 `NO_USABLE_OBSERVATIONS`。
不同患者可用 CPU 进程并行。该阶段不需要 GPU，也不进行训练。

## S2：相对提示 → 原Regional8候选 → 严格四腔审核

```bash
PYTHONPATH=/tmp/heartssm_inverse4d_host_geometry311 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tto.run --stage chamber_candidates --workers 4 \
  --alignment-root outputs/s1_native_alignment_20261017/cohort \
  --output-dir outputs/s2_regional_alignment_20261017/cohort
```

本机已存在的几何依赖目录包含 requirements.txt 固定版本 CGAL/ipctk，
不修改原 conda 环境。没有这些固定依赖时原 GeometryGuard 会返回
UNVERIFIED，S2 拒绝晋级，不能静默换用近似交叉检测。

S2 显式导入 S1 首候选 Pose 和完整24帧预测，核对患者、观测及checkpoint
身份并重跑原Decoder，要求源预测逐位相等。S1 的其它 Pose 假设保留在
诊断中；本轮不联合搜索其组合，也不将首候选当作真实三维标定。

先在固定切面上重新渲染全部24帧，逐Cell保存二值质心偏差、面积比、
Dice、ASSD/HD95。按腔室/视图对可信时相作鲁棒聚合，利用
`R_v[:,:2].T @ delta_c = -e_vc` 估计相对三维位移提示。Mesh 修正与 S1
切面修正符号相反，已经过已知投影测试。

求解联立分离共同平移 nuisance，并施加四腔位移和为零；不能先给RV/RA
补任意深度再减均值，因为减均值可能重新引入不可辨识深度。每腔室的
位移限制在多视图投影矩阵的SVD可观测子空间，单A4C的RV/RA提示法向
分量为零。canonical z 分量不等于切面法向深度，应看报告中的
single_view_normal_component_mm。提示上限10mm只约束初始化提示，
不是最终latent搜索的硬半径。

提示不是3D GT。临时整周期逐腔平移仅用于检查相对运动不变性和潜在
解剖破坏，既不写入最终状态，也不作为最终Mesh修正。原 Regional
线性头/FFD对提示作带正则回拟合，生成0、0.5、1倍初始化分支。匹配
误差完整保留；该正则回拟合误差不是原模型可达性下界。

实际checkpoint核验为 Shape64=Global32+四个Regional8，Motion32。
S2 固定 Global32、Motion32、网络权重和三视图共享 Pose，原 Regional8
经原 regional_heads/apply_regions 输出四腔表面。Global分支缓存只利用
其已冻结条件；基线和候选均与原 decode_shape 逐位比较，不修改Decoder
或构造新的自由顶点模型。区域参数另按原Decoder的均值位移Jacobian
去除共同平移自由度，最终参考Mesh的四腔平均位移为零（数值误差除外）。

首轮优化范围为参考phase0的全部已观察腔室，以原生真实标签的中心、
边界和soft Dice加软anatomy先验为目标；这不是完整周期训练。每个更新
都评价全部参考腔室，先对残差最大腔室的区域参数作6步优先更新，再
联合区域参数；共同平移投影会产生协调的其它腔室修正。总24步/分支。
每个端点重新运行完整原24帧Decoder，对所有已观察时相/腔室计算硬
Native指标，不用参考帧loss下降冒充完整周期改善。

软 anatomy 项包含原表面/latent锚定、边长、法向及固定原AV cap邻域
匹配，仅作为优化先验。最终是否允许接触、穿透或自交由未经修改的
GeometryGuard和原SourceAVCapPolicy决定，不由soft loss定义。三分支
端点均先审核参考几何；完整24帧可信腔室宏平均Dice须提高至少0.0001，
每个已观察腔室的可信Dice退化不超过0.005（共享工程准入容差，非临床
阈值）。最优合格端点再在90秒预算内审核完整24帧、固定AV规则及
23→0在内的连续顶点弦；FAIL、超时或UNVERIFIED均不接受。其余候选
保留为未完整核验假设，不能因参考几何通过就晋级。

`s2_state.pth` 只保存接受后的原Decoder结果；没有合格候选时保持S1的
Shape/Motion/Mesh和Pose，状态明确NO_ACCEPTED_CANDIDATE。各分支的
state.pth、optimizer.pth、history.json、cells.csv、report.json独立保留。
提示和实际三维相对位置只作诊断字段，不存在最终chamber_offsets参数。
带 regional-only-v1 的状态会拒绝 Global32、Motion32 或S1 Pose改变，
也拒绝把临时位移标为最终Mesh变换。

Motion32冻结不意味着Motion场固定：原FineMotionHead直接读取Shape64，
且运动FFD在更新后的顶点上求值。每个候选必须重跑原Decoder，报告
逐腔24帧相对位移RMSE、幅度/方向、体积曲线和闭合步变化，不以“运动
变化必须为零”作为准入条件。不能把静态Shape候选当作正式Motion适配。

2026-10-17 本轮执行/验证：17例、每例3个候选、共1224步Regional优化；
35项相关测试通过，51个原Decoder完整轨迹及9792个原生Cell冷复核一致。
12例通过预定可信二维/原完整几何准入，5例保持S1。C04、C08、C13分别
因连续或离散几何失败被拒绝，未发生超时。接受候选仍可有相对运动变化，
最大逐腔RMSE为1.02549mm；未接受候选最大1.39727mm，Motion32始终冻结。

**不能据此声称S2整体有效**：272个人工Cell Dice仅0.750298→0.750758，
全部3264个Cell Dice为0.740189→0.739975，中心/边界均值也略退化。
C09/RV面积比中位数4.235→4.066，中心误差均值12.95→13.24mm，问题
仍明显。参考帧loss下降、共享工程准入和真实四腔对齐成功必须区分。
完整失败、来源分层、运动变化及原生叠图见
`outputs/s2_regional_alignment_20261017/{final_report.rst,verification.json,diagnostics.png}`。

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
S1 只读取其中的 TRAIN16 初始 Pose，并固定校验文件 SHA256 与 checkpoint
身份；不读取 Shape 搜索范围、shape_mean/std 或旧运动阈值。
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

2026-10-17 S1 本轮验收：17/17例、51个视图均完成共享策略搜索，每视图
保留3个候选；独立 NativeEvaluator 冷重渲染全部153个候选的9792个Cell，
Dice与面积逐项完全一致。Shape/Motion latent、参考及完整24帧Mesh与S0
逐位相同，重新运行原Decoder也一致；global_transform严格为单位变换。
相关29项测试（S1新增6、原生11、scaffold8、aligned24四项）全部通过，
这不是对整个仓库的全套回归声明。

全部3264个标注Cell的平均native Dice：初始0.4262 → 仅中心修正0.5908
→ 深度/角度首候选0.7402；中心偏差中位数19.00 → 9.43 → 4.77mm。
最终最低Cell Dice仍只有0.0363，C09/RV面积比中位数仍约4.24。二维改善
不表示Shape/真实3D Pose正确；S0相位语义和三维绝对Pose仍未核实。
完整结果、来源审查及逐Cell失败保留在
`outputs/s1_native_alignment_20261017/{final_report.rst,verification.json,reference_review.json}`。
