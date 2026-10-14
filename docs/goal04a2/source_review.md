# Goal 04-A2 源码审核

审核起点：2026-10-13，服务器 `main` / `80b4b4505c19bfb08ba82895bf421f52fe53fc6e`，本次续接审核发现已有未提交 v2 实现与部分诊断产物，不能将其归为干净工作树。先阅读当前 v1 报告、README、配置、训练入口、测试及其 CNN/Lifting/U-Net/FFD/Loss/Renderer 实现。原 v1 是已通过 TRAIN4 工程收敛的模型，没有局部运动、独立视图或高分辨率 SDF 条件路径；这不等于解决泛化问题。

以下 Commit 为本轮实际 shallow checkout 的 HEAD；只取得源码和许可证，没有下载数据或模型。GitHub API 限流后改用公开 Git checkout。第三方源码留在 `/tmp/heartssm-goal04a2-sources`，不导入第三方模型、权重或未经确认许可的实现。

| 仓库 / 固定 Commit | 论文对应 | 实际阅读文件 / 函数 | 许可证 | 复用决策 |
|---|---|---|---|---|
| [Ma HeartSSM](https://github.com/m-qiang/HeartSSM/tree/f3b2dd3e41229508b3be44e92a40fa4a8df71e35) `f3b2dd3e41229508b3be44e92a40fa4a8df71e35` | README: Learning a dynamic four-chamber shape model…95,695 UK Biobank participants | `net/ffdnet.py:FFDNet`, `net/module.py:ConvBlock`, `utils/ffd.py:bspline_basis/bspline_ffd`, `train_ffd.py` 参考及递推应用 | Apache-2.0，读取根 LICENSE | 直接调用本仓库已有相同 ConvBlock/U-Net 层；保留 Joint 3*T 输出思想。使用当前物理 FFD，拒绝 Batch=1 索引、pad、归一化索引及50帧合同。 |
| [DeepMesh](https://github.com/ImperialCollegeLondon/DeepMesh/tree/4753673e56c0f35b5c920c2f52a534e1a60ff909) `4753673e56c0f35b5c920c2f52a534e1a60ff909` | README: DeepMesh, IEEE TMI 2024, DOI 10.1109/TMI.2023.3340118 | `network_motion.py:MotionMesh_25d.forward/generate_grid/transform`, `utils.py:projection` | MIT，读取根 LICENSE | 借鉴参考/当前多尺度特征对及视图来源保留；不复制 MRI 对齐后的通道→体数据重解释、固定 SAX/2CH/4CH、CUDA硬编码、体素坐标和顶点投影损失。 |
| [TetHeart](https://github.com/Scalsol/TetHeart/tree/94990284931966116274915ccbf8598dd903a5b6) `94990284931966116274915ccbf8598dd903a5b6` | README: End-to-End 4D Heart Mesh Recovery Across Full-Stack and Sparse Cardiac MRI, TMLR；arXiv:2509.12090 | `tetheart/models/segmentors/nnunet_tet_mix_attn_dyn.py:construct_3d_from_2d/forward`, `backbones/dynunet_geo_mix_attn.py`, `segmentors/geometry_mixins.py:sample_multi_feats`, `decoders/lstm.py:LSTMWarper` | 在完整 git tree 未找到 LICENSE/COPYING/NOTICE；许可未确认 | **不复制代码**。只参考稀疏切片支持感知和局部空间采样思路。不采用单病例索引、各类别独立运动 Decoder、LSTM 逐步坐标仿射、marching tetrahedra 可变拓扑及 MRI 栅格。 |
| [GHDHeart](https://github.com/Luo-Yihao/GHDHeart/tree/f150dbd6fedbdedce656be2012f784870b41c2ce) `f150dbd6fedbdedce656be2012f784870b41c2ce` | README: Explicit Differentiable Slicing and Global Deformation for Cardiac Mesh Reconstruction；arXiv:2409.02070，另列 DVS arXiv:2407.11272 | `GHD/core.py:ghf_to_vertf/vertf_to_ghf`, `ops/mesh_geometry.py:DifferentiableRasterizer/ApproxDVS`, `ops/torch_warping.py:feature_align`, `losses/meshlossclass.py` | MIT，读取根 LICENSE | 参考将可导切片梯度与权威评价分开的思想；继续使用本仓库精确 mesh-plane 切片。不引入全局谐波顶点位移、稠密体素 soft occupancy、额外 PyTorch3D 依赖或另一套像素归一化。 |
| [Kong HeartFFDNet](https://github.com/fkong7/HeartFFDNet/tree/a0c5a8472aa0d8c97f03f8dbf10e0c0c63897460) `a0c5a8472aa0d8c97f03f8dbf10e0c0c63897460` | README: Whole Heart Mesh Generation For Image-Based Computational Simulations By Learning Free-From Deformations, MICCAI 2021 | `src/custom_layers.py:Projection`, `src/model.py:_graph_decoder_bspline`, `template/make_control_grid.py:construct_bspline_volume/construct_bspline_matrix/bspline`, `src/loss.py` | Apache-2.0，读取根 LICENSE 和文件头 | 参考多尺度空间查询与控制场逐层优化；直接复用已验证的本地物理 FFD。原代码 TensorFlow、3D CT/MRI、channels-last、归一化网格和 clamped-knot 矩阵不等于当前 Cardinal lattice 当前顶点递推；不复制静默索引截断和归一化 displacement。 |

## 坐标、形状与变形差异

- Ma：输入 `[B,T+1,96,96,128]` 稠密 CMR 标签；特征网格24/12/6；输出 padded 网格和3*T通道。v2 输入 `[B,24,3,4,224,224]` Mask/SDF；已知 Probe 几何、物理毫米域和固定10631顶点；无50帧或五组件划分。
- DeepMesh：当前/参考 SAX、2CH、4CH 影像分别提取多尺度特征，再把融合通道重解释为体积；输出 normalized tanh dense vector field。v2 以真实 Pose 把各视图提升到物理节点，保留 view/time 支持；输出共享 mm B-spline 控制增量。
- TetHeart：2D sparse MRI 切片 `[S,C,H,W]` 通过邻近 slice/grid attention 提升为体特征，采样 tetrahedral coordinates 并按类别 recurrent warp。v2 不使用 MRI slice index 作为物理位置、不学习 tetrahedral SDF、不改变固定拓扑。
- GHD：谐波系数 `[B,K,3]` 乘 eigenbasis 得 `[B,N,3]` 自由顶点形变；可导 occupancy/rasterization 使用 normalized 3D samples。v2 仅用已有 mesh-plane 几何梯度和共享物理 FFD，不新增顶点参数化。
- Kong：volume feature sampling 在 normalized mesh/control coordinates，输出多层 graph-grid displacement，用预计算 spline matrices作用模板。v2 每个递推步在当前 mm 顶点重新求 Cardinal B-spline basis，不能替换成静态矩阵绝对 warp。

## 本地直接复用 / 扩展 / 隔离

直接复用：`GeometrySpatialEncoder`（含原 spatial_features）、`project_physical_points`、`feature_sampling_grid`、v1 的共享 U-Net 与几何缓冲区、`control_geometry`、`bspline_ffd_physical`、`NeuralFFDLoss`、`MeshSliceSupervisor`、原生 `exact_metrics`。

新增可审查模块：逐视图 SDF/CNN Lifting 适配层、支持门控与冲突条件、显式 masked Eulerian temporal conditioner、一次输出24场的局部条件 Joint generator，以及独立审核脚本/Checkpoint。只在最终组合 raw controls 之后统一限幅与 Lipschitz 约束。

旧 `NeuralFFDNet`、`DynamicSpatialConditioner`、Motion32/FiLM、v1 文件/配置/Checkpoint 保持原样。InstanceNorm 暂不替换；需要实际信息损失证据才能另设实验。

## GPU 与现有结果核对

本次执行环境可直接访问8张 NVIDIA GPU；早期默认沙箱不可访问 GPU 的记录仅属于历史执行条件。现有 `4dheart` / `4dheart_whpilot` 为 PyTorch 2.1.2+cu118；`4dheart` NumPy1.26.4/OpenCV4.10.0/trimesh4.11.5，符合历史 Renderer 环境。使用宿主机现有环境测试，不改变环境依赖，不停止共享 GPU 的其他进程。需记录资源竞争和实际 CUDA 非确定性，不能把默认沙箱 CPU 环境的结果当作 GPU 结果。

O 初步确认 `outputs/neural_ffd_1k_long_20261010_225918/best_joint.pth`：epoch162、step129600；best_dice为epoch160，last是epoch162中途step130200，不能混同。G1 冻结 `outputs/geometry_goal04_final/latest.pth`：随机初始化TRAIN4、step144、known_pose、binary_native_v1，仅作冻结诊断对象。公平从零对照另行训练，不沿用这份权重。


## 独立腔间几何工具

使用宿主机现有 VTK 9.6.2 的 native `vtkCollisionDetectionFilter`（OBB-tree surface contacts）和 `vtkSelectEnclosedPoints` / `vtkImplicitPolyDataDistance`（包含与边界距离）；API已与[官方collision文档](https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html)、[enclosed-points文档](https://vtk.org/doc/nightly/html/classvtkSelectEnclosedPoints.html)及安装对象方法核对。再用独立 Trimesh ray containment交叉验证。调用库API，不复制第三方实现；原native Renderer不变。

## 本次续接审核与物理安全决策

重新读取指定五仓库的固定 checkout，逐一核对实际 HEAD（2026-10-13）。HeartSSM 另用 git ls-remote 核对公开 HEAD。未确认许可的 TetHeart 不复制实现。冻结文件与 main HEAD 一致，未提交 v2 工作继续审核而非覆盖。

固定生成源 `synthetic_bundle_20261009/configs/geometry/source_support.json` 已定义房室接合 Face-ID 支持；其 SHA256 为 `faa136bc...f78e4`。原严格分离诊断发现430个包含顶点，均位于这份不可变支持区，区外为0。因此 v2 按源表示合同审核，仅许可 LV/LA 和 RV/RA 固定支持区内的封口接触，拒绝远端面接触、区外包含和整体包裹。此决定来自源码与数据表示，未按训练结果扩展支持、修改 GT 或改变原生 Renderer。源 CGAL6.0.1.post202410241521、IPC1.6.0 用于实际三角片与连续审核；VTK9.6.2 为独立包含/接触检查。原 v1 审核源码保持原样。

实际源码阅读与复用决策包括：DeepMesh 当前/参考特征拼接及 voxel-z 距离权重；TetHeart normalized grid_sample、MRI affine[0] 与逐类别 recurrent warp；GHD eigenbasis→顶点和 normalized feature_align；Kong normalized Projection 的静默 clip 和 clamped-knot control matrix。以上均与本项目物理 Pose、24 相位和 Cardinal FFD 合同逐项对照，无直接第三方代码移植。
