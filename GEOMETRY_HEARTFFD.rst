Goal 04-A：Geometry-aware Joint 4D HeartFFDNet
============================================

范围与代码复用
--------------

这是独立的 ``geometry-ma-joint-ffd-v1``，只使用 Known-Pose 合成输入。
本轮没有修改原 Baseline、合成数据、MaskTransform、物理 FFD、九项 Loss 或
原生 Exact Dice 核心模块；没有启动完整 1K/10K 训练。

按照用户后续选择，原拟拆分的 ma_spatial_unet.py / ma_joint_ffd.py 已删除。
三维主干和 Head 适配类合并在 ``net/geometry_heartffdnet.py``，直接复用
``net/ffdnet.py:FFDNet`` 实例的 conv2–conv5、deconv2–deconv5、deform1–deform3。
只新增联合输入卷积，将该实例 conv2 的步长改为1，并适配批次、空间 Skip 和毫米控制场。
原始文件和新建的其他 FFDNet 实例不受影响。Head 保留原有小随机输出初始化。

已实际阅读 Ma 原始代码，并和仓库中的相同实现核对：

* `ffdnet.py <https://github.com/m-qiang/HeartSSM/blob/main/net/ffdnet.py>`_
* `module.py <https://github.com/m-qiang/HeartSSM/blob/main/net/module.py>`_
* `train_ffd.py <https://github.com/m-qiang/HeartSSM/blob/main/train_ffd.py>`_
* 仓库内 ``utils/ffd.py`` 的原始基函数和递推应用思想。

原 forward 不能原样调用：其输入为稠密体数据、默认50帧，两个参考输出使用
Batch=1索引，控制场补零尺寸与当前物理控制域不兼容。
``net.module.ConvBlock`` 的 InstanceNorm3d→SiLU→Dropout→Conv3d 原样复用。
Ma 的共享三维编码器、空间 Skip、多尺度解码和 3*n_frame 通道 Head 得以保留。

新增文件及职责
--------------

* ``net/geometry_heartffdnet.py``：统一模型、复用原 FFDNet 层的 U-Net、单次 Joint
  Head、有界控制场、共享物理 FFD 递推以及 Loss 字段适配。
* ``net/geometry_spatial_encoder.py``：保留现有 GroupNorm ResNet18 的空间层及
  spatial_features()；无 Attention 残差、Shape/Motion latent 或旧 FiLM。
* ``net/geometry_lifting.py``：物理 Pose→原生像素→模型像素→CNN 特征采样、距离
  权重融合和明确的 Unknown / 腔室标注支持字段。
* ``train_geometry_heartffdnet.py``：独立训练、三阶段短程验收、严格 Checkpoint
  Resume、正式 Loss 适配、几何拒绝门及原生 VAL Exact Dice 调用。
* ``utils/geometry_joint_checks.py``：源模板/输出三角形自交、退化、支持域、
  控制场导数上界及采样组合 Jacobian 检查。
* ``configs/geometry_heartffd_1k.yaml``：固定数据/输入合同、网络宽度、Known-Pose、
  历史正式九项 Loss、短程与后续训练参数。
* ``tests/test_geometry_heartffdnet.py``：解析几何、实际模块复用、Batch/时间、
  缺失通道、真实参数梯度、FFD 递推、严格保存恢复和下一步优化器一致性。
* ``scripts/check_geometry_heartffdnet.py``：真实 10631 顶点、224像素输入的
  Batch=1/2、一次 CNN/Head 计算、参数梯度、尺寸、耗时和内存检查。

数据流与张量
------------

输入 Mask/SDF 均为 [B,24,3,4,224,224]，视图按 A2C/A3C/A4C 排列。
未标注腔室在 CNN 前屏蔽；整视图无有效标注时在 CNN 后也屏蔽。
ResNet18 layer2/layer3 的两尺度空间输出分别为
[B,24,3,128,14,14] 和 [B,24,3,128,7,7]，每个输入只经过一次 CNN。

每尺度先共享投影到4通道，随后物理采样并按视图权重融合。
每帧有8个观察特征通道、1个观察支持通道、1个最近平面距离通道以及4个腔室
标注支持通道，总计14通道。Lifting 输出 [B,24,14,24,24,32]，沿时间和通道
展开为 [B,336,24,24,32]。全部相位共同输入三维 conv0，时间位置不求均值。

U-Net 的 Fine / Medium / Coarse 编码、解码特征分别为
[B,32,24,24,32] / [B,32,12,12,16] / [B,32,6,6,8]；三个尺度均有患者空间 Skip。
单个 deform3 一次输出 [B,72,24,24,32]，按通道分组为24个 XYZ 控制场，转为
[B,Time=24,Gx=24,Gy=24,Gz=32,XYZ=3]。
Coarse / Medium 分别输出 [B,6,6,8,3] / [B,12,12,16,3]。

先应用 Coarse、Medium、Fine0 得到 vertices_ref=[B,10631,3]，再按当前顶点
递推应用 Fine1…Fine23，得到 vertices_4d=[B,24,10631,3]。
神经网络只运行一个 Joint Head；逐帧循环只在物理 FFD 应用阶段。
四腔共享同一控制场，顶点顺序和拓扑固定；Frame23 保留独立采样值。
既有 velocity Loss 和评价保留真实的 Frame23→Frame0 时间间隔。

坐标与 Unknown 合同
-------------------

采用原 Renderer 的行向量关系 q=(X_mm/norm_to_mm-t)@R；d_mm=q_z*norm_to_mm。
原生像素 x=(q_x/(probe_s_x+1e-8)+1)*W_native/2，y 同理。
原生像素维数由 probe_s 和 native_pixel_spacing 校准推得；无需读取 GT Mesh 或
native_masks。再用 model_to_native 的逆仿射得到模型图像整数像素坐标。
因此保留 Renderer 的 W/2、H/2 原点，不混用 (W-1)/2。

CNN 采样按真实步长和 adaptive-pool 区间中心计算，而不是直接把224像素当作
14/7像素索引。ResNet layer2/layer3 的中心步长为8/16；224输入池化后，14尺度
中心为4+16*j，7尺度中心为8+32*j。反解相应特征索引，使用 grid_sample 的
align_corners=True。图像内边缘到首末特征中心的区间使用边界特征；图像外
采样先明确无效，不能作为观察输入。

查询点使用实际 Fine 控制节点 origin+i*spacing，XYZ 对应数组 Gx/Gy/Gz。
plane-distance 权重为 exp(-0.5*(d/8mm)^2)，24mm外权重为0；缺失视图、无标注
视图和模型 FOV 外点权重为0。无任何支持时观察特征为0、support=False，并
保留 Unknown 距离编码；未标注通道不会被当作背景。Pose 可随病例及相位变化。

物理形变与非法几何处理
----------------------

直接调用既有 control_geometry() 与 bspline_ffd_physical()。
三尺度使用同一 [-120,-110,-110] 到 [120,110,110] mm 支持域。
参考控制各XYZ分量限制为12mm，逐步运动控制各分量限制为2mm；均使用可导 tanh。
再由相邻控制差/物理间距计算导数算子范数的保守上界，必要时统一缩放到0.45。
这限制光滑 FFD 的局部折叠，不引入自由顶点位移或额外 Loss 项。

源模板各腔非相邻三角形自交先独立检查。每次训练检查输出非有限值、支持域、
实际三角形退化和控制场导数上界，失败即停止，不跳过非法病例。
初始/最终验收及原生评价另外检查完整24帧三角形自交，并采样每一递推步骤的
组合 Jacobian。网格三角片的实际自交检查独立于光滑 FFD 的导数证书。
检查不等于真实患者病理有效性认证。

监督与 Checkpoint
-----------------

保留原 shape/motion/velocity/volume/edge/laplacian/normal/control/slice 九项
Loss 及权重 1/1/0.2/0.01/0.05/0.05/0.01/0.001/0.5，Huber beta=1mm。
Joint 使用 binary_native_v1；shape、motion 阶段使用原有 stage 权重开关。
Loss 适配层只将 fine 名称映射到 fine_motion，将不存在的 regional 控制设为None。
没有修改 GT、Slice Target 或核心计算。

训练数据保持800 TRAIN / 100 VAL / 100 TEST，原 NPZ/QC/GT 未修改。
入口固定核对五项 manifest/asset/template 哈希，数据读取继续执行既有身份检查。
TRAIN小样本固定为原训练列表前4例（candidate_id=376/112/620/604），没有病例筛选。
模板只加载共享几何缓冲区，CNN/U-Net/Head 所有可学习参数从零随机初始化。

Checkpoint 版本为 geometry-ma-joint-checkpoint-v1，保存 model_state、optimizer_state、
epoch/step/sample_offset/training_stage、config、data_identity、geometry_identity、
pose_mode、runtime 和 RNG；固定模板/拓扑/控制格/查询格点逐一核对，strict=True加载。
旧 NeuralFFD Checkpoint、改变数据、配置或控制域均拒绝。保存 latest 与 best_stage。
仅显式 --train 可以启动正式训练；--acceptance 的144更新不会延伸为完整1K训练。

验收结果
--------

**PASS：新架构工程合同和固定4例TRAIN短程学习验收通过。**
32次Shape、96次Motion、16次Joint，共144次更新；在20步Checkpoint处实际停止并
严格Resume，最后重载最新Checkpoint，输出逐元素一致。
正式九项Loss权重及binary_native_v1不变；短程学习率为0.001，正式配置为0.0001。

.. list-table:: 固定TRAIN小样本指标（mm，Volume为mL）
   :header-rows: 1

   * - 阶段
     - Reference RMSE
     - Full 4D RMSE
     - Relative Motion RMSE
     - Volume MAE
   * - 随机初始化
     - 5.675
     - 7.116
     - 4.579
     - 27.737
   * - Shape 32更新
     - 4.926
     - 6.342
     - 4.579
     - 22.574
   * - Motion结束（128更新）
     - 3.155
     - 3.669
     - 2.448
     - 6.300
   * - Joint结束（144更新）
     - 3.120
     - 3.424
     - 2.349
     - 5.226

最终Reference / Full4D / Motion误差分别下降45.03% / 51.88% / 48.70%，超过
预先规定的Shape和Motion至少下降10%、Full4D下降门槛。
最终四腔分项（病例等权RMSE、病例平均幅度/容积误差）：

.. list-table:: 四腔运动响应与容积
   :header-rows: 1

   * - Chamber
     - Reference RMSE mm
     - Motion RMSE mm
     - Motion Amplitude Ratio
     - Volume MAE mL
   * - LV
     - 2.216
     - 2.273
     - 0.864
     - 4.346
   * - LA
     - 2.555
     - 1.905
     - 0.846
     - 4.558
   * - RV
     - 3.242
     - 2.575
     - 0.859
     - 6.891
   * - RA
     - 4.120
     - 2.577
     - 0.870
     - 5.108

所有更新梯度有限，Joint Head梯度L2最低1.391；无长期静态运动。
最终4例全部24帧：实际非邻接三角形自交0，退化三角形0，采样组合Jacobian最小
0.203573，最大控制场导数上界0.446120。没有非有限值或支持域越界。
各阶段CPU平均每次更新8.62 / 7.41 / 11.69秒。

结果以 ``outputs/geometry_goal04_final/acceptance.json`` 为准。
阶段及分腔简表见同目录 stage_summary.json；工程记录见 engineering.json。
完整阶段指标见 after_shape.json / after_motion.json / final.json。
每次更新的Loss、四腔运动、容积、控制RMS、梯度及几何状态写入history.jsonl。
Checkpoint为latest.pth及best_shape.pth / best_motion.pth / best_joint.pth。

已验证真实 Batch=2 网格 [2,24,10631,3]，CNN观察总数144，Head调用次数1。
实际参数量 3,364,646；CPU Batch=2前向8.84秒，进程峰值RSS约2.24GiB
（包括该进程的数据、前向及后续B=1梯度检查；它不是GPU显存）。
当前环境没有 /dev/nvidia*，GPU显存及GPU吞吐未测量。

最终Checkpoint已实际执行1例VAL原生Exact评价，192个有效Cell的GT重绘与保存标签
逐像素一致，GT mismatch=0；预测Mean Exact Dice=0.913003。
VAL1 Reference / Full4D / Motion RMSE分别为4.637 / 5.197 / 2.786mm。
这仅验证独立评价入口可运行；不是VAL100性能验收或历史基线比较。
评价报告为outputs/geometry_goal04_val_smoke/evaluation.json。

11项新架构单元测试通过。原72项测试中有1项旧 DynamicSpatial 的早期帧
逐元素相等断言失败，单独执行也复现。原模块和原断言保持原样，不将其宣称为
浮点误差，也不把旧分支诊断扩展为本 Goal 的工作；简短记录见这里即可。

已执行的命令
------------

以下从仓库根目录执行，Python为本轮实际 /tmp CPU环境::

    /tmp/heartssm-goal04-env/bin/python -m unittest discover -s tests -p test_geometry_heartffdnet.py -v
    /tmp/heartssm-goal04-env/bin/python scripts/check_geometry_heartffdnet.py --device cpu --output outputs/geometry_goal04_final/engineering.json
    /tmp/heartssm-goal04-env/bin/python train_geometry_heartffdnet.py --acceptance --device cpu --max-steps 20 --output outputs/geometry_goal04_final
    /tmp/heartssm-goal04-env/bin/python train_geometry_heartffdnet.py --acceptance --device cpu --resume outputs/geometry_goal04_final/latest.pth --output outputs/geometry_goal04_final
    /tmp/heartssm-goal04-env/bin/python train_geometry_heartffdnet.py --evaluate outputs/geometry_goal04_final/latest.pth --device cpu --val-cases 1 --exact --output outputs/geometry_goal04_val_smoke

20步命令有意触发不完整验收，返回FAIL/退出码2，然后严格Resume至固定144步预算。
临时环境的软件版本将在输出目录 environment.txt 保存；原生评价使用历史
OpenCV 4.10.0 / trimesh 4.11.5。CPU PyTorch版本与历史GPU环境不同。
后续用户可在自己的相同依赖环境使用 python 替换此路径。

下一阶段
--------

本次TRAIN小样本是工程学习能力门槛，不能与VAL100科研精度混为一谈。
完整从零1K训练、VAL100正式比较、GPU显存/吞吐、10K和未知位姿/真实患者TTO
尚未执行；它们需要各自后续任务。未对历史Epoch162指标作性能超越声明。
