# 研究上下文：两条路线、已确定决策与参考文献

## 1. 区分提出者与本轮范围

### 路线 P：PhD mentor 的事件表征改进——本轮实现

已确定的方向是：从当前 pairwise 主线出发，把 cosine InfoNCE 改为已讨论的 AnInfoNCE，并学习五个表征空间。用户选择方案 B：共享主干、五个 head、四组可计算的物理辅助目标。

五空间不是五个粒子、五条 jets、五种投影或五个增强视图。它们是同一事件的五份 64-D 表征。general/energy/eta/phi/local 是项目采用的任务命名；不能暗示 PhD mentor 逐一指定了所有观测量、谱阶、尺度和工程参数。

此前也讨论了 donor overlay，但当前用户指定的前两项工作是重构，以及 loss+五空间。新增 overlay 不应延长这两项的交付关键路径。已有接口允许后来接入即可。

### 路线 T：教授的对象级/来源级物理理解——未来独立里程碑

用户转述的方向为：原生 3D hit cloud → graph coarsening/hierarchy → particle-like objects 或真正粒子候选 → OT 或 copy-and-paste → 理解多个重叠来源。

它需要明确：操作对象到底是 energy-flow pseudo-particles、clusters、reconstructed candidates 还是 truth particles；训练目标到底是压缩保持能量分布，还是重建对象数、来源份额和四动量。当前版本不代替用户/教授决定这些问题，也不把事件级物理摘要当作已经完成粒子级重建。

已讨论的一个具体 benchmark 是 `Y_c=A_c+M_c B_c`：在相同原生 cells 中，完整加入 B 被 mask 选中的后段能量，模型只看求和后的 Y，来源能量和份额在监督端保留。它描述的是部分观测来源的 synthetic deblending，不自动对应两条完整可观测 b jets。

## 2. 两条路线怎样合理连接，而不是强制拼接

### 现在可以共用

事件 ID、来源 split、线性能量语义、单位与 geometry metadata、输入与目标分离、可追溯 checkpoint、节点编码器/事件 readout 的明确边界、按物理定义做验证的方式。

### 现在不直接混合

当前对比任务可能要求同一事件的不同扰动视图相近；来源分解任务必须保留 full-strength 新来源。把混合事件的全局表示强行拉到 A，同时让同一唯一表示完整表达 A+B，是不同目标的潜在冲突。先定义每个表征/对象头负责什么，再决定结合。

五个全局 heads 不能代替可变数目的对象 slots；二维网格丢失的深度信息不能由输出头恢复；加权平方距离或 OT 也不能独立证明粒子数量与身份。

### 未来连接的可检验形式

一种合理连接是：教授路线先从混合 hit graph 重建对象/来源，再对重建出的目标对象应用当前表征和物理读出。另一种是共享早期节点 encoder，但分别设置事件表示目标和细节点来源 decoder，并检查梯度/任务收益。

只有明确了对象定义、数据监督与比较任务后，才进入共享训练。这里记录设计方向，不要求 Codex 本轮搭建这些模块。

## 3. 已确定但不在本轮必做的增强协议

### 3.1 相对能量预算 overlay

对共享固定物理网格的目标 E_i 与 donor E_b：

\[
S_i=\sum_cE_{i,c},\quad S_b=\sum_cE_{b,c}>0,\qquad
\widetilde E_{i,c}=E_{i,c}+\beta\frac{S_i}{S_b}E_{b,c}.
\]

新增能量是 beta*S_i；beta 是相对目标原有能量的比例，背景占最终总能量的比例为 beta/(1+beta)。系数 alpha=beta*S_i/S_b 由事件决定，不再独立调节。后续不把总能量缩回 S_i。

逐事件、逐视图独立抽 donor；可以 batch 向量化，但不使用 batch 总能量预算。训练 donor 来自训练池；不强制跨类别配对。线性能量叠加在 log/标准化前。阈值之后的能量比例不保证仍等于 beta，需以实际流程解释。

这是结构化背景扰动，不是后段 full-strength 来源组合的替代。固定 beta 时恢复 clean total energy 有解析校正，不以这个容易任务的高精度声称模型学会复杂分离。随机 beta 是可选择的研究设计，不在本轮默认打开。

### 3.2 能量守恒粗粒化

相邻 bin 能量求和，不做平均；保持物理边界，保存与原分辨率的关系。若后来启用，observed 与 reference 同分辨率，物理角尺度取实际坐标单位而不是偷偷随像素重定义。

这不等于教授提出的 learned graph coarsening：前者是输入的规则聚合增强，后者是可学习的层次表示与对象结构。

## 4. 之前两个参考代码包的地位

`hep_ssl_plan_b_reference.zip` 包含物理 targets、五头、AnInfoNCE、联合 objective 和 synthetic tests，不包含 ColliderML loader、GravNet backbone、完整 trainer 或 CHTC 接入。

`tail_overlap_reference.zip` 属于教授路线中的后段信号叠加与来源份额 benchmark，不属于本轮必须迁移的代码。

这两个附件不保证已经位于 Codex 的工作目录。**当前说明本身给出了完成本轮所需的数学和接口，不依赖 Codex 能够访问旧聊天附件。** 若附件已由用户放入工作树，可以借鉴或迁移函数，但要按新的模块职责拆分，不把一个大参考模块原样复制成第二套框架。

准备此说明时，重新执行了方案 B 参考模块的 synthetic 脚本，输出的六组检查均通过：目标的基本不变性与尺度关系、理想方位例子、overlay/粗粒化守恒、Lambda=I 的 loss/梯度等价、各 head/readout/Lambda 更新、state round trip。它不代表真实 ColliderML 或完整 PyG 模型已经通过。

尤其注意参考模块不是权威默认配置：它的测试温度、合成网格范围、local scales 和小 MLP 宽度不应覆盖项目配置。本轮实施说明规定的统一数据 split、真实能量 pooling、目标/reference 时序和完整 checkpoint 才是接入契约。

## 5. 参考文献与使用边界

文献列入参考不等于要求本轮实现。以下短述区分文献的直接内容与本项目的设计选择。

### R1 — 本轮直接相关：AnInfoNCE

**InfoNCE: Identifying the Gap Between Theory and Practice**，arXiv:2407.00143。

[arXiv 摘要与全文入口](https://arxiv.org/abs/2407.00143)

用途：可学习各向异性距离及其理论动机。项目采用归一化投影上的正对角平方距离，并明确固定 trace 和温度。这个工程版本不是对论文全部数据生成假设、可识别性保证或官方代码的逐行复刻；不要承诺下游准确率必然提高。

### R2 — 未来骨干/训练参考：Sonata

**Sonata: Self-Supervised Learning of Reliable Point Representations**，arXiv:2503.16429。

[arXiv](https://arxiv.org/abs/2503.16429)

用途：点云自监督表示及几何捷径问题。不是本轮要求加入的 teacher–student 蒸馏系统。当前辅助目标与 two-view AnInfoNCE 不应被改写为 Sonata 而仍沿用同一实验名。

### R3 — 未来骨干参考：PTv3

**Point Transformer V3: Simpler, Faster, Stronger**，arXiv:2312.10035。

[arXiv](https://arxiv.org/abs/2312.10035)

用途：点云编码器架构参考。与 loss/物理辅助目标是不同改动轴；本轮保留 GravNet。

### R4 — 增强不变性背景

**Rethinking the Augmentation Module in Contrastive Learning: Learning Hierarchical Augmentation Invariance with Expanded Views**，arXiv:2206.00227。

[arXiv](https://arxiv.org/abs/2206.00227)

用途：理解增强组合、层次化不变性与扩展视图。不能据此把方案 B 改成层级增强或增强参数预测任务。

### R5 — 方案 A 的来源，非本轮目标

**What Should Not Be Contrastive in Contrastive Learning**，arXiv:2008.05659。

[arXiv](https://arxiv.org/abs/2008.05659)

用途：LooC 的不同不变性与变化相关信息保留。用户本轮选择的是方案 B；因此不使用 LooC 六视图或同事件敏感分支负样本。

### R6 — 用户新增：物理距离保持的 embedding

Sang Eon Park, Philip Harris, Bryan Ostdiek，**Neural Embedding: Learning the Embedding of the Manifold of Physics Data**，arXiv:2208.05484；JHEP 07 (2023), 108。

[arXiv](https://arxiv.org/abs/2208.05484) · [DOI](https://doi.org/10.1007/JHEP07%282023%29108)

用途：把具有物理度量的高维数据映射到较低维度量空间。适合作为后续 EMD/距离保持目标的参考；与“预测指定物理摘要”是不同监督方式。本轮不默认新增 pairwise EMD 标签计算、hyperbolic encoder 或 OT loss。

### R7 — 数据定义

**CERN/ColliderML-Release-1**。

[官方 dataset card / schema](https://huggingface.co/datasets/CERN/ColliderML-Release-1)

当前需要 calo_hits 的 event_id、total_energy 和坐标；保留必要元数据。数据卡将 total_energy 定义为 cell 沉积能量 GeV、位置为 mm，粒子贡献列表为 truth provenance。使用官方字段，不从变量名臆测 particle-level 输入。

### R8 — 核心代码证据

[已审阅分支](https://github.com/imaneekjana/hep-ssl/tree/Kunhe-Li)

以下链接固定到已审阅 commit，便于区分旧代码事实和新规格：

- [pairwise 训练入口](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/src/train_planar/train_colliderml_planar_pairwise.py)
- [数据与投影](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/src/data/dataset.py)
- [GravNet 与当前 pooling](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/src/models/gnn.py)
- [当前 InfoNCE 与 trainer](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/src/models/contrastive_learning.py)
- [pairwise 下游评估](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/src/evaluation/run_pairwise_classifier.py)
- [CHTC wrapper](https://github.com/imaneekjana/hep-ssl/blob/994cd472b19b7fde8d7c295cec23eb696b08323d/chtc/pretraining/run_experiment.sh)

### R9 — Codex 项目指令

[官方 AGENTS.md 文档](https://developers.openai.com/codex/guides/agents-md)

根 AGENTS 文件只放稳定约定与阅读入口；具体数学和当前阶段验收放独立说明。项目已有指令应合并而非覆盖；详细文件需要在任务开始时显式读取。

## 6. 用词与结论

当前方案 B 是 physics-guided auxiliary supervision / self-supervised observable prediction 与 contrastive learning 的组合。目标来自输入数据可计算摘要，而不是新增人工 process 标签；但它也不是没有任何人为先验的纯无监督学习。

energy、eta、phi、local 代表希望可读出的信息类别，不代表证明了潜在物理因子的唯一分解。general 可以和其他空间共享信息；不同物理观测量本来可能相关。

最终科研结论需依赖固定数据协议、必要的结构性消融、物理读出与下游迁移结果。代码能运行、辅助 loss 变小、输出五个不同向量，都不能单独代替这些证据。
