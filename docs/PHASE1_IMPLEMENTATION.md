> 本文件保留任务原始规格；当前实现、运行方式和验证状态见 [PHASE1.md](PHASE1.md) 与 [VALIDATION.md](VALIDATION.md)。

# 第一阶段实施说明：pairwise 重构、AnInfoNCE 与方案 B 五空间

**状态：待 Codex 在实际工作树中实现的规格，不是已完成代码的报告。**

参考仓库：`imaneekjana/hep-ssl`，分支 `Kunhe-Li`。本说明核对的快照：`994cd472b19b7fde8d7c295cec23eb696b08323d`。本地可能已有更新；先核对，保留用户变更，不根据此 SHA 回退。

本文区分三种信息：**用户已选定的方向**、为把方向变成可运行代码而规定的**首版工程决策**、**未来研究**。物理观测量组合、谱阶与尺度等是讨论中的项目设计，不声称是导师逐项指定或文献保证的最优解。来源链接在 `RESEARCH_CONTEXT.md`。

## 1. 范围与成功标准

### 1.1 本轮必做

用户已选定：先重整 pairwise 代码，再实现 PhD mentor 提出的 AnInfoNCE 和五个 embedding spaces；五空间采用方案 B 的物理辅助任务，不采用 LooC 方案 A。

成功交付必须是可运行的端到端路径：

```text
ColliderML calo_hits
→ 固定物理网格上的能量聚合
→ 两个增强视图 + 各自对应的无污染参考目标
→ 稀疏节点 + 共享 GravNet
→ 五个 64-D 表征 / 五个 32-D 对比投影 / 四个物理读出
→ 联合训练、保存恢复
→ 单视图表征提取、下游分类与物理信息评估
```

重构应使人能够直接找到每项数学操作和数据语义；不只是换目录或格式化代码。训练参数保留可配置，但不开展密集 sweep，不分析旧训练结果来决定实现。

### 1.2 本轮不要求实现

教授路线中的原生 3D 层次图粗化、粒子候选、b-jet 组成识别、tail overlay、逐 cell 来源份额预测、OT/EMD loss、Geant4 联合模拟，以及 PTv3/Sonata 替换骨干。它们保存在研究上下文中，不作为当前任务的依赖。

此前的相对能量预算 overlay 与能量守恒粗粒化增强也不在本轮必做范围。保留输入/参考双路径接口即可；不要为尚未使用的功能增加空模块、未实现选项或复杂注册系统。若本地已实现这些功能，保留在独立可关闭路径，默认配置不得自行启用。

### 1.3 为什么本轮不强行合并两条路线

PhD 路线目前学习事件表示及可读出的物理摘要；教授路线需要由混合信号恢复多个来源。弱背景可被当前任务视作干扰，但 full-strength donor 在来源分解任务中是应识别的对象。两种训练关系不能不加区分地放进同一个“不变性”loss。复用读取、能量语义、统计量、checkpoint 和表征接口是合理的；合并输入层级与监督目标现在没有必要。

## 2. 已核对的代码结构与整改点

主入口为 `src/train_planar/train_colliderml_planar_pairwise.py`；不要把 `src/train_planar/train_colliderml_planar.py` 当作最新逻辑。

| 现有位置 | 当前职责 / 核对到的问题 | 目标 |
|---|---|---|
| `src/data/dataset.py` | `ColliderMLHits` 取 x/y/z/total_energy；事件身份没有向模型管线保留 | 分离原始事件适配、稳定 ID、split 与图数据 |
| 同文件 `projected_hits` | 每个事件/视图分别取 min/max；请求 eta-phi 仍计算全部投影 | 只计算所需投影；统一固定 GridSpec |
| 同文件 `bin_points_to_grid` | 先能量求和再 log，但变量名容易把原始 E 误称 logE | 用明确名称、单位和张量排列 |
| 同文件 `EventGraphBuilder` | 即使 radius=0 仍构造所有节点对距离 | GravNet 路径直接创建 Data，不预计算不会使用的边 |
| `src/data/augmentation.py` | transform 内采样并修改副本；采样状态不显式 | 采样与应用可区分，支持与干净参考重放相同几何 |
| `src/models/gnn.py` | GravNet→节点 64-D→mean/max/weighted 三池→192→event_mlp→64→projection→32 | 提取 backbone/pooling/head；在 event_mlp 前分叉 |
| 同文件 | 用标准化后的 log-energy 做 softmax pooling 权重 | 单独保留原始 E，用 E/sum(E) 加权 |
| `src/models/contrastive_learning.py` | trainer 与 cosine InfoNCE、checkpoint 耦合；两视图硬编码 | 分开 loss、objective 与 trainer，仍明确只支持两视图 |
| pairwise 入口 | optimizer 只含 model.parameters() | 增加 objective 内的所有 Lambda 参数 |
| `src/evaluation/run_pairwise_classifier.py` | 重新读取整个样本集并重新划分下游数据 | 编码器与下游共用持久化事件 split |
| `run_pairwise_random_classifier.py` | 与预训练评估大量重复 | 同一评估入口，切换 encoder 初始化来源 |
| `linear_probe.py` | 旧路径及输入标准化约定与当前 pairwise 不一致 | 新流程不从这里复制第二份 preprocessing |
| 训练入口 / CHTC wrapper | 移动文件后仍有旧入口路径、root parents 偏差；部分参数未实际使用 | 修正真实执行路径，统一配置入口，删除误导性参数/打印 |

上表是快照审阅，不是宣称所有问题在用户机器上都已导致报错。Codex 应检查实际依赖和 wrapper 的打包布局，再迁移。

### 2.1 把纯重构和方法修正分开

纯重构包括移动函数、消除重复 imports、集中 checkpoint、消除不使用的边计算、增加主函数边界。用固定输入、权重、增强状态的小样本回归检验其输出一致性。

固定网格、物理能量 pooling、事件级 split、新的 194-D pooled 输入是**方法修正/设计改变**。它们构成新的共同 baseline，不能声称复现旧训练曲线。先记录旧行为，再在独立变更中修正。无需把全部错误旧行为长期维护成第二套生产路径；少量旧公式/固定 fixture 可留在测试中。

历史 checkpoints 不保证与新模型直接兼容。保留旧文件及对应代码版本；新 checkpoint 使用新 schema。不要用 `strict=False` 静默跳过不匹配权重。若做旧→新迁移，显式说明迁移哪些参数、哪些初始化、哪些统计量不兼容；迁移不是本轮必需品。

## 3. 建议模块边界

保留 `src` 为包名以降低迁移成本；不强制另造 `src/hep_ssl`、复杂打包或 Hydra。以下为建议布局，允许在职责不重复的前提下少量合并文件：

```text
AGENTS.md
configs/
  pairwise_base.json
src/
  __init__.py
  config.py
  prepare_pairwise.py
  train_pairwise.py
  evaluate_pairwise.py
  data/
    events.py
    projection.py
    augmentation.py
    views.py
  physics/
    targets.py
  models/
    gravnet.py
    multispace.py
  losses/
    contrastive.py
    multitask.py
  training/
    trainer.py
  evaluation/
    representations.py
    classification.py          # 尽量复用、整理已有分类函数
  train_planar/
    train_colliderml_planar_pairwise.py  # 如保留，只是兼容薄 wrapper
chtc/
  pretraining/
  classifier/
tests/
docs/
```

模块 import 不得开始下载、解析命令行、训练或改工作目录。使用 `if __name__ == '__main__': main()`。提供 `python -m src.prepare_pairwise`、`python -m src.train_pairwise`、`python -m src.evaluate_pairwise` 的帮助和实际工作命令。旧 CLI 只通过少量 alias/wrapper 转到同一配置和 main。

一个直接的嵌套 dataclass/JSON 配置足够。只为实际使用的功能保留配置。已使用的分类器仍可保留，但可选 xgboost 依赖不应阻止基本训练导入。复用现有 PyTorch/PyG 环境，不无故升级依赖或换 optimizer。

原项目已经把有限事件表加载入内存，因此 map-style Dataset 是合理首版，避免多层 IterableDataset 嵌套及多 worker 重复迭代。若保留 iterable，必须正确分片；不要仅增加 `num_workers` 就宣称并行读取已正确。

## 4. 数据、几何、能量与 split

### 4.1 原始事件身份

来源为 `CERN/ColliderML-Release-1`、`pu0`；保留 ggf/ttbar/dihiggs 三组二分类任务和 channel 参数化。一次 pairwise run 使用两种不同 channel，不把 process label 作为模型特征或对比正样本定义。

事件 key 至少为 `(dataset_revision, channel, pileup, event_id)`。不能单独假设 event_id 跨 channel 唯一，也不能使用当前行号作为稳定身份。若上游记录需要 shard 标识去歧义，应在适配器中包含它；不要通过随机新 ID 掩盖重复事件。

先确定事件集合和 split，再做任何统计量拟合。新 baseline 使用固定种子的按 channel 分层 60/20/20 划分；比例沿用当前 pairwise 的职责，分层和持久化是明确修正。若工作树已有用户指定 manifest，优先复用。下游分类 train/val/test 不另造随机测试集。

保存 manifest，内容包括稳定 IDs、数据版本、channel、划分角色及构造配置。真值类别可用于分层和下游评估，但不能进入 SSL/辅助目标特征。已读测试输入不能用于拟合 grid 范围或统计量。

### 4.2 原始单位和投影

官方 calo 数据使用位置 mm、能量 GeV；实际适配时确认没有另行校准或单位转换。不要把无量纲 log-energy 当作 GeV。

投影定义为：

\[
\rho=\sqrt{x^2+y^2},\quad
\phi=\operatorname{atan2}(y,x),\quad
\eta=\operatorname{asinh}(z/\rho).
\]

原点/轴上异常坐标按明确的有效性与接受范围策略处理并计数，不用随意加大 epsilon 伪造角度。

`GridSpec` 保存真实 bin edges、轴顺序、接受范围、单位、cell cutoff 和投影版本。phi 固定 `[-pi, pi)`；把 +pi wrap 到 -pi，跨边界角差使用周期公式。eta 对所有事件和视图固定，不再逐事件 min/max。

eta 接受范围不是可以凭空填成“探测器范围”的常数。`prepare` 优先使用已有明确配置或可验证 geometry。没有该信息但有数据时，首版可从该比较队列**训练池**的有限 eta 取共同对称包络、略扩展数值边界并冻结，标记 `range_source=training_envelope`，不是 detector acceptance。所有消融共享这份准备产物；不同数据队列不要声称其 grid 完全相同。真实数据和明确范围都不存在时，可继续 synthetic tests，但不要捏造真实实验范围。

超出固定范围的 hit 不挤入边缘 bin；丢弃并记录能量份额。grid 不是 image 输入：它是线性能量聚合的中间表示，之后仍转为稀疏节点。

### 4.3 统一能量定义

\[
E_{ba}=\sum_{i:(\eta_i,\phi_i)\in C_{ba}}E_i,
\qquad E.shape=[n_\phi,n_\eta].
\]

batch shape 为 `[B,n_phi,n_eta]`，不能把 eta/phi 两轴悄悄交换。默认沿用 32×32，且只计算 eta-phi，不同时预计算无用投影。

用预先固定的 GeV cutoff 对 clean/reference 和 observed 分别处理；首版 cutoff=0、保留正能量 bin。两条路径使用同一**阈值规则**，不是同一个由 corrupted input 决定的 mask。观测被遮掉的 cell 不应同时从干净目标中删掉。未来改阈值时要记录该目标定义变化。

稀疏输入行：`[eta_center, phi_center, log(E/E_ref + epsilon/E_ref)]`。首版 `E_ref=1 GeV`、`epsilon=1e-6 GeV` 对应旧 log 稳定项的单位化写法。标准化使用训练池的干净有效节点统计量，按节点加权；验证/测试只读取。

`graph.energy` 独立保存阈值后的线性观测能量。所有 pooling、观测能量摘要和目标计算不得反过来从标准化特征猜原始能量。

不在这一阶段偷偷把三维节点输入改成 `[eta,sin(phi),cos(phi),logE]` 四维；保留当前三特征，正确处理增强与目标中的周期性。编码器本身的连续旋转不变性不是此输入格式自动保证的。

### 4.4 空样本、小图与小 batch

零接受能量事件不能生成概率目标；在准备阶段用确定规则标记并计数。不要给空事件凭空添加有能量的假节点。

稀疏图的邻居数限制按实际 PyG 实现验证。小图若不受支持应显式报出约束；不静默丢弃或复制 hits。不要用整个 batch 最小图大小改变所有其他图的 k，使同一事件因同批其他事件不同而改变输出。

InfoNCE 每个 batch 至少两个不同原始事件。train 可沿用 drop_last；val/test 对末尾 singleton 用确定的 batch sampler 合并到前批，不复制一个事件冒充负样本、不把零对比损失纳入均值。用于同一消融比较的评估分批必须一致。

## 5. 两视图与无污染参考：明确到每一步

方案 B 仍然只有两个视图。支持当前 pairwise 真正使用的 rotate、energy_noise、xyz_noise、shift、crop 及其显式顺序；不要因为重构擅自改变用户提供的参数。首版配置从现有入口/用户配置转录，不从旧训练结果挑选超参数。

每个 transform 的随机状态应可采样并重放，不必设计大型 transform 框架。概念接口：

```python
params = transform.sample_params(observed, rng)
observed = transform.apply(observed, params)
```

每个视图从原事件的两个独立副本开始：`observed` 与 `reference`。

- 坐标变换：rotate、shift、xyz_noise 的同一组实际采样坐标参数应用于二者。
- 能量污染/遮挡：energy_noise、crop 只应用于 observed；reference 保留未污染能量。
- 按原配置顺序执行，不擅自重排非交换变换。crop 中心可以从该步骤已有的观测几何采样，但不因此修改 reference 的能量。
- 最后两条路径各自投影到同一 GridSpec，使用相同阈值定义。物理目标从 reference 计算，模型输入从 observed 计算。
- 坐标抖动/平移不被宣称为严格碰撞对称性。这个阶段的含义是“同一已变换坐标系中的能量去噪目标”，不是同时监督模型反演所有几何增强。

形式上：

\[
E_{i,\mathrm{obs}}^{(v)},\;E_{i,\mathrm{ref}}^{(v)}
\longrightarrow
\begin{cases}
G_i^{(v)}=\mathrm{nodes}(E_{i,\mathrm{obs}}^{(v)}),\\
t_i^{(v)}=\mathrm{PhysicsTargets}(E_{i,\mathrm{ref}}^{(v)}).
\end{cases}
\]

两个视图可以有不同目标数值，保存 `targets1` 和 `targets2`，不要盲目复制原事件的绝对 eta 标签。固定输入尺寸且只有 phi 旋转时，一些目标相等是数学性质，不代表所有增强下都相等。

未来粗粒化被启用时，clean/reference 与 observed 使用同一层级、能量求和而非平均；目标在匹配分辨率上计算。未来 overlay 的确定公式另见研究上下文。本轮无这些功能时直接走上述两条路径，不创建只含 `pass` 的扩展类。

所有标签在 no_grad 下构建，不从网络输出计算，不通过目标计算泄漏梯度。`event_id` 仅用于配对与存储，process label 仅用于分层/下游；donor ID、beta、mask、干净能量摘要均不能成为模型输入。

## 6. AnInfoNCE：唯一数学定义

### 6.1 距离和正样本索引

对某个空间，两个投影为 `z1,z2: [B,d]`，d=32。按 `[全部 view1, 全部 view2]` 拼接并逐向量归一化：

\[
u_i=\frac{z_i}{\|z_i\|_2},\qquad
D_\Lambda(u_i,u_j)=\sum_{r=1}^d\lambda_r(u_{ir}-u_{jr})^2,
\qquad
s_{ij}=-\frac{D_\Lambda(u_i,u_j)}{2\tau}.
\]

实现用带小稳定常数的 `F.normalize`；等价性测试使用非零投影。`h` 不在此处归一化。采用从 0 开始的索引，正样本索引为 `p(i)=(i+B) mod (2B)`。

\[
\mathcal L_{\mathrm{An}}
=-\frac1{2B}\sum_{i=0}^{2B-1}
\log\frac{\exp(s_{i,p(i)})}{\sum_{j\ne i}\exp(s_{ij})}.
\]

只排除自己，保留正样本和其他事件两视图。相同 process 的不同事件仍然是负样本。不要对不同物理空间互相做对比。

### 6.2 Lambda 参数

\[
a\in\mathbb R^d,\qquad
\lambda_r=d\,\frac{e^{a_r}}{\sum_te^{a_t}},\qquad
\Lambda=\mathrm{diag}(\lambda),\quad \operatorname{Tr}\Lambda=d.
\]

`a=0` 初始化；数学上 lambda 为正、均值为 1。只学习 d 个数，不学习全矩阵；五空间各一组。softmax 的共同平移自由度无需引入新正则器。首版不加 learnable temperature、entropy penalty 或未请求的 eigenvalue constraint。浮点下出现非有限数/权重下溢应诊断，不用隐式截断改变公式并隐瞒。

trace normalization 是项目已确定的版本；不应把它直接称为逐行复现文献的全部假设与理论。它固定度量总体尺度，由固定 tau 控制 logits 尖锐程度。

### 6.3 必须保留 1/2 并验证等价性

对单位向量且 Lambda=I：

\[
-\frac{\|u-v\|^2}{2\tau}
=\frac{u^Tv}{\tau}-\frac1\tau.
\]

共同常数被 softmax 消去，因此和**相同 tau** 的原 cosine InfoNCE 等价。不能省略 1/2 后再声称只改变各向异性。不能把完整平方距离替换成加权点积；候选端 `v^T Lambda v` 一般不是常数。

### 6.4 实现要求

`AnInfoNCE(nn.Module)` 管理 raw_lambda；cosine baseline 使用同一正负样本和 reduction。用 logsumexp/交叉熵计算，不先 exp 再做除法。mask 根据实际 B 在当前 device 构建，不写死训练 batch 大小或 cuda。

可使用：

```python
norm2 = (u.square() * lam).sum(-1)
dist2 = norm2[:, None] + norm2[None, :] - 2 * (u * lam) @ u.T
scores = -dist2 / (2 * tau)
scores = scores.masked_fill(self_mask, -torch.inf)
loss = F.cross_entropy(scores, positive_index)
```

只允许为浮点舍入消除极小负距离；严重负值意味着实现问题。数值核心退出 AMP，FP16/BF16 输入至少升到 FP32；若单元测试使用 FP64 则保留 FP64，不无条件 `.float()` 破坏高精度参考测试。

迁移时 tau 从实际 pairwise 配置显式传入，不能使用旧参考模块的 0.1 默认值覆盖入口中的配置。Lambda 参数属于 objective，必须与模型一起优化和保存；用明确的参数列表或组合 system 管理，不遗漏或重复注册。

## 7. 方案 B 的四组物理目标

设参考网格的有效线性能量为 `E_c`：

\[
S=\sum_cE_c>0,\qquad p_c=E_c/S.
\]

以下目标是**有限接受度、有限分辨率上的探测器能量流摘要**，不是粒子类别标签，不是严格 shower 分离，也不自动满足完整 IRC 安全性。五空间的语义来自这些辅助约束，不来自给头命名。

### 7.1 energy：两维能量尺度目标

\[
S_T^{\mathrm{dep}}=\sum_c\frac{E_c}{\cosh\eta_c},\qquad
t_E=\left[\log(S/E_{\mathrm{ref}}),\log(S_T^{\mathrm{dep}}/E_{\mathrm{ref}})\right].
\]

这是沉积方向的横向能量代理，不是重建 jet 的 H_T、缺失横动量或真实入射粒子能量。目标在 reference 上计算；模型额外观察到的能量摘要在 observed 上计算，二者不得互换。

### 7.2 eta：八个固定区域的能量份额

将固定 eta 范围分成 K_eta=8 个相邻区间 I_a：

\[
P_a^\eta=\sum_{c:\eta_c\in I_a}p_c,
\qquad \sum_aP_a^\eta=1.
\]

目标为 `[B,8]` 的概率分布。32 eta bins 首版每四个合并一组；保存区间边界。线性读出输出 logits，用 `KL(target || softmax(logits))`。不对概率目标作 z-score；零概率 bins 允许存在。

### 7.3 phi：四阶相对方位功率

\[
Q_n=\sum_cp_ce^{in\phi_c},\qquad
A_n=|Q_n|^2
=\sum_{c,d}p_cp_d\cos[n(\phi_c-\phi_d)],
\quad n=1,2,3,4.
\]

目标 `[B,4]`，不是复数的相位，也不是绝对 phi 直方图。理想连续坐标中对共同 phi 旋转不变；离散网格严格测试整 bin 循环平移，不宣称任意重新分 bin 的数值完全相等。谱阶必须低于网格 Nyquist 限制。

### 7.4 local：四尺度平滑两点能量关联

\[
\Delta\phi_{cd}=\operatorname{atan2}(\sin(\phi_c-\phi_d),\cos(\phi_c-\phi_d)),
\quad
\Delta R_{cd}^2=(\eta_c-\eta_d)^2+\Delta\phi_{cd}^2,
\]

\[
M_\ell=\sum_{c,d}p_cp_d\exp[-\Delta R_{cd}^2/(2\ell^2)].
\]

包含 self-pairs；不要改成排除对角线的相关函数或固定块 I2 后仍叫同一个目标。四个 ell 用实际 eta-phi 角坐标单位表示并保存，不是标准化特征单位。

首版工程决策：若用户无已选尺度，准备阶段取 `ell_min=max(delta_eta,delta_phi)`，然后保存 `[ell_min,2*ell_min,4*ell_min,8*ell_min]` 的实际数值。这是跨分辨率可解释的尺度覆盖，不是宣称四个自然物理常数。所有方法对照复用同一数值；以后改变输入分辨率时不得重新按新像素大小偷偷改变目标定义。

固定几何核只构造一次，缓存为 buffer 或按 geometry hash 缓存，不让每个样本重建 O(Ncell²) 核；可预计算/分块计算 targets，不需要为 32×32 首版引入新算法库。full-grid 目标计算要能与稀疏双循环小例子一致。

### 7.5 目标统计与退化情况

energy/phi/local 的各分量用 clean training reference targets 拟合均值和总体标准差；首版准备时用未污染基准训练事件，冻结用于所有视图，不在验证或每个 epoch 重拟合。

\[
t_k^{\mathrm{std}}=(t_k-\mu_k)/\sigma_k.
\]

近常数分量用 scale=1 并记录，不能除以极小数放大舍入噪声。eta 始终保持分布。

回归输出预测标准化目标，不在计算训练误差前强行裁到物理区间。评估再逆变换到物理单位，报告越界率或误差，不用裁剪制造更好结果。

## 8. 五空间模型的明确形状

### 8.1 共享节点主干与 pooling

保留当前三层 GravNet 的骨干设定，不把改变主干宽度、邻居数或换 PTv3 混进此任务。将 forward 拆到 post-conv 的 64-D 节点特征：`node_h: [N_total,64]`。

每个图计算：

\[
r_{\mathrm{mean}},r_{\mathrm{max}},
\quad r_E=\sum_c\frac{E_c^{\mathrm{obs}}}{S^{\mathrm{obs}}}h_c.
\]

三者拼接为 192-D。**不得使用 standardized logE 的 softmax 代替 E/sum(E)**，二者一般产生不同权重。

首版固定再接两个标准化后的**观测**能量摘要 `log(S_obs/E_ref), log(S_T_obs/E_ref)`，得到 `pooled: [B,194]`。这继承方案 B 的具体设计，使总体能量有明确输入通道。对应摘要标准化统计从 clean training pool 拟合并冻结；不使用当前 clean label 或 beta 提供答案。

这两个额外输入也应提供给新的单空间 baseline，避免把信息输入差异误当作多空间收益。无污染情况下能量读出容易，不能把其高 R² 当作复杂物理推理的证明。

### 8.2 分叉点与 heads

在旧 `event_mlp(192→64)` **之前**分叉，不从旧 64-D 向量再复制五份。首版每个 representation head 可用：

```text
Linear(194,64) → LayerNorm(64) → ReLU → Linear(64,64)
```

固定名称和输出顺序：`general, energy, eta, phi, local`。

每个 head 各有独立 projector，沿用旧 projector 的结构级设定，例如 `64→128→32`、相同 LayerNorm/ReLU 规则；不要因参考小模块使用了另一 hidden width 就无声改变所有 baseline。h 不 L2-normalize，z 只在 loss 中 normalize。

四个线性物理读出：

```text
energy: Linear(64,2)
eta:    Linear(64,8)  # raw logits
phi:    Linear(64,4)
local:  Linear(64,4)
```

各物理读出只读取对应 h，不读取拼接后的其他 spaces。general 没有指定物理 readout，不代表它不能包含同样的物理信息。

### 8.3 模型接口

```python
out = model(graph_batch)
# out['h'][space]:    [B,64]
# out['z'][space]:    [B,32]
# out['pred'][task]:  [B,target_dim]

h_dict = model.encode(graph_batch)
h_concat = torch.cat([h_dict[s] for s in SPACE_ORDER], dim=-1)  # [B,320]
```

用 `ModuleDict` 注册层。推理只需一个干净或指定污染程度的事件，不需要两个视图、不需要辅助标签。不得通过把 projector 改为 Identity 来切换输出语义。

每个视图在 PyG 中是独立图；相同原事件 ID 只用于跨视图 loss 配对，不作为共享图 batch id。GravNet 消息不得跨不同事件或两视图传播。

## 9. 联合目标、优化与 checkpoint

### 9.1 联合目标

每个空间都训练对比表示，四个物理空间另有直接施加在 h 上的辅助约束：

\[
\mathcal L_{CL}=\frac15\sum_k\mathcal L^{(k)}_{An},
\]

\[
\mathcal L_{phys}=\frac14\sum_{t\in\{E,\eta,\phi,L\}}
\frac{\ell_t^{(1)}+\ell_t^{(2)}}2,
\qquad
\mathcal L=\mathcal L_{CL}+\gamma\mathcal L_{phys}.
\]

energy/phi/local：标准化目标的 MSE，对 batch 与该任务分量平均；eta：

```python
F.kl_div(F.log_softmax(pred_eta, dim=-1), target_eta,
         reduction='batchmean')
```

因此不因为某个任务输出维数更多就默认更重。首版 gamma=1，固定，不学习辅助权重。保留 gamma=0 作为五头纯对比消融，不密集扫一串相近值。

五个 CL 有相同正样本关系，但物理目标不同，鼓励不同物理侧重；这不是严格信息独立或保证每个空间 64 个有效自由度。不要加正交/互信息惩罚来伪造“物理解耦”。

### 9.2 训练状态

model、readout、objective 的所有 trainable parameters 进入 optimizer 一次。沿用当前 Adam 与 cosine scheduler，显式保存学习率/weight decay/tau/epochs；不把这些常规参数优化作为本轮工作重点。模型参数沿用已有 weight decay；raw_lambda 单独设 weight_decay=0，避免无意中通过其 logits 的 L2 惩罚加入趋向单位度量的先验。两组使用同一基础学习率，首版不单独扫 metric 学习率。

评价时 model 与 objective 都进入相应 eval 状态；不要在训练数据流中全局重设随机状态并留下副作用。可通过 `(seed, epoch, event_key, view_id)` 的稳定哈希派生增强种子；验证种子不含训练 epoch。不要用每次进程变化的 Python `hash()` 作为可复现随机源。

以样本数/anchor 数正确加权 epoch 平均，不把小尾 batch 与满 batch 赋相同权重。不同负样本数量会改变对比任务，应固定验证 batching；不要直接用不同 loss 配方的 raw loss 大小给不同方法排序。

同一运行内 best checkpoint 使用预先固定的 validation total，保存各分项；跨模型结论使用同一独立评估协议。测试不用于调增强、统计量、checkpoint 选择或超参数。

### 9.3 新 checkpoint 的最小内容

```text
schema_version
source_commit / dirty-state note
model config + model_state
objective config + objective_state  # 含所有 raw_lambda
optimizer / scheduler / scaler states
epoch / global_step / best criterion / history
RNG state 或足以重建 epoch-wise 增强的状态
GridSpec + feature stats + observed-summary stats
target definitions + regression target stats
split manifest 标识及可定位副本 / 数据 revision
```

checkpoint 不重复塞入全部事件原始数据；小型 manifest 和配置可在 run 目录保存并附 hash。保证迁移机器后不只剩绝对路径。`resume` 恢复原 split/stats/config 并验证一致性；fresh evaluation 只读状态，不能悄悄训练 scaler。

## 10. 下游评估与有意义的对照

### 10.1 表征导出

对固定 manifest 的 clean 事件输出五个 h、320-D concat、稳定 ID、split 角色、下游 label 和物理目标。辅助 readout 的预测不是 embeddings。存 npz/parquet 等直接格式，避免 notebook 成为唯一可复现流程。

评估器共享新 preprocessing 和 encoder constructor。随机编码器用同样结构、grid、统计量和数据 split，不加载预训练权重，不重复另写一套数据流程。

### 10.2 必须可表达的对照，不要求现在全部长期训练

| 模式 | 表征 | 距离 | 物理辅助任务 |
|---|---|---|---|
| single_cosine | 单个 64-D | cosine InfoNCE | 无 |
| single_anisotropic | 单个 64-D | AnInfoNCE | 无 |
| five_anisotropic_no_aux | 五个 64-D | 各自 AnInfoNCE | 无，gamma=0 |
| five_anisotropic_physics | 五个 64-D | 各自 AnInfoNCE | 方案 B |
| random | 指定同构随机编码器 | 不训练 | 无 |

这些是同一模型/目标模块的少量配置分支，不是五份训练文件。single baseline 沿用相同的修正后 194-D 输入与 split。

后续研究增加一个较宽共享表征、所有物理读出作用于该共享表征的容量对照，以及直接使用物理摘要的分类器。这些是解释多空间收益的重要对照，但不要求在本轮搭好完整大规模实验矩阵。

### 10.3 指标

- 下游类别任务：同一 train/val/test 上的 logistic regression/linear probe，accuracy 与 ROC-AUC；可复用已有其他分类器，测试只在选择完成后使用。
- 物理信息：冻结 h 后，使用统一线性 probe 对各物理任务构成 `space × task` 读出矩阵；原网络自带 readout 的结果另存，不代替跨空间评估。
- 形状与退化：各 h 的方差、奇异值/有效秩、跨头相关性作为诊断；不作为硬性“必须彼此独立”的 loss。
- 共享 320-D concat 的提升不能单独证明解耦；容量与手工摘要对照需要在形成科研结论时加入。
- 本轮没有 overlay 训练时，不报告背景去噪成功。能量辅助任务若直接对应 observed summaries，高精度只是信息可读出的基本检查。

所有 probe/scaler 只在下游训练子集拟合。保留单个 h 和 concat，不默认只报告最好看的那个 head。各方法数据和 preprocessing 一致，不把修复评估泄漏的收益算给 AnInfoNCE。

## 11. 实施顺序：代码里程碑，不是六轮科研训练门槛

**M0 — 快速审阅与依赖映射。** 输出简短旧→新路径表，核对配置/依赖/工作树，保留已有文件。不要消耗整个任务只写审阅长文。

**M1 — pairwise 重构与共用数据基础。** 提取纯函数与 trainer，添加小样本回归检查；在可区分的变更中完成 fixed grid、原始能量 pooling、稳定 IDs 与共享 split、两视图/reference 接口。单空间 cosine 路径仍能运行。

**M2 — AnInfoNCE。** 接通 metric 参数、optimizer、checkpoint 与单空间模式；验证 identity 等价。无需先跑完整对照训练再继续。

**M3 — 五空间方案 B。** 接目标计算与统计、五头模型、联合 objective、h 导出和评估，完成合成端到端 smoke。reference 模块只是可借鉴逻辑，不能以“它已有测试”为由跳过仓库接入。

**M4 — 集成与交付。** 小型真实数据可用时运行有界 smoke；校验 CHTC 包装与模块路径，不发起生产作业。报告实际执行情况、启动命令、纯重构/方法修正/新功能的归类。

无需每完成一步都让用户确认；也无需为每个数学无关的重构改动训练一次。遇到缺少数据/硬件，继续完成不依赖它的工作，但不能将 synthetic passes 等同于真实训练已成功。

## 12. 验收测试：只测试决定实验含义的事项

### 12.1 能量、网格与目标

- 手工 hits 落在同 cell 时能量求和后才 log；排列 hits 不改变输出。
- 不同事件共用真实 edges；验证/测试不改范围。phi ±pi 边界和整 bin 循环平移正确。
- `graph.energy`、观测能量摘要、能量 pooling 均与手工结果一致；标准化节点特征改变不会悄悄改变物理权重。
- eta 目标非负且和为 1；两组等能量背对背 cell 的 phi 功率为 `[0,1,0,1]`；均匀 phi 分布低阶模式为 0。
- 全能量乘 c：两维 log-energy 增加 log(c)，eta/phi/local 不变；M_ell 随 ell 增大单调不减。
- local 核含周期角差和 self-pairs，与小型直接双循环一致。
- reference/observed 几何参数一致；噪声/crop 不污染 reference。不同视图 reference 不盲目共享。
- 目标是 no_grad；修改 target tensors 不改变前向输入或模型输出。统计拟合只访问训练身份。

### 12.2 Loss 与可学习参数

- Lambda=I 与相同 tau cosine loss、对 z 的梯度一致；加权距离对称、对角为 0，正样本索引正确。
- 非均匀 Lambda 的结果与直接平方差求和一致；不能只测试初始化 I（那会漏掉加权点积错误）。
- raw_lambda 有非零梯度并真实更新；trace=d，五组参数独立；model/readout/objective 均在 optimizer 中且没有重复参数。
- B=2、一般 batch、尾 batch 策略、CPU dtype/device 和 AMP 数值核心均按接口工作。FP64 参考测试不被隐式降精度。
- 单独改变某物理 readout 只改变对应辅助项；gamma=0 恢复五头纯对比 objective。

### 12.3 模型与端到端

- 分叉前 pooling shape、五个 `[B,64]`、五个 `[B,32]`、四个物理预测 shape 正确；concat 为 `[B,320]`。
- 所有 heads、projectors、readouts 和共享 backbone 获得梯度；不以 pooled 随机向量的小测试代替实际 GravNet 测试。
- 消息传递不跨图；同事件两个视图仍为独立图。能运行一个干净事件的 encode，无须 targets。
- 保存/加载恢复 h、预测、loss、Lambda 和统计量；epoch 边界 resume 的 CPU 确定性 smoke 与连续执行对照。
- 同一事件的预处理在 train/eval/随机基线一致；manifest 中编码器 train 与最终 test 没有交集。
- 验证不改变下一轮训练增强序列；eval 不更新 feature/target scaler。

### 12.4 文档与命令

提供真实可工作的示例：

```text
python -m src.prepare_pairwise --config <实际配置> ...
python -m src.train_pairwise --config <实际配置> ...
python -m src.train_pairwise --resume <实际checkpoint> ...
python -m src.evaluate_pairwise --run-dir <实际run> ...
python -m pytest -q
```

命令形式是期望接口，不是本说明已经验证仓库存在这些模块。Codex 完成后用具体可执行路径替换占位符并记录测试。无真实数据时写明受阻步骤、已运行 synthetic 范围及启动真实集成所需的最小输入，不伪造下载、测试数目或训练结果。

## 13. 最终交付清单

交付修改后的代码、最小真实配置、数学/集成测试、训练与评估命令、旧→新路径映射、checkpoint schema 和简短实现报告。核心科研决策摘要必须包括：

- pairwise 为基线，教授与 PhD 路线分开；本轮仍是事件级表示，不是粒子分解。
- AnInfoNCE 使用平方距离、32-D 正对角 trace-normalized Lambda、固定 tau。
- 方案 B 使用两视图；五个 h 由不同辅助目标引导，不是 LooC，不宣称严格解耦。
- 四组目标、无污染 reference、观测全局摘要的来源、194-D 分叉点和 linear readout。
- 哪些变化是修复旧方法定义，哪些是纯代码重构；旧实验不可无条件与新结果直接归因比较。

不要把“实现已经完成”和“科研假设已经被实验证实”混为一谈。
