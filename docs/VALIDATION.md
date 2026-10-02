# 第一阶段实际验证记录

工作树：`/Users/clintli/Desktop/hep_ssl`；基线分支 `Kunhe-Li`，提交 `994cd472b19b7fde8d7c295cec23eb696b08323d`。原有 `.DS_Store` 修改和未跟踪的 `AGENTS.md` 保留，没有提交或推送。

## 本机环境

独立测试 Python：`/Users/clintli/Documents/Codex/2026-09-16/wo/work/phase1-env/bin/python`。它复用 `/opt/anaconda3/envs/graph` 的系统包，只在独立 venv 中补齐依赖，没有升级原环境。

- macOS arm64；Python 3.11.14
- PyTorch 2.8.0、PyG 2.7.0、torch-cluster 1.6.3
- NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0
- pytest 9.1.1、Polars 1.44.2
- CUDA 不可用

## 最终测试

在项目根目录实际执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/clintli/Documents/Codex/2026-09-16/wo/work/phase1-env/bin/python -B -m pytest -q -p no:cacheprovider
```

结果：**67 passed, 1 skipped，15.46 秒**。唯一跳过项为 CUDA AMP 检查。两条非失败 warning 分别来自 PyG distributed 弃用提示，以及测试断言将 requires_grad tensor 转成标量；不影响测试通过。

覆盖内容：

- 固定网格、线性能量求和后 log、phi wrap/整数 bin 旋转、cutoff、能量损失统计与空事件。
- Scheme B 目标、周期 local kernel/self-pairs、能量缩放不变性、目标标准化。
- 稳定事件身份、分层持久化 split、仅训练身份拟合统计、本地真实格式 Parquet/NPZ fixture、有界读取、metadata/data hash 检查和已有文件保护。
- 两视图的共享几何、独立 reference 能量、目标不作为输入、逐事件哈希随机流。
- AnInfoNCE 在 Lambda=I 时与 cosine 的 loss/梯度等价；非均匀距离、实际 Lambda 更新、state roundtrip、FP64/CPU autocast。
- 真正 PyG GravNet 前向/反向、所有 heads/readouts/metric 梯度、跨图隔离、单个干净事件 encode、小图 N=1/2/7；原节点主干与旧实现逐位回归。
- 四种训练模式、optimizer 参数覆盖及 metric weight_decay=0、singleton 合并、样本加权、严格 checkpoint 和配置匹配。
- 两轮连续训练与第一轮后恢复：模型/metric 权重、history、scheduler 状态逐位相同。
- num_workers=0 与 2 的事件身份、视图和目标一致，事件不重复。
- 预训练/随机编码器共用构造与 clean 数据；每空间+concat 分类/物理 probe；改变 holdout 不影响拟合参数；网络 readout 独立导出。

多 worker 测试最初在沙箱内因 `torch_shm_manager: Operation not permitted` 失败；申请运行权限后，同一测试通过。最终完整测试集在允许该共享内存管理器的环境中执行，未把此失败隐去或误记为代码已通过。

另外：活动变更 Python 文件 AST 解析、三个新入口和兼容入口 `--help`、`bash -n` 与 `git diff --check` 通过。全目录额外检查曾碰到原有 `.ipynb_checkpoints/cnn_transformer-checkpoint.py` 的语法错误；它属于未使用的历史 notebook 快照，没有为此修改历史文件。

## 完整合成与 wrapper smoke

完成两组有界 smoke：

1. 24 个合成事件、8×16 网格、实际 GravNet 训练一轮；预训练/随机编码器完整评估，共用 IDs/split/labels/targets，concat=[24,320]。
2. 20 个合成事件、默认 **32×32 网格与 hidden=16/k=8 原骨干宽度**、训练两轮；按实际归档布局执行三个 shell wrapper。

第二组实际执行命令：

```bash
bash run_experiment.sh smoke_phase1 pairwise_base.json
bash run_classifier.sh smoke_phase1
bash run_random_classifier.sh smoke_phase1
```

三个命令退出码均为 0；生成三份结果归档；pretrained/random 的事件、split、labels、四组目标完全一致，h_concat 均为 [20,320]，不同初始化来源确实得到不同表征。运行目录为 `/Users/clintli/Documents/Codex/2026-09-16/wo/work/phase1-wrapper-smoke-j3a8jbec`，其 `verification.json` 保存逐条命令、stdout/stderr 与归档列表。

这些 wrapper 在本地测试环境运行，**不是**在 CHTC 容器或调度器内执行。没有提交集群作业，也没有长期训练或超参数 sweep。合成分类数值不作为科研性能结果报告。

## 尚未验证的条件

- 项目目录没有可用的真实 ColliderML calo_hits 数据，因此未对真实事件执行训练/评估。启动所需最小输入为两个 channel 的本地 calo_hits Parquet/NPZ、可核实的数据 revision，及用户选择的配置。
- 本机无 CUDA，因此 CUDA AMP、CUDA 小图、GPU 确定性与原 PyTorch2.4.1/CUDA12.1 CHTC 容器未实际测试。
- 未开展扰动强度标定、多种子正式实验或证明五空间解耦；这些是后续科研实验，不是当前 smoke 的结论。
