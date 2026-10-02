# Pairwise 第一阶段：运行与实现说明

本阶段已实现固定数据协议上的 AnInfoNCE 与方案 B 五空间。它学习事件表征和可读出的能量流摘要，不是粒子重建、来源分解或严格物理解耦。生产训练、真实 ColliderML 性能与 CUDA/CHTC 验证需要另行执行；合成 smoke 只说明实现链路可运行。

## 入口与配置

在仓库根目录运行：

```bash
python -m src.prepare_pairwise --help
python -m src.train_pairwise --help
python -m src.evaluate_pairwise --help
```

`configs/pairwise_base.json` 统一配置四种训练模式：

- `single_cosine`
- `single_anisotropic`
- `five_anisotropic_no_aux`
- `five_anisotropic_physics`（默认）

随机编码器使用评估参数 `--encoder-mode random`，构造与参考运行相同的模型，完全不加载预训练权重。它仍使用监督拟合的下游分类器。

默认沿用旧 pairwise 的 hidden=16、三层 GravNet、node/h=64、z=32、k=8、space_dim=4、propagate_dim=16；Adam lr=0.0003、weight_decay=0.0001、tau=0.07、batch=32、epochs=18。增强顺序为 energy_noise→rotate→crop，参数分别为 1e-4 GeV、uniform ±π/8、0.5 倍空间标准差的遮挡框。XYZ jitter 和 shift 可显式配置，但默认顺序没有启用它们。

## 可立即运行的有界合成流程

以下命令生成明确标记的 synthetic 事件，不下载 ColliderML，也不是科研结果。`work/phase1_demo` 必须不存在；重复运行请选择新的目录。默认骨干宽度和 k 保留，仅将事件数、网格、batch 和训练 epoch 缩小。

```bash
python - <<'PY'
from pathlib import Path
from src.config import load_config, save_config
root = Path('work/phase1_demo')
root.mkdir(parents=True, exist_ok=False)
config = load_config()
config['data']['events_per_channel'] = 10
config['grid'].update(n_eta=8, n_phi=16)
config['training'].update(epochs=2, batch_size=4, device='cpu')
save_config(config, root / 'config.json')
PY
python -m src.prepare_pairwise --config work/phase1_demo/config.json --synthetic --output-dir work/phase1_demo/prepared
python -m src.train_pairwise --config work/phase1_demo/config.json --prepared work/phase1_demo/prepared --run-dir work/phase1_demo/run --stop-after-epoch 1
python -m src.train_pairwise --resume work/phase1_demo/run/checkpoints/last.pt --prepared work/phase1_demo/prepared
python -m src.evaluate_pairwise --run-dir work/phase1_demo/run --prepared work/phase1_demo/prepared --output-dir work/phase1_demo/pretrained
python -m src.evaluate_pairwise --run-dir work/phase1_demo/run --prepared work/phase1_demo/prepared --encoder-mode random --output-dir work/phase1_demo/random
python -m pytest -q
```

`--stop-after-epoch` 是已完成总 epoch 数的边界，用于短跑与验证恢复，不修改 scheduler 的总周期。恢复读取 checkpoint 的原配置；不接受学习率、epoch 总数、loss 等静默覆盖。

## 本地 ColliderML 数据

准备程序直接读取本地文件，不会触发下载。官方 calo_hits 字段是每事件一行的 `event_id` 和 `x/y/z/total_energy` 列表；位置 mm，沉积能量 GeV。相关定义见 [ColliderML dataset card](https://huggingface.co/datasets/CERN/ColliderML-Release-1)。粒子贡献 truth 列不进入模型。

编辑基础 JSON：明确两个 channels、events_per_channel、已验证的 dataset_revision，以及可选的固定 eta_min/eta_max。不要以 `main` 等可移动引用假装冻结版本。`--input` 必须对应这两个 channel 的本地 Parquet 文件/目录，或下述 NPZ。目录应只含该 channel、pileup 的 calo_hits shards。

真实数据命令形式：

```text
python -m src.prepare_pairwise --config configs/pairwise_base.json --input ggf=/本地/ggf_calo_hits --input ttbar=/本地/ttbar_calo_hits --dataset-revision 已核实的数据版本 --output-dir prepared/ggf_ttbar
python -m src.train_pairwise --config configs/pairwise_base.json --prepared prepared/ggf_ttbar --run-dir "experiments/09_28 training/pretraining/five_physics_seed42"
python -m src.evaluate_pairwise --run-dir "experiments/09_28 training/pretraining/five_physics_seed42" --prepared prepared/ggf_ttbar
```

这段中的真实输入路径和版本必须替换为自己的值；它不声称本机已有这些数据。未指定 eta 接受范围时，只从训练池有限 eta 得到共同对称包络、扩展数值边界并冻结，保存 `range_source=training_envelope`；它不是探测器接受度测量。训练和评估复用同一份准备产物。可用 `--manifest` 复用同一事件集合的已有划分。

NPZ 本地输入约定（`allow_pickle=False`）：

- `hits`: `[sum(N_i),4]`，x/y/z/E；`offsets`: `[M+1]` 整数、以 0 开始。
- `event_id`: `[M]` 稳定身份。
- 可选逐事件 `channel/dataset_revision/pileup/shard`；提供时必须与指定来源一致。
- 原始 ID 若需要 shard 去歧义，Parquet 提供真实 `shard_id`。不能用行号或随机 ID 掩盖重复事件。

准备产物包括 `prepared.json`、`manifest.json`、`events.npz`。所有统计只拟合 clean training identities；事件集合与数据内容通过哈希校验。空接受能量事件在 manifest 标记并计数，不伪造节点。增强导致空 observed/reference 时明确报错，不静默重采样。需要调整时修改明确的增强/接受范围配置并创建新的准备或运行目录。

## 数学与张量接口

- 共享固定网格 `[B,n_phi,n_eta]`，默认 32×32；每 cell 线性能量先相加，再截断阈值，再 `log((E+epsilon)/E_ref)`。
- 图输入 `x[N,3]` 是标准化的 eta/phi/logE；`energy[N]` 是独立观测线性能量；`summary[B,2]` 是标准化 observed logS/logST。targets、类别与事件身份均在图之外。
- 同一事件仅两个独立视图；rotate/shift/xyz_noise 的同一次实际采样作用于 observed 和 reference，energy_noise/crop 只作用于 observed。每视图的目标从各自 reference 计算。
- 原 GravNet 输出 `[N,64]`，mean/max/原始能量加权池化得到 192D，再接两维 observed summaries，194D 处分叉。
- 固定顺序 general/energy/eta/phi/local：每个 h 为 `[B,64]`，z 为 `[B,32]`；concat 为 `[B,320]`。单空间同样使用 194D 输入。
- AnInfoNCE：归一化 z，`lambda=32*softmax(raw_lambda)`，完整加权平方距离 `-Σlambda*(u-v)^2/(2*tau)`；每空间独立 Lambda，trace=32。h 不归一化。
- energy 目标为 logS/logST；eta 为八区域能量份额；phi 为 n=1..4 方位功率；local 为四尺度高斯两点能量关联，包含 self-pairs 和周期角差。
- energy/phi/local 用冻结 training target stats 标准化后 MSE；eta 使用 KL(target||softmax(logits))；总损失为五个 CL 均值 + gamma×四个双视图辅助损失均值，默认 gamma=1。
- Lambda 单独 optimizer group、weight_decay=0；沿用模型 Adam weight decay。不加入新温度、正交约束或隐式正则。

## 保存、恢复与评估

新 checkpoint `schema_version=1` 保存 model/objective/optimizer/scheduler/scaler、epoch/global_step/history/best、Python/NumPy/Torch/CUDA RNG、完整配置和 preprocessing/manifest/hash、Git来源和当前src内容哈希。`last.pt` 用于继续训练，`best.pt` 按 validation total 选择。加载模型与 objective 都是 `strict=True`。旧 checkpoint 不自动迁移。

增强种子由 seed/epoch/stable event key/view id 的稳定哈希派生，验证不依赖训练 epoch。训练 drop_last；验证将最后一个 singleton 合并入前批；epoch 指标按事件数加权。多 worker 是 map-style、非持久 worker，避免事件重复与 epoch 状态滞留。

评估保存：

- `representations.npz`：h、concat、z、稳定 IDs、split、labels、原单位/标准化 targets、网络 readout、简单物理摘要；h 是分类输入。
- 每空间及 concat 的分类 accuracy、ROC-AUC、混淆矩阵与预测。
- 所有 space×task 的统一线性 physics probes；网络自带 readouts 独立报告。
- h/z 方差、奇异值/有效秩、跨头相关性；回归物理范围之外的预测比例，不裁剪来改善指标。
- 所有分类/probe scaler 仅 fit 下游 train；没有另造测试划分。

标准实验目录为 `experiments/<MM_DD training>/{pretraining,classifier}/<run_id>/`。随机编码器、单空间和五空间对照都应复用同一准备产物。

## 旧代码到新代码

| 原职责 | 当前活动实现 |
| --- | --- |
| dataset.py 的读取/投影/嵌套 iterable | data/events.py、projection.py、views.py、prepare_pairwise.py |
| gnn.py 的 GravNet + bottleneck | models/gravnet.py + multispace.py |
| contrastive_learning.py 的 trainer/loss | losses/contrastive.py、multitask.py、training/trainer.py、checkpoint.py |
| pairwise 训练入口 | train_pairwise.py；旧路径为薄 wrapper |
| 两个 pairwise classifier | evaluate_pairwise.py；旧路径为薄 wrapper |

纯结构改动：分离职责、统一入口/配置/状态、移除活动 GravNet 路径不消费的 O(N²) 外部边。原节点编码器与旧权重的小样本回归测试验证逐位一致。

方法修正：固定网格、物理能量池化、按来源身份共享 split、194D 输入、singleton/样本加权统计。新方法：AnInfoNCE、五头与 reference 物理辅助目标。历史结果不可把这些变化一起归因于新 loss。旧非 pairwise、其他模型、notebook 与历史实验保留为非主线参考，未删除。

## 环境与 CHTC

现有容器采用 PyTorch 2.4.1/CUDA12.1、PyG2.6.1、torch-cluster1.6.3；本机实测版本见测试记录。使用与 PyTorch 版本匹配的 torch-cluster wheel，安装说明见 [官方 torch-cluster](https://github.com/rusty1s/pytorch_cluster)。不要为运行脚本无意升级现有训练环境。

本机本次验证使用独立 venv，复用已有 Anaconda graph 环境，没有改动原环境：

```bash
source "/Users/clintli/Documents/Codex/2026-09-16/wo/work/phase1-env/bin/activate"
cd "/Users/clintli/Desktop/hep_ssl"
```

基础依赖：numpy、torch、torch-geometric、torch-cluster、scipy、scikit-learn；本地 Parquet 读取需 polars；测试需 pytest。无需 xgboost 就能导入基本流程；旧分类模块只在显式调用 XGBoost 时导入它。

CHTC 活动 wrapper 现在只接运行名/JSON配置，所有数学参数来自 JSON。准备好：

- `hep_ssl-code.tar.gz` 根目录包含 `src/`、`configs/`。
- `prepared-data.tar.gz` 根目录包含 `prepared/prepared.json`、`prepared/manifest.json`、`prepared/events.npz`。
- 训练 staging 目录中的 `pairwise_base.json` 必须与准备产物的非空 data/grid/targets 设置匹配。
- 评估另带 `result_<run_id>.tar.gz`，根目录包含 `<run_id>/checkpoints/best.pt`。

`.sub` 默认只 queue 1，没有自动 sweep。保留既有容器资源约定；没有自动提交 CHTC 作业。旧16参数 wrapper 命令已由配置入口替代，不维护第二套trainer。wrapper无论成功失败都会打包已有输出和退出码。
