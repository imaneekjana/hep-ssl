# Augmentation 组合实验：51 组 CHTC 部署

活动入口为 `chtc_phase1_steps/build_deployment.py`。它读取当前工作树的基础配置，生成 51 份完整配置、清单、一份源码包及 CPU/GPU submit 模板；不会连接远端或提交作业。数据准备、视图、模型、AnInfoNCE、五个空间和 trainer 均使用原有实现。

## 实验定义

任务顺序：`ggf_ttbar`、`ggf_dihiggs`、`ttbar_dihiggs`。每个任务 17 组，共 51 组。

| 增强数量 | 每任务 | 三任务合计 | 标识后缀 |
|---|---:|---:|---|
| 0 | 1 | 3 | none |
| 3 | 10 | 30 | rex, res, rec, rxs, rxc, rsc, exs, exc, esc, xsc |
| 4 | 5 | 15 | rexs, rexc, resc, rxsc, exsc |
| 5 | 1 | 3 | rexsc |

例如 `ggf_ttbar_rex`。完整 51 行见每个部署目录的 `manifest.csv` / `manifest.json`；配置在 `configs/`。生成器使用 combinations，不展开顺序排列。

固定相对顺序为 **rotate → energy_noise → xyz_noise → shift → crop**。

| 增强 | 字母 | 启用值 | 含义 |
|---|---|---|---|
| rotate | r | rotation=0.3926990817，uniform | ±该角度，单位 rad |
| energy_noise | e | 0.0001 | GeV，现有独立高斯噪声后截断到非负 |
| xyz_noise | x | 5.0 | mm，逐 hit 坐标扰动 |
| shift | s | **shift_std=2.0** | mm，现有共同 XY 位移，z 不变 |
| crop | c | crop_fraction=0.5 | 现有空间框半径定义，不是删除 50% hits/能量 |

禁用项不进入 order，强度为 0。none 的 order=[]、全部强度为 0，但仍训练完整五空间模型和相同 objective，两个无增强视图保留同一事件身份。

| 设置 | 全部 51 组采用值 |
|---|---|
| mode / optimizer | five_anisotropic_physics / 当前 Adam |
| 数据 / grid | 每类 2500，pu0，32×32 |
| epochs / batch_size | 18 / 32 |
| lr / weight_decay | 0.0003 / 0.0001（沿用现有参数组规则） |
| tau / gamma | 0.07 / 1.0 |
| hidden / latent / proj | 16 / 64 / 32 |
| k / space / propagate | 8 / 4 / 16 |
| split_seed / training.seed | 42 / 42 |
| augmentation_seed / validation_seed | 142 / 242 |
| device / amp | cuda / false |

未指定的设置保留基础配置值。当前基础配置与本轮的差异是：order 原为 energy_noise→rotate→crop；shift_std 原为 0；device 原为 auto；rotation 原为 math.pi/8 的更多位精度。部署采用上表值；未启用的强度清零，三个任务分别设置 channels。**基础配置本身不改动**。每份生成配置与实际基础配置的差异都写入 `overrides.json`，包括未来基础配置出现的其他冲突。

## 公平比较和 prepared 成功条件

每个任务只有一份 prepared，其 17 个组合共享完全相同的归档字节、事件、train/val/test manifest、GridSpec、特征/summary/physics-target 统计、定义及 fingerprint。原始准备阶段不施加随机增强，只有视图阶段按运行配置增强；原有 observed/reference 和输入语义保持不变。

验证仍使用该运行的增强和固定 validation 随机流；none 验证也是无增强。**不同增强组合的 train/validation contrastive loss 难度不同，不能仅按这些 loss 排名。跨组合应采用相同 held-out clean 评估协议。**本流程不自动提交下游评估。

准备回执包含完整 metadata、归档 SHA256、配置和源码 SHA256。`check-prepared` 要求三个任务的回执、status、设置及校验和匹配，并要求 HTCondor history 显示作业正常完成、退出码 0。正在运行、held、失败、缺文件、synthetic 均不能通过。通过后保存完成记录；后续每次训练仍核对回执和配置。执行节点还会重新检查实际归档 SHA256、prepared 文件及 fingerprint。

默认缺少数据版本时仍使用现有策略：真实 raw 归档的 SHA256 形成 `local-cache-sha256:...`，不冒充上游 revision。三份 prepared 要来自同一数据版本，已知 raw SHA256 也必须一致。匹配文件依赖精确 `CHANNEL_pu0_calo_hits` 目录及 `train...parquet`；实际 schema 要含 event_id/x/y/z/total_energy。路径歧义、缺列或稳定身份错误都会失败，不会切换 synthetic。

## 1. 本地 VS Code：检查、Commit、Push

在 VS Code 打开本地 hep_ssl 项目，进入 **Source Control**：

1. 逐个查看本轮文件 diff，核对下表参数和脚本；`deployment/` 已被忽略，不应暂存生成包/结果。
2. Stage 本轮源码、测试、文档及旧流程删除项。
3. 填写提交说明，点击 Commit，再点击 Push 或 Sync Changes。
4. 等待 VS Code 显示同步完成。本工具没有替你执行 Git 操作。

当前活动流程取代旧 Mac 上传脚本和“第一轮→继续”的默认拆分。旧 `01_upload_from_mac.sh`、`status.py`、`chtc/pretraining/run_experiment.sh`、`train_sweep.sub` 已删除，避免混用。历史产物和 checkpoint 不删除。

## 2. CHTC VS Code：Pull 当前分支

使用 VS Code Remote SSH 打开 CHTC 已有项目 `/home/kli398/hep_ssl_chtc`，在 Source Control 的菜单选择 **Pull**，确认与本地提交一致。先处理远端未提交修改，不能用 reset 覆盖。若该目录尚未是 Git checkout，先通过 VS Code 的 Clone Repository 使用自己的真实仓库 URL；不要把其他历史目录直接覆盖进去。

路径/资源默认集中在仓库 `chtc_phase1_steps/settings.json`：项目 `/home/kli398/hep_ssl_chtc`，raw `/staging/k/kli398/colliderml-data-pairwise-2500.tar.gz`，容器 `/staging/k/kli398/hep_ssl.sif`。可在生成前修改该文件，或用 `--settings 自己的覆盖文件.json`。所有部署默认值会写进 deployment.json。这里未宣称已经远程验证文件存在。

## 3. 在 CHTC 项目终端生成部署

以下命令在 **CHTC access point** 执行，生成只需 Python 3.9+ 标准库，无需在登录节点安装训练依赖。

```bash
cd /home/kli398/hep_ssl_chtc
DEPLOYMENT=$(python3 chtc_phase1_steps/build_deployment.py --project "$PWD")
cd "$DEPLOYMENT"
pwd
python3 manage.py locations
```

记下这个具体部署目录；后续所有 `manage.py`、condor 命令均在这里执行。目录名格式为 `deployment/YYYYMMDD_HHMMSS_augmentation_随机后缀/`。生成配置没有 Mac 路径。

`locations` 会打印一条带**实际部署 staging 目录**的命令，形如 `ssh ... 'test -r raw && test -r container && mkdir -p staging目录'`。**由你复制执行打印的完整命令**，在 transfer 主机检查两个输入文件并创建输出目录；脚本本身不会执行 SSH。小源码、配置、日志在 /home，大 raw/prepared 经 staging 传输，不从 /staging 提交。

可先预览三份 CPU 提交清单，完全不提交：

```bash
python3 manage.py prepare --dry-run
```

会在新的 attempts 子目录写计划和 TSV，打印实际 condor_submit 命令。不要直接提交这个预览：下面用管理入口正式提交，才能记录 job ID。submit 模板的默认 `ready_*.tsv` 故意不存在，防止绕过检查误提交。

若已知原始 cache 目录不符合自动发现约定，用 VS Code 修改部署中的 `input_paths.json`，填入真实 raw 解包目录以内的相对路径；允许同时列出三类路径。不要填占位符或 Mac 路径。各任务只读取自身两类。

## 4. 准备三份数据

```bash
python3 manage.py prepare
condor_q
```

一次提交三个 CPU 作业，真正的数据读取/准备在执行节点容器中完成。记录输出的 job IDs 和 attempts 路径。初始申请 2 CPU、64GB memory、80GB disk；训练为 1 GPU、2 CPU、64GB memory、60GB disk。这是待实测资源，不是已确认的实际峰值。

查看队列、日志（把 JOB_ID 替换成返回的 Cluster.Proc）：

```bash
condor_tail JOB_ID
condor_q -hold JOB_ID
python3 manage.py status
```

作业完成后：

```bash
python3 manage.py check-prepared
```

须显示三个 `PREPARE_OK`。尚在运行、输出回传未完成、调度器失败或回执不匹配时退出码非零；此时不能提交训练。若需排查，打开对应 `attempts/prepare_.../prepare_PAIR_*.out/.err/.log`、status 和 details。修复路径/schema/资源问题后，对已退出的失败任务可单独重提：

```bash
python3 manage.py prepare --pair ggf_ttbar --retry
```

重试采用新的归档名，避免同名失败内容的缓存；不覆盖以前的回执。held/running 作业不会重复提交，先在 HTCondor 确认或处理原作业。修改科学配置或源码后要重新生成部署。

## 5. 批量提交完整 51 组

```bash
python3 manage.py check-prepared
python3 manage.py train --dry-run
python3 manage.py train
condor_q
```

`train` 自己也会执行完整准备检查；默认一次使用一个 GPU submit 模板和 51 行 TSV，全部直接跑 18 轮。预览命令不会提交 GPU 作业；检查可以保存已验证的准备完成回执。每次正式提交记录 `submission.json` 和 job ID，输出位于独立 `attempts/train_.../`，含每个 run_id 独有的 log/out/err/status/result。传输主机路径与执行节点 basename 分列在 TSV 中。

默认 GPUJobLength=medium；按当前 CHTC 文档这类作业最长 24h。应根据实际耗时选择资源；若需更长，在生成前把 settings 中 gpu_job_length 设为 long。不会为满足限时缩减训练 epochs。

## 可选：先选择一个短跑

这不是必经步骤。只给某一个 run_id 设置停止边界，原 epochs=18 和 scheduler 周期保持不变：

```bash
python3 manage.py train --run-id ggf_ttbar_rex --stop-after-epoch 1
python3 manage.py status --run-id ggf_ttbar_rex
```

短跑成功且作业已退出后，使用 status 显示的**真实结果路径**，例如将下面变量设为那个文件：

```bash
RESUME_ARCHIVE=attempts/实际短跑目录/result_ggf_ttbar_rex.tar.gz
python3 manage.py train --run-id ggf_ttbar_rex --retry --resume-from "$RESUME_ARCHIVE"
python3 manage.py train --remaining
```

`--remaining` 仅提交从未提交过的组合，因此这里是余下 50 组；它不会偷偷重新提交失败或 pending 的实验。各个失败实验按下一节处理。恢复会检查 checkpoint 的配置、增强、prepared fingerprint、优化器/调度器/RNG 状态；仍以18为总轮数。只加载自己信任的 checkpoint。不同 GPU 之间不承诺逐位一致。

## 6. 查看、重跑指定失败实验

```bash
python3 manage.py status
python3 manage.py status --run-id ggf_ttbar_rex
```

查看输出中指定 attempt 目录下的 status JSON、`.err/.out/.log`。`result_RUN_ID.tar.gz` 内根目录为 `RUN_ID/`，含已有 checkpoint、history、config、job_details 和 wrapper.log；即使执行失败也尽量返回诊断与已有结果，退出码非零。CUDA 不可用明确失败，不回退 CPU，不在容器内下载或升级依赖。

修复运行环境/资源问题后，确认原作业已退出，再从头重跑单个实验：

```bash
python3 manage.py train --run-id ggf_ttbar_rex --retry
```

若旧归档有有效 last.pt，可用 `--retry --resume-from 真实归档路径` 恢复。每次重试都是新的 attempt 目录，保留原结果。没有自动抢占恢复平台；强制 kill/eviction 不保证能执行退出归档，因此仅能恢复已经回传的 checkpoint。

## 复用已经存在的 prepared

**前一轮本工具的部署：**先在旧部署运行 `check-prepared` 成功，回到项目根目录生成新部署并指定旧目录；它复用 archive URL 和完全相同的文件 fingerprint，不重新准备。新旧预处理源码摘要和 data/grid/targets 设置必须匹配。

```bash
cd /home/kli398/hep_ssl_chtc
PREVIOUS_DEPLOYMENT=/home/kli398/hep_ssl_chtc/deployment/实际旧目录
DEPLOYMENT=$(python3 chtc_phase1_steps/build_deployment.py --project "$PWD" --reuse-deployment "$PREVIOUS_DEPLOYMENT")
cd "$DEPLOYMENT"
python3 manage.py check-prepared
```

**旧单实验 prepared 包或其他已有 prepared 包：**在新部署中用 `reuse` 提交一个 CPU 校验作业，仅加载并验证已有 prepared（不调用 prepare、不重新打包、不改变归档字节）。给出真实 URL，不能把 raw cache 当 prepared：

```bash
python3 manage.py reuse --pair ggf_ttbar --archive osdf:///chtc/staging/k/kli398/实际旧目录/prepared-data.tar.gz
```

等校验作业成功后，再运行 `prepare`，它跳过成功注册的 pair，只准备其余缺失任务；三个任务均成功后照常 `check-prepared` 和 `train`。已有包的事件数、grid、targets、版本、统计文件内容必须通过当前读取器检查。旧包原始创建源码未知时不会伪造来源；回执明确记录这是当前程序验证的既有产物。校验失败请检查错误，在新的部署中使用正确包或重新准备。

`--reuse-deployment` 不会复制大包，旧 staging 文件在新实验结束前仍需保留。CHTC 指南建议 <1GB 留 /home、1–30GB 用 OSDF、30–100GB 用 file://；本默认面向原有大数据。如果回执实测 prepared 超30GB，先采用 file 协议并核对磁盘/配额再部署。当前管理入口自动管理的是 staging 路径，尚未提供将小于1GB的 prepared 迁移到 /home 的命令；不能直接把 /home 路径传给 `reuse`。若实际数据需要这种存储安排，应先调整部署传输配置后再提交。本地回执检查不读取 staging 大文件；实际字节检查在执行节点进行。

## 验证范围

测试命令和结果见 `TESTING.md`。本地未连接 CHTC、未执行真实提交、未读取真实 ColliderML cache、未在 CHTC 容器或 CUDA 上跑训练。小型 Parquet fixture、synthetic 元数据及模拟 scheduler 状态只用于测试接口/错误分支，不能解释为科研结果。

模板依据官方文件传输、批量 queue 和 GPU 规则；已于 2026-10-01 查阅，实际集群运行仍由用户执行：

- [CHTC staging 与传输](https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata)
- [CHTC GPU 作业](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs)
- [HTCondor condor_submit / queue](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html)
