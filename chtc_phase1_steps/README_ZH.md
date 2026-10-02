# CHTC 第一阶段部署：Mac → 数据准备 → GPU 第一轮 → 继续训练

本工具只生成部署文件，不改模型、loss、目标、原始 Dataset 或你已有的实验目录。不应用 patch，不依赖 GitHub 是否已经推送。它使用本机已由 Codex 写入的项目。

## 1. 哪些文件在哪里修改

原始配置：`/Users/clintli/Desktop/hep_ssl/configs/pairwise_base.json`。
上传脚本把它复制到新的 `deployment/phase1_<时间戳>_<随机后缀>/pairwise_chtc.json`，仅把 `training.device` 设置为 `cuda`，并清除该副本的旧 prepared_dir 路径。原配置不改，epochs/batch/增强/loss 等沿用原文件。

`prepared-data.tar.gz` 是程序生成的数据产物，不是需要手写的文件。`prepared.json`、`manifest.json`、`events.npz` 也不手工编辑。

本工具新增三个提交文件：

- `01_prepare.sub`：CPU 数据准备；调用已有 `src.prepare_pairwise.prepare`。
- `02_first_epoch.sub`：GPU 跑正式数据集的第一轮，18轮总调度周期不变。
- `03_continue.sub`：从第一轮 last.pt 继续到原配置设定的总轮数。

这些文件在上传目录中，不覆盖项目旧的 `chtc/pretraining/train_sweep.sub`。本教程不要混用旧提交文件。

## 2. 已有资源与边界

默认项目：`/Users/clintli/Desktop/hep_ssl`。
登录节点：`kli398@ap2002.chtc.wisc.edu`。
传输节点：`kli398@transfer.chtc.wisc.edu`。
原始数据包：`/staging/k/kli398/colliderml-data-pairwise-2500.tar.gz`。
容器：`/staging/k/kli398/hep_ssl.sif`。

脚本会检查这两个远程文件是否存在。它不下载新的数据或自动安装/升级依赖，不假定你的旧缓存就是当前 Hugging Face main。

上传和提交目录位于 `/home/kli398/hep_ssl_chtc/phase1_<时间戳>_<随机后缀>/`。新 prepared 大文件位于独立 `/staging/k/kli398/phase1_<同一标识>/`。训练结果和小日志回到 /home 提交目录。

## 3. 在 Mac 执行

下载并解压整个 `chtc_phase1_steps.zip` 到 Downloads（需要保留包内脚本在一起）。

```bash
bash "$HOME/Downloads/chtc_phase1_steps/01_upload_from_mac.sh"
```

只需要系统可用的 `python3`、`ssh`、`scp`，不用激活训练环境。

脚本自动检查本地新文件、创建配置副本、打包 src/configs/tests/docs，上传并显示实际目录。不上传历史 experiments/notebooks/权重，也不重新上传现有原始数据。

完成后出现 `UPLOAD_COMPLETE`；此时没有自动提交作业。每次运行会创建独立目录，不会重用旧 prepared 文件。

## 4. 登录 CHTC，进入新的提交目录

```bash
ssh kli398@ap2002.chtc.wisc.edu
cd "$HOME/$(cat "$HOME/hep_ssl_chtc/LATEST_PHASE1_DEPLOYMENT.txt")"
pwd
ls
```

latest 文件只是指向最近一次成功上传目录的便利入口。并行管理旧实验时，使用上传输出中那个具体目录，避免选错。

## 5. 提交 CPU 数据准备作业

```bash
condor_submit 01_prepare.sub
condor_q
```

记录 condor_submit 返回的作业号。运行时查看 `condor_tail 作业号.0`；挂起时 `condor_q -hold 作业号.0`。不要重复提交来催促排队。

数据准备在执行节点内：解压既有原始缓存，匹配指定 channel/pileup 的 train calo_hits shards，调用现有 prepare，生成并打包 prepared，再由调度器送到 staging。

自动匹配依赖路径中存在精确目录分量，例如 `ggf_pu0_calo_hits`，以及 `train-...parquet` 文件名。这是支持的缓存约定，不是已经远程核实的目录清单。脚本会检查必需的 Parquet 字段；若没有匹配或存在多个不同目录，不会猜选，而会打印实际目录供核对。

配置默认每类2500，总计5000事件（以实际配置为准）。训练/验证/测试划分、grid和统计都由原有 prepare 生成。数据版本缺失时，程序用实际原始压缩包 SHA256 标记为 `local-cache-sha256:...`；它标识本地数据快照，不冒充上游 Git revision。有已核实 revision 则沿用。

作业结束、输出回传后：

```bash
python3 status.py prepare
```

只有显示 `PREPARE_OK`，再进入下一步。若缺文件，先看队列和 .log，可能还没完成输出传输。不能仅凭 staging 有一个 tar 包判断成功；失败也可能返回诊断占位包。

## 6. 提交第一轮 GPU 作业

```bash
python3 status.py prepare && condor_submit 02_first_epoch.sub
```

先打印容器中实际 Python/PyTorch/CUDA/PyG/torch-cluster 版本和 GPU 名称，做很小的实际 CUDA 前向/反向环境检查，再运行真实数据的第一轮。

这里不是把 `epochs` 改成1。调用的是已有 trainer 的 `stop_after_epoch=1`，总调度周期保留18（或你原配置的总轮数）。第一轮的权重不会浪费。

运行中使用 `condor_tail 作业号.0`；完成后：

```bash
python3 status.py first
```

应显示 `FIRST_EPOCH_OK`、完成轮数1/18、loss、5组Lambda范围、实际环境。返回包 `first_epoch.tar.gz` 中含原始运行目录、config、history、last.pt 和 best.pt。

## 7. 继续训练剩余轮数

```bash
python3 status.py first && condor_submit 03_continue.sub
```

从第一轮 last.pt、原配置、optimizer/scheduler/objective/RNG 状态恢复，重用同一 prepared 指纹。不要把18改成17，不要在恢复时更改loss、batch等数学配置。

第一轮CUDA跨机器续跑使用原有checkpoint机制，但不保证不同GPU型号上逐位一致。本工具没有新增自动抢占重提或抢占恢复逻辑。

`03_continue.sub` 默认 `+GPUJobLength = "medium"`（CHTC当前24小时上限）。参考第一轮耗时；若预计剩余训练接近24小时，在提交前仅把该资源字段改为 `"long"`，不改 epochs。数据准备初始申请64GB内存/80GB磁盘，GPU任务64GB内存/60GB磁盘；这是含压缩输入、解压数据与输出的初始资源申请，尚未实测实际需求。

完成后：

```bash
python3 status.py continue
mkdir -p completed
tar -xzf result_training.tar.gz -C completed
```

应显示 `TRAINING_COMPLETE`、完成轮数18/18。默认结果位于 `completed/five_physics_seed42/`。没有自动运行下游分类评估。

## 8. 下载模型结果（回到 Mac 的另一个 Terminal）

```bash
mkdir -p "$HOME/Downloads/hep_ssl_results"
REMOTE_REL=$(ssh kli398@ap2002.chtc.wisc.edu 'cat ~/hep_ssl_chtc/LATEST_PHASE1_DEPLOYMENT.txt')
scp "kli398@ap2002.chtc.wisc.edu:${REMOTE_REL}/result_training.tar.gz" "$HOME/Downloads/hep_ssl_results/"
```

没有必要把整个 prepared 大包下载到 Mac 才能训练或取得模型。

## 9. 出错时看哪一个文件

- 准备：`prepare_*.err`、`prepare_*.out`、`prepare_status.json`、`prepare_details.json`。
- 第一轮：`first_*.err`、`first_*.out`、`first_status.json`。
- 续跑：`continue_*.err`、`continue_*.out`、`continue_status.json`。

如果容器缺少 polars，准备会在解压大数据之前报错；如果缺 CUDA/PyG 扩展，GPU检查会明确失败，不会退回CPU。此时先修复实际环境，不要改loss凑运行。

如果原始缓存布局不匹配，查看准备日志中真实 paths，在 `input_paths.json` 写相对 raw/ 的精确每类目录，例如 `{ "ggf": "实际ggf目录", "ttbar": "实际ttbar目录" }`。目录只能含对应channel/pileup的训练calo_hits文件；不要把这些中文占位符直接当路径提交。event_id重复也不会用随机ID/行号来掩盖，需检查真实shard身份。

修复数据准备问题后，优先创建新的部署标识、重新准备，避免重复使用已被OSDF缓存的同名失败归档。不要删除原始数据包或旧checkpoint。

## 10. 大文件位置

准备输出默认按完整5000事件的大包安排到个人staging。status会报告实际压缩包字节数。CHTC建议小于1GB的文件留在/home；若prepared小于1GB，可通过transfer主机把它转至提交目录并把两个GPU .sub 的 prepared输入改为本地 `prepared-data.tar.gz`，之后清理不再使用的staging副本。超过30GB则应使用file:///路径并核对资源，不能盲目沿用OSDF模板。

## 验证范围与来源

本包是根据上传的phase1 patch和运行说明新增的部署层；未改动源模型或方法。已完成的本地检查记在 TESTING.md 中，未在用户Mac、SSH账户、真实ColliderML缓存或CHTC调度器上执行，未上传、未提交作业。

项目入口和epoch边界恢复：用户上传 phase1_running.md、hep_ssl_phase1.patch。
CHTC外部部署规则核验（2026-09-29）：
- https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata
- https://chtc.cs.wisc.edu/uw-research-computing/htc-job-file-transfer
- https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs
- https://chtc.cs.wisc.edu/uw-research-computing/htc-monitor-jobs
- https://chtc.cs.wisc.edu/uw-research-computing/transfer-files-computer
