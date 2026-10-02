# 本轮 augmentation sweep 验证记录

日期：2026-10-01。对象为当前 hep_ssl 工作树中的部署层改动。

## 已实际执行的完整测试

在项目根目录执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/clintli/Documents/Codex/2026-09-16/wo/work/phase1-env/bin/python -B -m pytest -q -p no:cacheprovider
```

该绝对解释器路径只记录本次 Mac 验证环境，不是 CHTC 依赖。其他机器使用其已有、依赖齐全的项目 Python 环境执行同样的 `python -B -m pytest -q -p no:cacheprovider`。

最终结果：

```text
147 passed, 1 skipped, 2 warnings in 23.48s
```

首轮在受限沙箱中运行时，多进程 DataLoader 测试因 `torch_shm_manager: Operation not permitted` 失败。随后在沙箱外用完全相同的测试命令重跑，得到上述结果；没有删除、改写或跳过这项测试。唯一 skip 是已有 CUDA 测试，因为本机没有 CUDA。两条 warning 分别来自 PyG distributed 弃用提示和原有测试的 tensor 转 scalar 提示。

## 覆盖范围

| 测试部分 | 通过数 | 验证内容 |
|---|---:|---|
| 原有数学、模型、数据、训练及评估回归 | 67 | 使用真实项目实现，包括 GravNet 和多 worker 数据一致性；另 1 项 CUDA skip |
| test_augmentation_deployment.py | 59 | 51 配置与参考文件逐项比较、3任务×17组合、固定顺序、shift=2.0、none、设置公平性、现有配置解析、manifest、归档、shell、准备门槛、复用、dry-run、提交参数覆盖 |
| test_prepare_deployment.py | 9 | 小型 Parquet 准备 wrapper 成功/缺 schema/不安全归档、旧 prepared 校验与失败后原包字节不变 |
| test_gpu_deployment.py | 12 | prepared/archive/receipt/pair 检查、resume 配置约束及实际无 CUDA 时明确拒绝 |

新增测试单独执行命令（使用同一项目 Python 环境）：

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_augmentation_deployment.py
python -B -m pytest -q -p no:cacheprovider tests/test_prepare_deployment.py
python -B -m pytest -q -p no:cacheprovider tests/test_gpu_deployment.py
```

这些命令分别实际得到 59、9、12 项通过。生成器测试真实写出完整配置和归档；批量清单测试检查 3 个 CPU / 51 个 GPU / 单个短跑的展开关系，未向调度器提交。

收尾将测试记录和变更清单加入随包文档后，重新执行部署测试文件：`59 passed in 3.50s`。同时重新执行两个活动 wrapper 的 `bash -n`、六个部署 Python 模块的 AST 语法检查以及 `git diff --check`，均通过。未因文档调整重复整套科研回归测试。

数据 wrapper 测试实际读取制造的小型 Parquet fixture 并调用原 prepare 接口；这些测试记录不是 ColliderML 科研数据。GPU receipt/resume 测试中的人工元数据和最小 checkpoint 只用于接口验证，没有拿占位模型权重去执行训练，也没有伪造 CUDA 成功或替换 GravNet。

## 本机环境

- macOS 26.6.2，Apple arm64；Python 3.11.14。
- PyTorch 2.8.0，PyG 2.7.0，torch-cluster 1.6.3。
- NumPy 2.3.5，Polars 1.44.2，pytest 9.1.1。
- `torch.cuda.is_available()` 为 false。
- `condor_submit` 在本机不可用。

## 检查与边界

已执行 `bash -n` 检查活动 shell wrapper；`git diff --check` 无错误；`git diff --name-only -- src configs` 为空，确认核心源码和基础配置保持原样。

尚未在本地 HTCondor 解析 submit 文件，未连接 CHTC、未提交真实 CPU/GPU 作业、未检查远端 raw/container 是否存在，未验证真实 ColliderML cache 或 CHTC 容器。生成正式部署包不等于准备数据或完成 51 次训练。

CHTC 上的实际步骤见 [README_ZH.md](README_ZH.md)。提交前 gate 要求三个真实 prepared 任务成功；任何缺失、失败或 synthetic 都不能被当作研究准备完成。
