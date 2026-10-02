# 本轮文件变更清单

本轮只将现有准备/训练能力组织为固定 augmentation sweep。`src/` 和 `configs/pairwise_base.json` 没有改动；模型、AnInfoNCE、物理目标、增强操作、数据划分及 trainer 沿用当前实现。

## 修改

| 文件 | 修改前 → 修改后 |
|---|---|
| chtc_phase1_steps/build_deployment.py | 单实验、第一轮/续跑部署 → 3 个任务 × 17 组合、51 完整配置、清单、共享源码包、CPU/GPU 模板和配置差异记录 |
| chtc_phase1_steps/prepare_data.py | 一个默认 pair 的准备 → 参数化 pair、完整成功回执和指纹；支持只校验复用旧 prepared |
| chtc_phase1_steps/run_prepare.sh | 固定输入输出名 → 每 pair/attempt 独立文件名、失败回执；旧包校验保留原始字节 |
| chtc_phase1_steps/gpu_worker.py | 固定 first/continue 阶段 → 每 run 独立配置、prepared 校验、默认完整18轮、可选单次短跑/恢复 |
| chtc_phase1_steps/run_gpu.sh | 固定阶段归档 → run_id 独立结果/status，失败保留日志与已有结果并返回非零 |
| chtc_phase1_steps/README_ZH.md | Mac 上传和固定首轮流程 → VS Code Commit/Push/Pull、生成、准备、检查、批量提交、重跑及复用说明 |
| chtc_phase1_steps/TESTING.md | 旧部署工具的7项检查 → 本轮147通过、1跳过的实际验证和未验证边界 |

## 新增

| 文件 | 用途 |
|---|---|
| chtc_phase1_steps/manage.py | 准备、成功检查、批量提交、状态、单实验重跑与已有包复用入口 |
| chtc_phase1_steps/deployment_checks.py | 配置/回执/指纹和调度器完成状态检查，阻止准备未就绪时提交训练 |
| chtc_phase1_steps/runtime_utils.py | 共享安全解包、SHA256和JSON写入工具 |
| chtc_phase1_steps/settings.json | 集中设置 CHTC 路径、传输协议和资源 |
| chtc_phase1_steps/CHANGES_ZH.md | 本文件，区分源码变更与生成实验产物 |
| chtc/pretraining/README.md | 指向唯一活动部署流程 |
| tests/test_augmentation_deployment.py | 组合矩阵、清单、gate、复用和提交参数测试 |
| tests/test_prepare_deployment.py | 真实准备接口的小型本地 fixture 和已有包复用测试 |
| tests/test_gpu_deployment.py | GPU wrapper 输入、resume 和缺少 CUDA 的检查 |
| tests/fixtures/experiments_reference.txt | 用户提供的51行上一轮参数，作为回归参考 |

## 删除

- chtc_phase1_steps/01_upload_from_mac.sh
- chtc_phase1_steps/status.py
- chtc/pretraining/run_experiment.sh
- chtc/pretraining/train_sweep.sub

删除上述旧入口是为了保留一套清晰的活动流程，不是修改科学方法的需要。没有删除历史部署、prepared 或训练 checkpoint。

## 生成产物（不作为源码提交）

每轮生成在被 Git 忽略的 `deployment/日期_实验/` 下：

- `configs/` 中51份完整配置；不是51个训练程序。
- `manifest.csv`、`manifest.json`：51组清单和共享 prepared 标识。
- `overrides.json`：逐配置记录相对实际基础配置的差异。
- `hep_ssl-code.tar.gz`：一份当前源码包，所有作业共用。
- `01_prepare.sub`、`01_verify.sub`、`02_train.sub`：准备、旧包校验和批量训练模板。
- 部署元数据、入口脚本副本、prepared registry；实际提交时才产生具体 attempts 清单和作业结果。

真实 prepared 和训练结果需在 CHTC 执行节点运行后才会产生；配置生成及本地 fixture 测试不能代表真实科研运行完成。
