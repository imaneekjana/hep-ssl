# 本次小型部署包装测试

实际执行：

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider tests/test_submission.py
14 passed in 1.26s
```

Python语法编译与 `bash -n run_classifier.sh` 均通过。

覆盖：51组配对、三份prepared URL复用、固定best.pt/冻结评价入口、GPU/CPU模板、原始输入不变、每组状态及18轮history检查、错误指纹/缺checkpoint/非有限loss/最近失败attempt拒绝、失败日志归档。

fixture中的checkpoint字节是明确标记的占位文件，仅用于检查归档成员是否存在；没有加载它们为Torch模型，没有据此宣称预训练或classifier训练成功。没有伪造真实研究结果。

此环境未安装PyG、torch-cluster、Polars或HTCondor，也无可用CUDA。因此未在这里执行真实编码器特征提取、完整classifier作业或HTCondor解析；没有访问CHTC真实结果。核心训练与评价代码不改动。用户提交前的一次 `condor_submit -dry-run` 使用真实CHTC parser检查提交模板。
