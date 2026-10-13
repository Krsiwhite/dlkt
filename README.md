# dlkt：固定语义的知识追踪实验

本分支 `experiment/semantic-irt-alignment` 用于验证两个方向：**题目参数与动态学生状态分离**，以及**训练行为难度与语义难度对齐**。原始题目和知识点向量保持不变，部署模型不使用可训练题目、知识点或学生 ID 表。

- [原始本地源码基线](https://github.com/Krsiwhite/dlkt/tree/baseline/local-original-20261013)
- [模型改动、数据协议和运行说明](experiments/README.md)
- [实验方案及追加复验记录](experiments/PLAN.md)
- [完整测试结果](reports/round1.md)
- [结果解读与论文表述建议](reports/READOUT.md)
- [难度参数的独立评价](reports/difficulty-validity.md)

实验入口为 `python -m experiments.run`。根目录的 `main.py` 保留原始实现，其历史训练流程曾用测试集选轮；本分支的正式结果使用独立验证集选择权重，再进行最终测试。

在仓库目录运行：

```sh
python -m unittest discover -s tests -v
python -u -m experiments.run --output runs/round1-s42
```

需要 Python 3.10 以上、NumPy 和与硬件匹配的 PyTorch；本次实跑环境及详细配置保存在本地运行记录中。多种子复验、报告汇总和已有权重推理复现命令见实验说明。

结果中的 warm 表示常规学生留出，cold 表示题目也完全留出且历史排除全部冷题作答。两者任务不同。此前保留的 80.6453% 三模型集成是另一套模型和评测配置，不能直接用两套分数相减来判断本方法涨跌。

训练权重和逐条预测保存在本地 `runs/`，不提交到 GitHub。公开报告只包含汇总指标、协议、配置和哈希。原始本地 `dlkt` 以及此前保留的模型目录均未修改，见 [文件保留校验](reports/preservation.json)。

新快照已去除源码中的明文 API 密钥；预处理通过环境变量读取。旧提交历史未重写，曾公开出现过的凭据需要在对应服务商处更换。
