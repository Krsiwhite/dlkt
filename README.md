# dlkt：固定语义的知识追踪实验

当前分支 `experiment/transformer-semantic-irt` 在**原始 Transformer/AKT** 上验证两个方向：让题目参数只由当前题目语义产生，以及使用训练集作答拟合的 Rasch 难度监督语义难度。原始题干和知识点向量保持固定，部署模型不使用可训练题目、知识点或学生 ID 表。

18 组正式实验已完成：两个协议、三组模型、三个训练种子。静态语义题目参数相对原始 Transformer 的 AUC 平均提升 warm **0.2056 个百分点**、cold **0.2790 个百分点**；额外难度对齐没有提高平均 AUC，但改善了难度一致性及 cold 的平均 NLL、Brier、ECE15。三个种子的证据仍有限，详细结果和边界见报告。

- [原始本地源码基线](https://github.com/Krsiwhite/dlkt/tree/baseline/local-original-20261013)
- [当前 Transformer 实验方案与训练配置](transformer_experiments/PLAN.md)
- [当前 18 组完整测试结果](reports/transformer-irt.md)
- [当前模型改动与结果解读](reports/TRANSFORMER_READOUT.md)
- [当前难度参数的独立评价](reports/transformer-difficulty.md)
- [当前保存权重的推理复现记录](reports/transformer-reproduction-warm-aligned-s42.json)
- [从 GitHub 新 clone 后复现静态版权重](reports/transformer-reproduction-fresh-clone-static-s2026.json)

当前实验入口为 `python -m transformer_experiments.run`。根目录的 `main.py` 保留原始实现，其历史训练流程曾用测试集选轮；本轮正式结果只使用验证集 AUC 选择权重，再进行最终测试。三组都保留原始 AKT、交互门控和 3PL 输出，参数量均为 2,591,164；能力输出仍依赖当前题目，不能把本实验称为完整心理测量解耦。

在仓库目录检查已有结果和权重，无需重新训练：

```sh
python tools/summarize_transformer.py --root runs/transformer-irt-20261013 --output reports/transformer-irt
python tools/reproduce_transformer.py --run-dir runs/transformer-irt-20261013/warm-aligned-s42 --data-root dataset/NIPS2020 --device cuda --output runs/transformer-irt-20261013/warm-aligned-s42/reproduction-check-2.json
```

复现输出必须使用新文件名。已完成的一次 GPU 核验中，415,735 条测试预测逐位一致，所有指标一致。需要 Python 3.10 以上、NumPy 和与硬件匹配的 PyTorch；本次实跑环境及详细配置保存在本地运行记录中。完整训练命令见实验方案，读取报告和检查保存权重不需要重新训练。

上一轮 GRU 实验作为历史记录保留，其改进组使用了不同于原始 AKT 的骨干，不能替代本轮 Transformer 对照：

- [历史 GRU 实验实现与运行说明](experiments/README.md)
- [历史 GRU 实验方案](experiments/PLAN.md)
- [历史 GRU 结果](reports/round1.md)
- [历史 GRU 结果解读](reports/READOUT.md)
- [历史 GRU 难度评价](reports/difficulty-validity.md)

结果中的 warm 表示常规学生留出，cold 表示题目也完全留出且历史排除全部冷题作答。两者任务不同。此前保留的 80.6453% 三模型集成是另一套模型和评测配置，不能直接用两套分数相减来判断本方法涨跌。

训练权重和逐条预测保存在本地 `runs/`，不提交到 GitHub。公开报告只包含汇总指标、协议、配置和哈希。原始本地 `dlkt` 以及此前保留的模型目录均未修改，见 [本轮结束后的文件保留校验](reports/transformer-preservation.json)。

新快照已去除源码中的明文 API 密钥；预处理通过环境变量读取。旧提交历史未重写，曾公开出现过的凭据需要在对应服务商处更换。
