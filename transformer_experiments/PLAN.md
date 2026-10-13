# 原始 Transformer/AKT 对照实验（2026-10-13）

本计划在本轮正式训练前固定。目的：验证静态语义题目参数和训练集难度对齐在原始 TextAKT3PL 上是否有效，避免上一轮 GRU 带来的骨干混淆。

## 三组模型

1. `original`：原始本地 TextAKT3PL，2 blocks、8 heads、128 hidden，保留所有原始运算。
2. `static`：仅将 mu / alpha / guess 三个参数头的输入从带位置及历史的语义特征，改为同题固定语义投影之和。原上下文语义特征仍送入原始交互门控及 AKT；不删除 Transformer 层。
3. `aligned`：与 static 完全相同，训练损失额外加入 0.05 × 训练集 Rasch 难度的可靠性加权 Huber。

三组可训练参数量、同种子初始化、3PL 公式、交互门控、位置编码、AKT 模块均相同。不引入 2PL、GRU、新的投影层、额外 anchor 或模型特有优化器设置。固定原始语义向量，无可训练题目/知识点/学生 ID 向量。

原模型的 theta 保留当前题目条件；本次只检验静态题目参数，不能把 theta 描述成完全独立于题目的学生能力。参数仍参与交互门控，不主张完整模型满足经典 IRT 的全局单调性或参数可识别性。

## 预定训练与评估

- warm 和严格 cold 两种既有协议，各 3 组 × 3 种子（42、2026、3407），共 18 次正式训练；原始组也重跑。
- 复用已核验的数据准备函数、训练学生划分和 cold 题目划分，原始输入文件哈希不变。训练/验证/测试学生为 2950/492/1476。cold 训练题 663、验证题 142、测试题 143；所有 cold 题作答不进入训练或评估历史。
- 40 epochs 上限，patience 8，batch 128，L201，评估 stride100，AdamW lr0.0005、weight decay0.0001、cosine T_max40、clip1。训练 BCE 不做标签平滑。
- 只根据验证 AUC 选权重；每个训练运行结束后才评估测试。固定计划不依据中途测试分数改变超参数或取舍种子。
- 训练前运行小规模 CPU 性质测试和 GPU smoke。smoke 不对测试集进行推理或评分，不作为结果。
- 教师仅使用对应协议的训练集记录，沿用上一轮同一份已拟合监督；缓存训练记录 SHA 校验。标准化目标不裁剪，即使极少数教师值超出模型 mu 的 [-3,3] 范围。
- 报告 AUC/ACC/NLL/Brier/ECE15，三种子均值与样本标准差；按种子比较 static-original、aligned-static 和 aligned-original，核对所有测试目标身份并重算指标。
- 所有 18 组结束后，选一组已完成 checkpoint 做加载复现；不重训、不据此改参。

## 可复现命令

在配置好的 WSL Ubuntu-24.04 登录 shell 运行：

```bash
cd /mnt/d/doc/knowledge_trace_graduation_project/dlkt-research
/home/ml/.venvs/pytorch/bin/python -m unittest discover -s tests -p 'test_transformer_models.py' -v
/home/ml/.venvs/pytorch/bin/python -u -m transformer_experiments.run --smoke --epochs 1 --seeds 42 --output runs/transformer-smoke
/home/ml/.venvs/pytorch/bin/python -u -m transformer_experiments.run --output runs/transformer-irt-20261013
```

正式运行前将 `runs/round1-s42/warm-teacher.pt` 和 `cold-teacher.pt` 复制到正式输出目录。代码、输入、协议、教师和 checkpoint 哈希均记录在运行产物中。所有既有 experiments/*.py、model/my_model.py 保持原字节，不破坏上一轮复现。

本协议不能直接与保留的 80.6453% 集成成绩比较：后者最终重训学生数、历史窗口和集成形式不同。warm 测试学生在本项目历史开发中已经被使用，因此不称为从未接触过的外部测试集；cold 仅一次固定题目划分，训练种子复验不能替代多个题目划分。
