# MiniOneRec 协同增强方案与面试故事

## 1. 问题定义

MiniOneRec 先把商品文本压缩为语义 ID，再让 LLM 生成下一件商品的 SID。它擅长复用商品语义和 LLM 先验，但两个文本相似、消费人群不同的商品，可能得到接近的语义表示；仅靠 SID 序列，局部共现和用户群体偏好不一定能被稳定恢复。

CoLLM 的启发不是“再加一个 ID token”，而是把协同信息视为独立模态：先由传统推荐模型压缩交互图，再用轻量映射层投到 LLM token embedding 空间。论文的两阶段训练也很关键：先让 LLM 学会推荐任务，再固定语言侧能力，单独完成协同空间对齐。

## 2. 不能照搬论文的地方

CoLLM 原任务是候选商品二分类，推理时已知候选 item，因此可以把 user 和 target item embedding 一起注入。MiniOneRec 是开放式下一 SID 生成，推理时目标 item 未知。

如果用目标商品作为 DIN query，离线指标会很好看，但这是把答案送进模型，属于标签泄漏。项目采用因果版 DIN：只读取历史行为，用最后一个已观测商品作为 query，对完整历史做兴趣激活，得到与候选无关的用户协同表示。该表示经 MLP 映射后替换 prompt 中唯一的 `<|collab_user|>` 占位 token。

这次迁移真正重要的收获是：论文复现不是组件名对齐，而是信息可用时点对齐。任何离线特征都要追问“线上推理时是否已经存在”。

## 3. 实现结构

- `scripts/train_din.py`：在相同训练划分上预训练因果 DIN，目标为下一 item ID，验证指标为 HR@10。
- `minionerec/models/collaborative.py`：DIN 编码器、MLP projector、占位 embedding 替换、adapter 保存与加载。
- `minionerec/data/collaborative.py`：文本 prompt 与原始 item ID 历史的双路数据契约，以及专用 collator。
- `minionerec/training/collaborative.py`：第二阶段训练；默认冻结 SFT 后的 Qwen 和 DIN，只训练 projector。
- `evaluate.py --collaborative_adapter ...`：在原有约束 beam search 中启用协同 adapter。
- `minionerec/evaluation/collaborative_metrics.py`：输出整体及历史长度分桶指标，比较收益来源。

推荐训练顺序：

```bash
python -m scripts.train_din \
  --train_file ./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \
  --valid_file ./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv \
  --output_path ./output/office_din.pt

python -m scripts.train_collaborative \
  --base_model ./SFT_Model/final_checkpoint \
  --din_checkpoint ./output/office_din.pt \
  --train_file ./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \
  --eval_file ./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv \
  --output_dir ./output/office_collm

python evaluate.py \
  --base_model ./SFT_Model/final_checkpoint \
  --collaborative_adapter ./output/office_collm \
  --test_data_path ./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv

python -m minionerec.evaluation.collaborative_metrics \
  --result_paths baseline=./result/base.json,collm=./result/collm.json \
  --item_info_file ./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt
```

## 4. 实验矩阵

主表必须统一数据划分、SID、SFT checkpoint、beam 数和随机种子：

| 组别 | 语义侧 | 协同侧 | 训练方式 | HR@10 | NDCG@10 |
|---|---|---|---|---:|---:|
| A | MiniOneRec | 无 | 原 SFT | 待跑 | 待跑 |
| B | MiniOneRec | DIN 分数后融合 | 独立训练 | 待跑 | 待跑 |
| C | MiniOneRec | 随机向量注入 | 只训 MLP | 待跑 | 待跑 |
| D | MiniOneRec | 因果 DIN 注入 | 单阶段联合训练 | 待跑 | 待跑 |
| E | MiniOneRec | 因果 DIN 注入 | 两阶段、冻结 DIN | 待跑 | 待跑 |
| F | MiniOneRec | 因果 DIN 注入 | 两阶段、微调 DIN | 待跑 | 待跑 |

还要按用户历史长度分桶，例如 `1-2`、`3-5`、`6-10`、`>10`。总体提升只能说明方案可能有效；分桶结果才能回答协同信号究竟帮助了谁。预期 warm 用户收益更明显，但这是待验证假设，不应提前写成结论。

至少记录三类工程指标：训练参数量、单 batch 耗时、beam search 推理耗时。Adapter 的价值不仅是效果，还包括复用同一 SFT 主干、快速替换协同编码器和降低实验成本。

## 5. 面试叙事

### 90 秒版本

我复现 MiniOneRec 后发现，它把商品文本压缩成 SID 再做生成，语言语义很强，但相似文本背后的消费群体差异不一定能被 SID 表达。我从 CoLLM 得到启发，准备引入外部协同编码器，把行为表示映射到 LLM 输入空间。

真正困难的不是写 MLP，而是发现论文任务和我的任务不一致。CoLLM 做候选二分类，推理时知道目标商品；MiniOneRec 直接生成下一商品，如果照搬 target-aware DIN，会发生标签泄漏。所以我把 DIN 改成因果版本，只用最后一个已观测行为激活历史兴趣，并通过单个占位 token 注入用户协同向量。

训练上我没有直接端到端混训，而是先固定已有 SFT 能力，预训练 DIN，再冻结两侧主干只学习 projector；同时保留单阶段、随机向量和分数后融合消融。这个项目让我形成了一个很实用的方法：迁移论文时先画清楚训练和推理的信息边界，再讨论模型结构；评估时不仅看总指标，还按冷热用户拆分，确认收益来自哪里。这个方法能避免离线虚高，也让模块可以独立替换和定位问题。

### 追问要点

1. 为什么不直接扩展 user token 词表：用户规模增长会让词表和参数线性膨胀，而且独立 token 很难保留协同表示的低秩结构。
2. 为什么只注入一个向量：先控制变量验证协同模态是否有效，也避免历史长度和 prompt 长度同步增长；后续可比较多兴趣向量。
3. 为什么两阶段：先保护语义推荐和冷启动能力，再学习跨空间对齐；单阶段混训容易让模型过早依赖协同信号。
4. 为什么冻结 DIN：先把“协同表示质量”和“跨模态对齐质量”拆开，实验失败时能定位责任；微调 DIN 作为后续消融。
5. 怎样证明不是参数变多带来的：随机向量注入和等参数 projector 对照。
6. 怎样证明不是简单集成：与 DIN 分数后融合对照，比较表示级融合和决策级融合。

## 6. 简历表述边界

当前简历中的 `HR@10=0.1802、NDCG@10=0.0755` 不能自动归因于本次协同增强，除非它们确实来自组 E/F 且实验日志可追溯。实验完成前，建议把协同条目写成：

> 针对语义 ID 对局部共现建模不足的问题，参考 CoLLM 设计因果 DIN 协同编码器，将仅依赖已观测历史的用户行为表示经 MLP 对齐至 Qwen 输入空间；采用“语义任务适配-协同空间对齐”两阶段训练，并通过随机注入、分数后融合及冷热分桶实验定位增益来源。

拿到稳定结果后，再在句末追加真实提升值和受益最明显的用户分桶。不要把基线复现结果写成改进结果。
