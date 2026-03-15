# Seq2Seq-Translation-en-ja

## Seq2Seq 翻译项目：英语-日语

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC--BY--2.0-green.svg)](https://creativecommons.org/licenses/by/2.0/)

一个基于 Seq2Seq 架构的英语-日语神经机器翻译项目，复现自论文 "Sequence to Sequence Learning with Neural Networks" (NIPS 2014)。

## 📋 项目简介

本项目实现了一个端到端的神经机器翻译系统，能够将英语句子翻译成日语。项目采用编码器-解码器（Encoder-Decoder）架构，结合注意力机制（Bahdanau Attention）和 GRU 循环神经网络，实现了高质量的翻译效果。

### 主要特性

- ✅ 基于 Seq2Seq 架构的神经机器翻译
- ✅ 使用 GRU 替代 LSTM，在保持效果的同时提升训练速度
- ✅ 集成 Bahdanau 注意力机制，提升长句翻译质量
- ✅ 完整的数据预处理流程，支持日语分词
- ✅ 提供训练好的模型文件，可直接用于推理

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.5.1
- **编程语言**: Python 3.12
- **核心模型**: GRU (Gated Recurrent Unit)
- **注意力机制**: Bahdanau Attention
- **日语分词**: Janome
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib

## 📦 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (可选，用于 GPU 加速)

### 主要依赖

```
torch >= 2.0
numpy >= 1.20
matplotlib >= 3.5
janome >= 0.4
tqdm >= 4.60
```

## 📥 安装

1. 克隆项目到本地：
```bash
git clone <repository-url>
cd Project1_Seq2Seq-Translation-en-ja
```

2. 安装依赖包：
```bash
pip install torch numpy matplotlib janome tqdm
```

## 📊 数据集

本项目使用的数据集来自 [Tatoeba Project](https://www.manythings.org/anki/)，包含英语-日语双语平行语料。

- **数据集文件**: `data_en_ja/jpn.txt`
- **数据来源**: Tab-delimited Bilingual Sentence Pairs from the Tatoeba Project
- **数据格式**: 制表符分隔的英语-日语句子对
- **数据规模**: 约 117,000 条句子对

数据集已按照训练集和测试集进行划分：
- 训练集：约 105,512 条
- 测试集：约 11,510 条

## 🚀 使用方法

### 训练模型

1. 打开 Jupyter Notebook：
```bash
jupyter notebook seq2seq_en_ja.ipynb
```

2. 按照 notebook 中的步骤依次执行：
   - 数据加载与预处理
   - 构建词表
   - 模型定义
   - 模型训练
   - 模型评估

## 📁 项目结构

```
Project1_Seq2Seq-Translation-en-ja/
├── README.md                 # 项目说明文档
├── seq2seq_en_ja.ipynb       # 主程序（Jupyter Notebook）
├── 5-自然语言处理.md          # 项目笔记文档
├── checkpoints/              # 模型检查点文件夹
│   ├── best.ckpt            # 本地训练的最佳模型
│   └── best_cloud.ckpt      # 云平台训练的最佳模型
└── data_en_ja/              # 数据集文件夹
    ├── jpn.txt             # 英语-日语平行语料
    └── _about.txt          # 数据集说明文件
```

## 🧠 模型架构

### 编码器（Encoder）
- 使用双向 GRU 编码源语言（英语）序列
- 输出编码后的隐藏状态序列

### 解码器（Decoder）
- 使用单向 GRU 解码目标语言（日语）序列
- 集成 Bahdanau 注意力机制，动态关注源序列的不同部分

### 模型特点
本项目使用 **GRU 模型**代替论文中使用的 LSTM。作为 LSTM 的一个流行变体，GRU 在保持 LSTM 效果的同时简化了结构，它将细胞状态和隐藏状态混合为一个单一的隐藏状态，计算比标准 LSTM 更简单，在减少了参数量、加快了训练速度的情况下，效果与 LSTM 相近。

## 📊 项目实现与原论文对比

| **对比维度** | **原论文 (Sutskever et al., 2014)**            | **本项目**                                        |
| ------------ | ---------------------------------------------- | ------------------------------------------------- |
| **核心架构** | 4层深层 LSTM 网络                              | 单层 GRU 网络                                     |
| **信息传递** | 无注意力机制，依靠单个固定维度向量传递所有信息 | 引入 Bahdanau 注意力机制动态传递上下文            |
| **关键技巧** | 翻转源句子词序                                 | 正常源端语序                                      |
| **参数规模** | 约 3.84 亿 (384M)                              | 约 3215 万 (~32.1M)                               |
| **解码策略** | 集束搜索 (Beam Search, 宽度 1-12)              | 贪心搜索 (Greedy Search, `argmax` 取最大值)       |
| **优化算法** | 随机梯度下降 (SGD) 无动量，固定学习率后减半    | Adam 优化器 (初始学习率 0.001)                    |
| **评估任务** | WMT'14 英法翻译 (English to French)            | 日英翻译 (Japanese to English)                    |
| **评估数据** | 测试集 BLEU 分数达 **34.81**                   | 测试集平均 **BLEU-1: 56.91%**，**BLEU-4: 21.27%** |

以下是基于上述差异的具体分析，重点探讨架构与策略不同带来的影响，以及对评估数据的解读：

### 1. 评估数据与翻译任务的差异分析

- **原论文**：在同语系、语法结构高度相似的英法翻译任务 (WMT'14) 上取得了 34.81 的测试集 BLEU 分数。
- **本项目**：在跨语系、语序完全倒置（从 SOV 到 SVO）的日英翻译任务上，取得了 BLEU-1: 56.91%，BLEU-4: 21.27% 的测试集平均分数。
- **分析**：绝对数值的差异主要源于任务本身的复杂度。日英翻译的对齐难度远高于英法翻译。本项目能取得 56.91% 的词汇精确命中率，证明模型已经成功跨越了语系障碍，建立起扎实的基础语义映射能力。BLEU-4 偏低则主要受限于解码策略与模型规模。

### 2. 架构深度与参数量的取舍

- **原论文**：使用 4 层深层 LSTM 网络，参数量高达约 3.84 亿 (384M)。没有使用注意力机制，完全依赖将整个输入序列压缩为一个固定维度的向量来传递信息。
- **本项目**：使用单层 GRU 网络，参数量仅约 3215 万 (~32.1M)，但引入了 Bahdanau 注意力机制。
- **分析**：原论文属于早期的纯 Encoder-Decoder 架构，完全依赖庞大的参数量和深层网络来强行记忆并编码整个句子的语义。本项目通过引入注意力机制，巧妙地绕过了固长向量带来的"信息瓶颈"问题。由于 Decoder 在生成每个词时可以动态"回头看"源句子的不同部分，因此即使只有单层网络和不到十分之一的参数量，模型也能精准捕捉复杂的对齐关系。

### 3. 长程依赖的解决路径

- **原论文**：在训练和测试时，将源语言句子的单词顺序进行了反转（输入逆序）。
- **本项目**：按照正常语序处理输入的日语句子，并未进行反转。
- **分析**：原论文引入"输入逆序"的技巧是为了人为缩短源语言开头与目标语言开头之间的"时间滞后"，让反向传播更容易建立连接，从而极大降低了长句子的优化难度。在本项目中，Bahdanau 注意力机制直接打破了这种时序距离的限制。无论原句多长，注意力机制都能通过全局打分找到当前需要关注的源词，因此无需逆序输入也能很好地处理日英语序倒置问题。

### 4. 解码策略对评估分数的限制

- **原论文**：在生成最终翻译时使用了自左向右的集束搜索（Beam Search，默认保留少量候选假设）。
- **本项目**：推理函数（`infer`）仅使用了贪心搜索（Greedy Search，每次通过 `argmax` 获取局部最大值）。
- **分析**：贪心搜索每次只选择局部概率最大的单词，这在翻译长句时极易陷入局部最优。一旦中间某个词预测偏差，后续生成的短语结构就会完全崩坏。这也是本项目中 3-gram 和 4-gram 命中率骤降，导致 BLEU-4（21.27%）远低于 BLEU-1（56.91%）的核心原因。原论文采用的集束搜索能从全局视角找出整体概率最大的句子组合，这对提升最终的 BLEU 评估分数至关重要。

### 5. 优化算法与收敛情况

- **原论文**：使用了无动量的随机梯度下降（SGD），并在训练后期手动将学习率逐渐减半。
- **本项目**：使用了自适应的 Adam 优化器（初始学习率 0.001），并配置了早停机制。
- **分析**：对于本项目中相对较小的数据集与模型规模，Adam 能够帮助模型非常快速地收敛（代码在第 4 个 Epoch 就触发了早停）。但也因为收敛过快，模型可能陷入了次优解。原论文的 SGD 配合学习率衰减策略虽然训练耗时极长，但往往能更稳定地找到泛化能力更强的全局最优解。

## 📚 参考文献

- **论文**: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
  - Ilya Sutskever, Oriol Vinyals, Quoc V. Le (NIPS 2014)

- **项目来源**: [Chatbot Tutorial — PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html#seq2seq-model)

- **数据集**: [Tab-delimited Bilingual Sentence Pairs from the Tatoeba Project](https://www.manythings.org/anki/)

## 📝 许可证

数据集遵循 [Creative Commons Attribution 2.0 License](https://creativecommons.org/licenses/by/2.0/)。

## 🙏 致谢

- Tatoeba Project 提供高质量的双语平行语料
- PyTorch 团队提供优秀的深度学习框架和教程

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
