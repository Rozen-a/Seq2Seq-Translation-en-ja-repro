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
