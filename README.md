# Seq2Seq-Translation-en-ja-repro

# Seq2Seq 翻译项目 英语-日语

复现自论文 "Sequence to Sequence Learning with Neural Networks" (NIPS 2014) 

项目来源：[Chatbot Tutorial — PyTorch Tutorials 2.11.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html#seq2seq-model)

论文链接：[ Sequence to Sequence Learning with Neural Networks_Ilya Sutskever, Oriol Vinyals, Quoc V. Le](https://arxiv.org/abs/1409.3215)

数据集来源：[Tab-delimited Bilingual Sentence Pairs from the Tatoeba Project (Good for Anki and Similar Flashcard Applications)](https://www.manythings.org/anki/)



本项目中使用 **GRU 模型**代替论文中使用的 LSTM ，作为 LSTM 的一个流行变体，GRU 在保持 LSTM 效果的同时简化了结构 ，它将细胞状态和隐藏状态混合为一个单一的隐藏状态，计算比标准 LSTM 更简单，在减少了参数量、加快了训练速度的情况下，效果与 LSTM 相近。

