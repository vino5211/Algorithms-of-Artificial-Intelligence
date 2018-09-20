# View of Sentiment Analysis

### Benchmarking Multimodal Sentiment Analysis
- 多模态情感分析目前还有很多难点，该文提出了一个基于 CNN 的多模态融合框架，融合表情，语音，文本等信息做情感分析，情绪识别。
- 论文链接：https://www.paperweekly.site/papers/1306

### Aspect Level Sentiment Classification with Deep Memory Network
- 《Aspect Level Sentiment Classification with Deep Memory Network》阅读笔记

### Attention-based LSTM for Aspect-level Sentiment Classification
- 《Attention-based LSTM for Aspect-level Sentiment Classification》阅读笔记

### Learning Sentiment Memories for Sentiment Modification without Parallel
+ Sentiment Modification : 将某种情感极性的文本转移到另一种文本
+ 由attention weight 做指示获得情感词, 得到 neutralized context(中性的文本)
+ 根据情感词构建sentiment momory
+ 通过该memory对Seq2Seq中的Decoder的initial state 进行初始化, 帮助其生成另一种极性的文本

### 