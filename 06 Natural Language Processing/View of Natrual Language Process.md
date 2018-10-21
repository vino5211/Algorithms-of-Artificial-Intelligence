# View of NLP
### Data
+ [相关NLP数据集](https://github.com/Apollo2Mars/View/blob/master/List-Data.md)

### QA相关资源/数据集/论文列表
- https://blog.csdn.net/amds123/article/details/72758936

### 2017年值得读的NLP论文
- Attention is all you need
- Reinforcement Learning for Relation Classification from Noisy Data
- Convolutional Sequence to Sequence Learning
- Zero-Shot **Relation Extraction** via **Reading Comprehension**
- IRGAN:A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
- Neural Relation Extraction with Selective Attention over Instances
- Unsupervised Neural Machine Translation
- Joint Extraction Entities and Relations Based on a Noval Tagging Scheme
- A Structured Self-Attentive Sentence Embedding
- Dialogue Learning with Human-in-the-loop

### ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
- 第一种方法ABCNN0-1是在卷积前进行attention，通过attention矩阵计算出相应句对的attention feature map，然后连同原来的feature map一起输入到卷积层
- 第二种方法ABCNN-2是在池化时进行attention，通过attention对卷积后的表达重新加权，然后再进行池化
- 第三种就是把前两种方法一起用到CNN中

### 语义分析
- https://bosonnlp.com/

### NLPIR
- https://github.com/NLPIR-team/NLPIR

### AllenNLP
### ParlAI
### OpenNMT
### MUSE
- 多语言词向量 Python 库
- 由 Facebook 开源的多语言词向量 Python 库，提供了基于 fastText 实现的多语言词向量和大规模高质量的双语词典，包括无监督和有监督两种。其中有监督方法使用双语词典或相同的字符串，无监督的方法不使用任何并行数据。
- 无监督方法具体可参考 Word Translation without Parallel Data 这篇论文。
- 论文链接：https://www.paperweekly.site/papers/1097
- 项目链接：https://github.com/facebookresearch/MUSE

### skorch
- 兼容 Scikit-Learn 的 PyTorch 神经网络库

### FlashText
- 关键字替换和抽取

### MatchZoo 
- MatchZoo is a toolkit for text matching. It was developed to facilitate the designing, comparing, and sharing of deep text matching models.
- Sockeye: A Toolkit for Neural Machine Translation
- 一个开源的产品级神经机器翻译框架，构建在 MXNet 平台上。
- 论文链接：https://www.paperweekly.site/papers/1374**
- 代码链接：https://github.com/awslabs/sockeye**


### Grammer Model
- Deep AND-OR Grammar Networks for Visual Recognition
	- AOG 的全称叫 AND-OR graph，是一种语法模型（grammer model）。在人工智能的发展历程中，大体有两种解决办法：一种是自底向上，即目前非常流形的深度神经网络方法，另一种方法是自顶向下，语法模型可以认为是一种自顶向下的方法。
	- 把语法模型和深度神经网络模型结合起来，设计的模型同时兼顾特征的 exploration and exploitation（探索和利用），并在网络的深度和宽度上保持平衡；
	- 设计的网络结构，在分类任务和目标检测任务上，都比基于残差结构的方法要好。

### NMT
- Machine Translation Using Semantic Web Technologies: A Survey
    - 本文是一篇综述文章，用知识图谱来解决机器翻译问题。
    - 论文链接：http://www.paperweekly.site/papers/1229



### Short Text Expand
- End-to-end Learning for Short Text Expansion
    - 本文第一次用了 end to end 模型来做 short text expansion 这个 task，方法上用了 memory network 来提升性能，在多个数据集上证明了方法的效果；Short text expansion 对很多问题都有帮助，所以这篇 paper 解决的问题是有意义的。
        - 通过在多个数据集上的实验证明了 model 的可靠性，设计的方法非常直观，很 intuitive。
        - 论文链接：https://www.paperweekly.site/papers/1313
        - 
### Transfer 
- Fast.ai推出NLP最新迁移学习方法「微调语言模型」，可将误差减少超过20%！
    - Fine-tuned Language Models for Text Classification







