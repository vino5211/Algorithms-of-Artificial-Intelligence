# View of Graph Model

https://speechlab.sjtu.edu.cn/pages/sz128/homepage/year/08/21/SLU-review-introduction/

- 在口语理解的语义槽填充（基于序列标注）任务上，循环神经网络首先取得突破。Yao 和 Mesnil同时将单向RNN应用于语义槽填充任务，并在ATIS评测集合上取得了显著性超越CRF模型的效果(Yao et al. 2013; Mesnil et al. 2013)
- 卷积神经网络（Convolutional Neural Networks, CNN）也经常被应用到序列标注任务中(Xu et al. 2013; Vu 2016)，因为卷积神经网络也可以处理变长的输入序列
- 除了与传统CRF模型的结合，基于序列到序列（sequence-to-sequence）的编码-解码（encoder-decoder）模型(Bahdanau et al. 2014)也被应用到口语理解中来(Simonnet et al. 2015)。这类模型的encoder和decoder分别是一个循环神经网络，encoder对输入序列进行编码（特征提取），decoder根据encoder的信息进行输出序列的预测。其核心在于decoder中tt时刻的预测会利用到t−1时刻的预测结果作为输入
- 受encoder-decoder模型的启发，Kurata等人提出了编码-标注（encoder-labeler）的模型(Kurata et al. 2016)，其中encoder RNN是对输入序列的逆序编码，decoder RNN的输入不仅有当前输入词，还有上一时刻的预测得到的语义标签，如图[11](fig:encoder-labeller)所示。Zhu (Zhu et al. 2016)和Liu (Liu et al. 2016)等人分别将基于关注机（attention）的encoder-decoder模型应用于口语理解，并提出了基于“聚焦机”（focus）的encoder-decoder模型，如图[12](fig:attention_focus)所示。其中attention模型(Bahdanau et al. 2014)利用decoder RNN中的上一时刻t−1t−1的隐层向量和encoder RNN中每一时刻的隐层向量依次计算一个权值αt,i,i=1,…,Tαt,i,i=1,…,T，再对encoder RNN中的隐层向量做加权和得到tt时刻的decoder RNN的输入。而focus模型则利用了序列标注中输入序列与输出序列等长、对齐的特性，decoder RNN在tt时刻的输入就是encoder RNN在tt时刻的隐层向量。(Zhu et al. 2016; Liu et al. 2016)中的实验表明focus模型的结果明显优于attention，且同时优于不考虑输出依赖关系的双向循环神经网络模型。目前在ATIS评测集合上，对于单个语义标签标注任务且仅利用原始文本特征的已发表最好结果是95.79%（F-score）
- 此外，许多循环神经网络的变形也在口语理解中进行了尝试和应用，比如：加入了外部记忆单元（External Memory）的循环神经网络可以提升网络的记忆能力(Peng et al. 2015)

### 关于序列建模，是时候抛弃RNN和LSTM了?

	-机器之心
	- 在 2014 年，RNN 和 LSTM 起死回生。我们都读过 Colah 的博客《Understanding LSTM Networks》和 Karpathy 的对 RNN 的颂歌《The Unreasonable Effectiveness of Recurrent Neural Networks》。但当时我们都「too young too simple」
	- 现在，序列变换（seq2seq）才是求解序列学习的真正答案，序列变换还在语音到文本理解的任务中取得了优越的成果，并提升了 Siri、Cortana、谷歌语音助理和 Alexa 的性能
	- 在 2015-2016 年间，出现了 ResNet 和 Attention 模型。从而我们知道，LSTM 不过是一项巧妙的「搭桥术」。并且注意力模型表明 MLP 网络可以被「通过上下文向量对网络影响求平均」替换

### Adaptive Graph Convolutional Neural Networks
  - Graph Convolutional Neural Network
  - 图卷积神经网络（Graph CNN）是经典 CNN 的推广方法，可用于处理分子数据、点云和社交网络等图数据。Graph CNN 中的的滤波器大多是为固定和共享的图结构而构建的。但是，对于大多数真实数据而言，图结构的大小和连接性都是不同的。
  - 本论文提出了一种有泛化能力且灵活的 Graph CNN，其可以使用任意图结构的数据作为输入。通过这种方式，可以在训练时为每个图数据构建一个任务驱动的自适应图（adaptive graph）。
  - 为了有效地学习这种图，作者提出了一种距离度量学习方法。并且在九个图结构数据集上进行了大量实验，结果表明本文方法在收敛速度和预测准确度方面都有更优的表现。
### SOTA of Sequence Labeling
	+ Kevin Clark, Minh-Thang Luong, Christopher D Manning and Quoc V Le. 2018. Semi-supervised sequence modeling with cross-view training. arXiv preprint arXiv:1809.08370.

### 概率图模型学习理论及其应用

+ https://max.book118.com/html/2016/0304/36880806.shtm

### Coursera
+ https://www.coursera.org/specializations/probabilistic-graphical-models

### 概率图模型: Coursera课程资源分享和简介
+ https://blog.csdn.net/thither_shore/article/details/52185758

### Deep Inductive Network Representation Learning
+ Adobe Research, Google, Intel Labs WWW2018
+ 通用的归纳图表示学习架构DeepGL

### Graph Edit Distance Computation via Graph Neural Networks
+ 改进了现有的网络图计算方法 GED/MCS 的时间复杂度较高的问题
+ **首次将神经网络引入图计算中**

### Graph2Seq:Graph to Sequence Learning with Attention-Based Neural Networks
+ IBM Research
+ ACL 2018
+ 由Graph 编码器和Seq 解码器构成

### TCN（498 stars on Github，来自Zico Kolter）
    - 序列建模基准和时域卷积网络
    - 项目地址：https://github.com/locuslab/TCN


### A Tutorial on Modeling and Inference in Undirected Graphical Models for Hyperspectral Image Analysis
- https://arxiv.org/abs/1801.08268
- https://github.com/UBGewali/tutorial-UGM-hyperspectral

### Hierachical Graph Representation Learning with Differentiable Pooling
+ 斯坦福
+ KDD 2018
+ 提出一种图卷积的网络层次化的池化方法
+ 图卷积网络 + 池化 : 将 Graph 表示为低维向量, 从而简化graph 间的计算
+ 本文与现有Graph Pooling 方法不同的是, 作者将graph 中的层次关系考虑了进去
