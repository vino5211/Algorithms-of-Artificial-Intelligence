- 302页吴恩达Deeplearning.ai课程笔记，详记基础知识与作业代码

- TensorFlow Hub（350 stars on Github，来自TensorFlow团队）
    - 该项目是一个发布、发现和重用TensorFlow中机器学习模块的开源库。
    - 项目地址：https://github.com/tensorflow/hub
- Chess-alpha-zero（1014 stars on Github，来自Samuel）
    - 通过AlphaGo Zero方法进行国际象棋的强化学习
    - 项目地址：
    - https://github.com/Zeta36/chess-alpha-zero
- Deep-neuroevolution（573 stars on Github，来自 Uber）
    - Uber AI Lab 开源的深度神经进化算法。
    - 项目地址：
    - https://github.com/uber-common/deep-neuroevolution
- TCN（498 stars on Github，来自Zico Kolter）
    - 序列建模基准和时域卷积网络
    - 项目地址：https://github.com/locuslab/TCN
- Ann-visualizer（654 stars on Github，来自Prodi Code）
    - ANN Visualizer 是一个 Python 库，可以用一行代码可视化人工神经网络，利用 Python 的 graphviz 库创建一个整洁的可视化神经网络。
    - 项目地址：https://github.com/Prodicode/ann-visualizer

- 机器学习入门第一课：决策树学习概述与实现
- 10个例子带你了解机器学习中的线性代数
    1. Dataset and Data Files 数据集和数据文件
    2. Images and Photographs 图像和照片
    3. One-Hot Encoding one-hot 编码
    4. Linear Regression 线性回归
    5. Regularization 正则化
        + 一种常用于模型在数据拟合时尽量减小系数值的技术称为正则化，常见的实现包括正则化的 L2 和 L1 形式。
        + 这两种正则化形式实际上是系数矢量的大小或长度的度量，是直接脱胎于名为矢量范数的线性代数方法。
    6. Principal Component Analysis 主成分分析
        + PCA 方法的核心是线性代数的矩阵分解方法，可能会用到特征分解，更广义的实现可以使用奇异值分解（SVD）
    7. Singular-Value Decomposition 奇异值分解
        - 该方法在线性代数中有广泛的用途，可直接应用于特征选择、可视化、降噪等方面
    8. Latent Semantic Analysis 潜在语义分析
        - 在用于处理文本数据的机器学习子领域（称为自然语言处理），通常将文档表示为词出现的大矩阵。
        - 例如，矩阵的列可以是词汇表中的已知词，行可以是文本的句子、段落、页面或文档，矩阵中的单元格标记为单词出现的次数或频率。
        - 这是文本的稀疏矩阵表示。矩阵分解方法（如奇异值分解）可以应用于此稀疏矩阵，该分解方法可以提炼出矩阵表示中相关性最强的部分。以这种方式处理的文档比较容易用来比较、查询，并作为监督机器学习模型的基础。
        - 这种形式的数据准备称为潜在语义分析（简称 LSA），也称为潜在语义索引（LSI）
    9. Recommender Systems 推荐系统
        - 涉及产品推荐的预测建模问题被称为推荐系统，这是机器学习的一个子领域。
        - 例如，基于你在亚马逊上的购买记录和与你类似的客户的购买记录向你推荐书籍，或根据你或与你相似的用户在 Netflix 上的观看历史向你推荐电影或电视节目。
        - 推荐系统的开发主要涉及线性代数方法。一个简单的例子就是使用欧式距离或点积之类的距离度量来计算稀疏顾客行为向量之间的相似度。
        - 像奇异值分解这样的矩阵分解方法在推荐系统中被广泛使用，以提取项目和用户数据的有用部分，以备查询、检索及比较
    10. Deep Learning 深度学习


- blog 

    - http://www.cnblogs.com/zhangchaoyang/default.html

- EM 算法

    - http://www.cnblogs.com/zhangchaoyang

- Tree-CNN：一招解决深度学习中的「灾难性遗忘」
  - 深度学习领域一直存在一个比较严重的问题——“灾难性遗忘”，即一旦使用新的数据集去训练已有的模型，该模型将会失去对原数据集识别的能力
  - 为解决这一问题，本文提出了树卷积神经网络，通过先将物体分为几个大类，然后再将各个大类依次进行划分、识别，就像树一样不断地开枝散叶，最终叶节点得到的类别就是我们所要识别的类
  - Tree-CNN: A Deep Convolutional Neural Network for Lifelong Learning

- 资源 | textgenrnn：只需几行代码即可训练文本生成网络
  - 本文是一个 GitHub 项目，介绍了 textgenrnn，一个基于 Keras/TensorFlow 的 Python 3 模块。只需几行代码即可训练文本生成网络
  - 项目地址：https://github.com/minimaxir/textgenrnn?reddit=1
  - textgenrnn 是一个基于 Keras/TensorFlow 的 Python 3 模块，用于创建 char-rnn，具有许多很酷炫的特性：
  	- 它是一个使用注意力权重（attention-weighting）和跳跃嵌入（skip-embedding）等先进技术的现代神经网络架构，用于加速训练并提升模型质量。
  	- 能够在字符层级和词层级上进行训练和预测。
  	- 能够设置 RNN 的大小、层数，以及是否使用双向 RNN。
  	- 能够对任何通用的输入文本文件进行训练。
  	- 能够在 GPU 上训练模型，然后在 CPU 上使用这些模型。
  	- 在 GPU 上训练时能够使用强大的 CuDNN 实现 RNN，这比标准的 LSTM 实现大大加速了训练时间。
  	- 能够使用语境标签训练模型，能够更快地学习并在某些情况下产生更好的结果。
  + Tweet Generator：训练一个为任意数量的 Twitter 用户生成推文而优化的神经网络

- 前沿|BAIR提出人机合作新范式：教你如何安全高效的在月球着陆
- 业界 | 在个人电脑上快速训练Atari深度学习模型：Uber开源「深度神经进化」加速版
  - Uber 在去年底发表的研究中发现，通过使用遗传算法高效演化 DNN，可以训练含有超过 400 万参数的深度卷积网络在像素级别上玩 Atari 游戏；这种方式在许多游戏中比现代深度强化学习算法或进化策略表现得更好，同时由于更好的并行化能达到更快的速度。
  - 不过这种方法虽好但当时对于硬件的要求很高，近日 Uber 新的开源项目解决了这一问题，其代码可以让一台普通计算机在 4 个小时内训练好用于 Atari 游戏的深度学习模型。现在，技术爱好者们也可以接触这一前沿研究领域了
  - 项目 GitHub 地址：https://github.com/uber-common/deep-neuroevolution/tree/master/gpu_implementation
  - 参考Uber五篇论文：前沿 | 利用遗传算法优化神经网络：Uber提出深度学习训练新方式
- 吴恩达深度学习工程师课程汇总(附中文视频笔记)

  - https://zhuanlan.zhihu.com/p/30870804

- SIGIR 2018 | 通过深度模型加深和拓宽聊天话题，让你与机器多聊两句
  - 目前大多数基于生成的对话系统都会有很多回答让人觉得呆板无趣，无法进行有意思的长时间聊天。近日，山东大学和清华大学的研究者联合提出了一种使用深度模型来对话题进行延展和深入的方法 DAWnet。
  - 该方法能有效地让多轮对话系统给出的答复更加生动有趣，从而有助于实现人与机器的长时间聊天对话。机器之心对该研究论文进行了摘要编译。此外，研究者还公布了他们在本论文中所构建的数据集以及相关代码和参数设置
  - 论文、数据和代码地址：https://sigirdawnet.wixsite.com/dawnet

- 业界 | Petuum提出深度生成模型统一的统计学框架
  - Petuum 和 CMU 合作的论文《On Unifying Deep Generative Models》提出深度生成模型的统一框架。该框架在理论上揭示了近来流行的 GAN、VAE（及大量变体），与经典的贝叶斯变分推断算法、wake-sleep 算法之间的内在联系；为广阔的深度生成模型领域提供了一个统一的视角。7 月份在 ICML 2018 的名为「深度生成模型理论基础和应用」的研讨会将更进一步探讨深度生成模型的研究
  - 论文地址：https://arxiv.org/pdf/1706.00550.pdf

- oxford-cs-deepnlp
  - https://github.com/oxford-cs-deepnlp-2017/lectures
  - http://study.163.com/course/introduction/1004336028.htm
- 今日头条AI实验室主任李航：自然语言的现状和发展 | 北大AI公开课笔记
  - https://c.m.163.com/news/s/S1521443845851.html
  - 合理行动的智能机
  	- Turing Test
  - 人脑如何做语言理解
  	- 词汇
  	- 句法
  	- 语义
  	- 语用
  	- 有几个重要的脑区是和语言密切相关的：布洛卡区域和维尼科区
  	- 比如维尼科区负责词汇，布洛卡区负责句法
  	- 语言理解是非常复杂的，大脑一共有1011 到1015 个神经元，这样复杂的计算系统还是并行处理，我们每个人在做这样复杂的处理
  - attention
  - seq2seq
  - 对话
  	- 多人对话中也是，现在用的最多的是深度强化学习
  	- 谷歌提出的Neural Symbolic Machines模型，特点结合符号处理和神经处理，其框架也是基于分析的模型
  	- 还有华为方舟提出的类似模型（Neural Responding Machine）。
  	- 在多人中，微软提出层次化的深度强化学习Hierarchical Deep Reinforcement Learning。
  	- 对话目标可以分层，展开和复述，将有限状态机变成层次化。学习就可以用层次化甚至强化学习来做这样的东西。
        - 单轮对话
            - 分析 ： 基于分析就是分类问题
            - 检索 ： 检索当成匹配问题
            - 生成 :  生成当做是翻译问题; 这是比较新的系统，目前还不太好做。把问句转化成内部表示，然后再转化为答句
            - 云助手一般是第一种，问答系统一般是第二种，聊天机器人一般应用第三种
        - 多论对话
            - 自然语言理解
            - 自然语言生成
            - 对话管理

- 有这5小段代码在手，轻松实现数据可视化（Python+Matplotlib）

- 一文简述ResNet及其多种变体
- 资源 | 概率编程工具：TensorFlow Probability官方简介
    - TensorFlow Probability 适用于以下需求：
    - 希望建立一个生成数据模型，推理其隐藏进程。
    - 需要量化预测中的不确定性，而不是预测单个值。
    - 训练集具有大量相对于数据点数量的特征。
    - 结构化数据（例如，使用分组，空间，图表或语言语义）并且你想获取其中重要信息的结构。存有一个逆问题 - 请参考 TFDS'18 演讲视频（https://www.youtube.com/watch?v=Bb1_zlrjo1c）以重建测量中的融合等离子体

- 教程 | 简述表征句子的3种无监督深度学习方法
  - 介绍了三个使用 RNN 创建句子向量表征的无监督方法，并且在解决一个监督任务的过程中展现了它们的效率。
  - 基线模型 : average word2vec
  - 自编码器的结果比我们的基线模型要差一些（这可能是因为所用的数据集相对较小的缘故）。
  - skip-thought 向量模型语言模型都利用语境来预测句子表征，并得到了最佳结果

- ETH-DS3Lab at SemEval-2018 Task 7: Effectively Combining Recurrent and Convolutional Neural Networks for Relation Classification and Extraction

  - 本文来自苏黎世联邦理工学院 DS3Lab，文章针对实体关系抽取任务进行了非常系统的实验，并在第十二届国际语义评测比赛 SemEval 2018 的语义关系抽取和分类任务上获得冠军。本文思路严谨，值得国内学者们仔细研读。
- Personalizing Dialogue Agents: I have a dog, do you have pets too?
  - 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
  - 本文致力于解决以下问题：
  	- 聊天机器人缺乏一致性格特征
  	- 聊天机器人缺乏长期记忆
  	- 聊天机器人经常给出模糊的回应，例如 I don't know
  - 数据集链接
  	- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat
- DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding
  - 本文是悉尼科技大学发表于 AAAI 2018 的工作，这篇文章是对 Self-Attention 的另一种应用，作者提出一种新的方向性的 Attention，从而能更加有效地理解语义。
  - 代码链接
  	- https://github.com/shaohua0116/Group-Normalization-Tensorflow
- DetNet: A Backbone network for Object Detection
  - 本文来自清华大学和 Face++，文章分析了使用 ImageNet 预训练网络调优检测器的缺陷，研究通过保持空间分辨率和扩大感受野，提出了一种新的为检测任务设计的骨干网络 DetNet。
  - 实验结果表明，基于低复杂度的 DetNet59 骨干网络，在 MSCOCO 目标检测和实例分割追踪任务上都取得当前最佳的成绩。
- Imagine This! Scripts to Compositions to Videos
  - Video Caption
  - 本文以《摩登原始人》的动画片段作为训练数据，对每个片段进行详细的文本标注，最终训练得到一个可以通过给定脚本或文字描述生成动画片段的模型。
  - 模型称为 Craft，分为布局、实体、背景，三个部分。虽然现阶段模型存在着很多问题，但是这个研究在理解文本和视频图像高层语义方面有着很大的意义。

- Generating Diverse and Accurate Visual Captions by Comparative Adversarial Learning
  - Image Caption
  - 本文来自华盛顿大学和微软，文章提出一个基于 GAN 的 Image Caption 框架，亮点如下：
  	- 1. 提出用 comparative relevance score 来衡量 image-text 的质量从而指导模型的训练，并且在训练过程中引入 unrelated captions；
  	- 2. 利用 human evaluations 评估 caption 的 accuracy，给出了和传统的六个评价指标的结果对比；
  	- 3. 提出通过比较 caption feature vectors 的 variance 来评估 caption 的 diversity。
- Simultaneously Self-Attending to All Mentions for Full-Abstract Biological Relation Extraction
  - Self-Attention
  - 本文是 Andrew McCallum 团队应用 Self-Attention 在生物医学关系抽取任务上的一个工作。这篇论文作者提出了一个文档级别的生物关系抽取模型，作者使用 Google 提出包含 Self-Attention 的 transformer 来对输入文本进行表示学习，和原始的 transformer 略有不同在于他们使用了窗口大小为 5 的 CNN 代替了原始 FNN。
  - 代码链接
  	- https://github.com/patverga/bran
- Evaluation of Session-based Recommendation Algorithms
  - Recommender System
  - 本文系统地介绍了 Session-based Recommendation，主要针对 baseline methods, nearest-neighbor techniques, recurrent neural networks 和 (hybrid) factorization-based methods 等 4 大类算法进行介绍。
  - 此外，本文使用 RSC15、TMALL、ZALANDO、RETAILROCKET、8TRACKS 、AOTM、30MUSIC、NOWPLAYING、CLEF 等 7 个数据集进行分析，在 Mean Reciprocal Rank (MRR)、Coverage、Popularity bias、Cold start、Scalability、Precision、Recall 等指标上进行比较。
  - 代码链接
  	- https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0
- On the Convergence of Adam and Beyond
  - Neural Network
  - 本文是 ICLR 2018 最佳论文之一。在神经网络优化方法中，有很多类似 Adam、RMSprop 这一类的自适应学习率的方法，但是在实际应用中，虽然这一类方法在初期下降的很快，但是往往存在着最终收敛效果不如 SGD+Momentum 的问题。
  - 作者发现，导致这样问题的其中一个原因是因为使用了指数滑动平均，这使得学习率在某些点会出现激增。在实验中，作者给出了一个简单的凸优化问题，结果显示 Adam 并不能收敛到最优点。
  - 在此基础上，作者提出了一种改进方案，使得 Adam 具有长期记忆能力，来解决这个问题，同时没有增加太多的额外开销。
- Neural Baby Talk
  - Image Captioning
  - 本文是佐治亚理工学院发表于 CVPR 2018 的工作，文章结合了 image captioning 的两种做法：以前基于 template 的生成方法（baby talk）和近年来主流的 encoder-decoder 方法（neural talk）。
  - 论文主要做法其实跟作者以前的工作"Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning"类似：在每一个 timestep，模型决定生成到底是生成 textual word（不包含视觉信息的连接词），还是生成 visual word。其中 visual word 的生成是一个自由的接口，可以与不同的 object detector 对接。
  - 论文链接
  	- https://www.paperweekly.site/papers/1801
  - 代码链接
  	- https://github.com/jiasenlu/NeuralBabyTalk
- Adaptive Graph Convolutional Neural Networks
  - Graph Convolutional Neural Network
  - 图卷积神经网络（Graph CNN）是经典 CNN 的推广方法，可用于处理分子数据、点云和社交网络等图数据。Graph CNN 中的的滤波器大多是为固定和共享的图结构而构建的。但是，对于大多数真实数据而言，图结构的大小和连接性都是不同的。
  - 本论文提出了一种有泛化能力且灵活的 Graph CNN，其可以使用任意图结构的数据作为输入。通过这种方式，可以在训练时为每个图数据构建一个任务驱动的自适应图（adaptive graph）。
  - 为了有效地学习这种图，作者提出了一种距离度量学习方法。并且在九个图结构数据集上进行了大量实验，结果表明本文方法在收敛速度和预测准确度方面都有更优的表现。

- (***)基于CNN的阅读理解式问答模型：DGCNN
  -  Dilate Gated Convolutional Neural Network
  - Ref : 一文读懂「Attention is All You Need」| 附代码实现
  - 本模型——我称之为 DGCNN——是基于 CNN 和简单的 Attention 的模型，由于没有用到 RNN 结构，因此速度相当快，而且是专门为这种 WebQA 式的任务定制的，因此也相当轻量级。
  - SQUAD 排行榜前面的模型，如 AoA、R-Net 等，都用到了 RNN，并且还伴有比较复杂的注意力交互机制，而这些东西在 DGCNN 中基本都没有出现。
  - 这是一个在 GTX1060 上都可以几个小时训练完成的模型！
  - CIPS-SOGOU/WebQA

-WebQA
​	-  WebQA 的参考论文 Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question :
​	- 1. 直接将问题用 LSTM 编码后得到“问题编码”，然后拼接到材料的每一个词向量中；
​	- 2. 人工提取了 2 个共现特征；
​	- 3. 将最后的预测转化为了一个序列标注任务，用 CRF 解决。

- Semantic Segmentation PyTorch
- Pytorch NLP
- Non-local Neural Networks for Video Classification
	- Facebook视频分类开源代码
- Keras Project Template
	- Keras项目模板
	- 本项目是一个基于 Keras 库的项目模板，模板能让你更容易地构建和训练深度学习模型，并支持 Checkpoints 和 TensorBoard。
	- https://github.com/Ahmkel/Keras-Project-Template
- Agriculture KnowledgeGraph
	- 面向智慧农业的知识图谱及其应用系统
- 《迁移学习简明手册》
- MobilePose
	- 支持移动设备的单人姿态估计框架
- Meka
	- 多标签分类器和评价器
	- MEKA 是一个基于 Weka 机器学习框架的多标签分类器和评价器。本项目提供了一系列开源实现方法用于解决多标签学习和评估。
- Quick NLP
	- Quick NLP 是一个基于深度学习的自然语言处理库，该项目的灵感来源于 Fast.ai 系列课程。它具备和 Fast.ai 同样的接口，并对其进行扩展，使各类 NLP 模型能够更为快速简单地运行。



- 从字符级的语言建模开始，了解语言模型与序列建模的基本概念
- 一文读懂机器学习需要哪些数学知识(附全套优秀课程的网盘链接资源)
- 机器之心GitHub项目：从循环到卷积，探索序列建模的奥秘
	- 这是机器之心 GitHub 实现项目的第四期，前面几期分别介绍了卷积神经网络、生成对抗网络与带动态路由的 CapsNet。

<<<<<<< HEAD
- KNN与K-Means的区别
	- https://www.tuicool.com/articles/qamYZv
-【Facebook数据分析工具(情感分析、词频分析等)】
	- Booksoup allows you to analyse and traverse your downloaded facebook data, including features such as sentiment analysis and message frequency analysis over time.'
=======
- (*)【机器学习基本理论】详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解
- (*)似然与极大似然估计
	- https://zhuanlan.zhihu.com/p/22092462
	- 概率是在特定环境下某件事情发生的可能性
	- 而似然刚好相反，是在确定的结果下去推测产生这个结果的可能环境（参数）
	- ！[](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D%28%5Ctheta%7Cx%29+%3DP%28x%7C%5Ctheta%29)
	- 解释了硬币问题的似然估计方法
- （*）入门 | 什么是最大似然估计、最大后验估计以及贝叶斯参数估计

  - 机器之心
- 关于序列建模，是时候抛弃RNN和LSTM了?
	-机器之心
	- 在 2014 年，RNN 和 LSTM 起死回生。我们都读过 Colah 的博客《Understanding LSTM Networks》和 Karpathy 的对 RNN 的颂歌《The Unreasonable Effectiveness of Recurrent Neural Networks》。但当时我们都「too young too simple」
	- 现在，序列变换（seq2seq）才是求解序列学习的真正答案，序列变换还在语音到文本理解的任务中取得了优越的成果，并提升了 Siri、Cortana、谷歌语音助理和 Alexa 的性能
	- 在 2015-2016 年间，出现了 ResNet 和 Attention 模型。从而我们知道，LSTM 不过是一项巧妙的「搭桥术」。并且注意力模型表明 MLP 网络可以被「通过上下文向量对网络影响求平均」替换
	
- 如何从零开始构建深度学习项目？

  - GAN 着色
- 入门 | 通过 Q-learning 深入理解强化学习
	-机器之心
- 学界 | Uber AI论文：利用反向传播训练可塑神经网络，生物启发的元学习范式
	- 机器之心
	- 当学习完成之后，智能体的知识就固定不变了；如果这个智能体被用于其他的任务，那么它需要重新训练（要么完全重来，要么部分重新训练），而这又需要大量新的训练数据。相比较之下，生物智能体具备一种出色的能力，这个能力使它们快速高效地学习持续性经验：动物可以学会找到食物源并且记下（最快到达食物源的路径）食物源的位置，发现并记住好的或者不好的新事物或者新场景，等等——而这些往往只需要一次亲身经历就能完成。
- 【Facebook数据分析工具(情感分析、词频分析等)】

  - Booksoup allows you to analyse and traverse your downloaded facebook data, including features such as sentiment analysis and message frequency analysis over time.' 
>>>>>>> 39bbc36434b4d134c348ccde50cba7e98e00a6ae
>
>- by Jake Reid Browning GitHub : https://github.com/Buroni/booksoup
- 《Keras and Convolutional Neural Networks (CNNs) | PyImageSearch》by Adrian Rosebrock
	- https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
- software license
	- http://geek-workshop.com/thread-1860-1-1.html

- Speech and Language Processing(3rd ed. draft)
- http://www.deeplearningbook.org/

- Attention
  - ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
- leetcode
  - Dynamic Programming
    - https://www.zhihu.com/question/23995189
- https://web.stanford.edu/~pkurlat/teaching/5%20-%20The%20Bellman%20Equation.pdf

+ 【RNN/LSTM的衰落】《The fall of RNN / LSTM》by Eugenio Culurciello 
	+ https://pan.baidu.com/s/1CdJsLFu1zM9dEjV-no3lng
+ DeepLearn blog
	+ http://deeplearn-ai.com/deeplearn/?i=1
	+ https://github.com/GauravBh1010tt/DeepLearn
+ wikidata
	+ https://www.wikidata.org/wiki/Wikidata:Main_Page
+ PyCharm选择性忽略PEP8代码风格警告信息
	+ https://blog.csdn.net/zgljl2012/article/details/51907663
+ 使用Keras搭建深度残差网络
	+ https://www.2cto.com/kf/201607/526863.html
+ 解决机器学习问题有通法
	+ https://www.jiqizhixin.com/articles/2017-09-21-10
+ 强化学习大讲堂
	+ https://zhuanlan.zhihu.com/sharerl
+ SVM 与LR 的区别 
	 http://www.cnblogs.com/zhizhan/p/5038747.html

+ attention is all your need
	+ http://nlp.seas.harvard.edu/2018/04/03/attention.html
	+ https://github.com/harvardnlp/annotated-transformer
+ Question answering over Freebase (single-relation)
	+ https://github.com/quyingqi/kbqa-ar-smcnn
+ App : https://lopespm.github.io/apps/2018/03/12/arxiv-papers

+ 贝叶斯机器学习前沿进展
	+ http://chuansong.me/n/2152434851911
- Imitation Learning
	- One Shot Imitation Learning
   	- 这篇论文提出一个比较通用的模仿学习的方法。这个方法在运行时，需要一个完成当前任务的完整演示，和当前状态。假设我要机器人搭方块，那么我给它一个完整的把方块搭好的视频演示，再告诉他当前方块都在哪里。这个模型会用CNN和RNN来处理任务的演示，这样，它就有一个压缩过的演示纲要。模型再用CNN处理当前状态，得到一个压缩过的当前状态信息。利用Attention Model来扫描演示纲要，我们就得到了“与当前状态最有关的演示的步骤”，再将这些信息全部传递给一个决策器。然后输出决策。
    - (不确定)https://github.com/tianheyu927/mil
- 概要：NIPS 2017 Deep Learning for Robotics Pieter Abbeel
  - https://zhuanlan.zhihu.com/p/32089849
- Meta Learning Shared Hierarchies
- 代码生成
	- DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning 25 apr 2017
      A Syntactic Neural Model for General-Purpose Code Generation 6 apr 2017
      RobustFill: Neural Program Learning under Noisy I/O 21 mar 2017
      DeepFix: Fixing Common C Language Errors by Deep Learning 12 feb 2017
      DeepCoder: Learning to Write Programs 7 nov 2016
      Neuro-Symbolic Program Synthesis 6 nov 2016
      Deep API Learning 27 may 2016
---
- Distribution RL
  - https://mtomassoli.github.io/2017/12/08/distributional-r1/
- 深度学习迁移学习简介
	- A Gentle Introduction to Transfer Learning for Deep Learning | Machine Learning Mastery by Jason Brownlee
- 【(Python)多种模型(Naive Bayes, SVM, CNN, LSTM, etc)实现推文情感分析】’Sentiment analysis on tweets using Naive Bayes, SVM, CNN, LSTM, etc.'
- 【AI与深度学习2017年度综述】《AI and Deep Learning in 2017 – A Year in Review | WildML》by Denny Britz
- “实用脚本：Ubuntu 16上全自动安装Nvidia驱动程序, Anaconda, CUDA, fastai等” 
---

- 【2018年深度学习十大警告性预测：DL硬件企业面临倒闭、元学习取代SGD、生成模型成主流、自实践自动知识构建、直觉机弥合语义鸿沟、可解释性仍不可及、缺少理论深度的DL论文继续井喷、打造学习环境以实现产业化、会话认知、AI伦理运用】《10 Alarming Predictions for Deep Learning in 2018》by Carlos E. Perez O网页链接
- 《Learning More Universal Representations for Transfer-Learning》Y Tamaazousti, H L Borgne, C Hudelot, M E A Seddik, M Tamaazousti [CEA & University of Paris-Saclay] (2017) O网页链接 
  - Universal Representations
  - Transfer-Learning
- 《Letter-Based Speech Recognition with Gated ConvNets》V Liptchinsky, G Synnaeve, R Collobert [Facebook AI Research] (2017) O网页链接 GitHub: https:\//github.com\/facebookresearch/wav2letter 
  - Letter-Based Speech Recognition 字符表示
  - Gated ConvNets

---

- 没能去参加NIPS 2017？这里有一份最详细的现场笔记（附PDF）
- 视频：Pieter Abbeel NIPS 2017大会报告 《Deep Learning for Robots》（附PDF）
  - 12月6日下午，加州大学伯克利分校教授、机器人与强化学习领域的大牛Pieter Abbeel在NIPS 2017的报告视频。

---

Super Repository

Papers:


- NLP
  - https://web.stanford.edu/~jurafsky/slp3/
- DRL4NLP

- Playing Atari with Deep Reinforcement Learning
  - https://arxiv.org/pdf/1312.5602.pdf
- Deep Reinforcement Learning with a Natural Language Action Space
  - https://arxiv.org/pdf/1511.04636.pdf
- Bidirectional LSTM-CRF Models for Sequence Tagging
  - https://arxiv.org/abs/1508.01991
- Hinton 2006 Deep Belief Network
  - list:
    - A fast learning algorithm for deep belief nets. Neural Computation
    - Greedy Layer-Wise Training of Deep Networks
    - Efficient Learning of Sparse Representations with an Energy-Based Model
  - key point:
    - unsup learn for pre train
    - train by layers
    - sup learn for tuning weight between layers

Office sites:
- pygame
  - http://www.pygame.org/news
- Deep Reinforcement Learning for Keras.
  - http://keras-rl.readthedocs.io/
  - https://github.com/matthiasplappert/keras-rl
- ai-code
  - http://www.ai-code.org/
- DeepMind
  - https://deepmind.com/
- 库
  开始尝试机器学习库可以从安装最基础也是最重要的开始，像numpy和scipy。
  - 查看和执行数据操作：pandas（http://pandas.pydata.org/）
  - 对于各种机器学习模型：scikit-learn（http://scikit-learn.org/stable/）
  - 最好的gradient boosting库：xgboost（https://github.com/dmlc/xgboost）
  - 对于神经网络：keras（http://keras.io/）
  - 数据绘图：matplotlib（http://matplotlib.org/）
  - 监视进度：tqdm（https://pypi.python.org/pypi/tqdm） 

Videos :

- 21 Deep Learning Videos, Tutorials & Courses on Youtube from 2016
  - https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/
- RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning
  - https://www.youtube.com/playlist?list=PLV_1KI9mrSpGFoaxoL9BCZeen_s987Yxb
- 斯坦福2017季CS224n深度学习自然语言处理课程
  - https://www.bilibili.com/video/av13383754/
- CS224d: Deep Learning for Natural Language Processing ( Doing )
  - https://www.bilibili.com/video/av9143821/?from=search&seid=9547251413889295037
- CS 294: Deep Reinforcement Learning, Fall 2017
  - http://rll.berkeley.edu/deeprlcourse/#lecture-videos
- Morvan
  - https://github.com/MorvanZhou
- 李宏毅深度学习(2017)
  - https://www.bilibili.com/video/av9770302/

---

Github Projects:

- Machine Learning Mindmap / Cheatsheet ( Only Pictures)
  - https://github.com/dformoso/machine-learning-mindmap
  - A Mindmap summarising Machine Learning concepts, from Data Analysis to Deep Learning.
- DeepMind : Teaching Machines to Read and Comprehend
  - https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend
  - This repository contains an implementation of the two models (the Deep LSTM and the Attentive Reader) described in Teaching Machines to Read and Comprehend by Karl Moritz Hermann and al., NIPS, 2015. This repository also contains an implementation of a Deep Bidirectional LSTM.
- A-Guide-to-DeepMinds-StarCraft-AI-Environment
  - https://github.com/llSourcell/A-Guide-to-DeepMinds-StarCraft-AI-Environment
  - This is the code for "A Guide to DeepMind's StarCraft AI Environment" by Siraj Raval on Youtube
    - Must install in venv to avoid wrong operation 

---

Other Resources:

- 模型汇总16 各类Seq2Seq模型对比及《Attention Is All You Need》中技术详解
  - https://zhuanlan.zhihu.com/p/27485097
    +模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用
  - https://zhuanlan.zhihu.com/p/31547842?utm_source=wechat_session&utm_medium=social 
- 人工智能 Java 坦克机器人系列强化学习-IBM Robo code
  - https://www.ibm.com/developerworks/cn/java/j-lo-robocode2/index.html
- 遗传算法
  - https://www.zhihu.com/question/23293449
- 蒙特卡罗算法
  - https://www.zhihu.com/question/20254139
- RL
  - 深度强化学习（Deep Reinforcement Learning）入门：RL base & DQN-DDPG-A3C introduction
    - https://zhuanlan.zhihu.com/p/25239682
  - 深度增强学习前沿算法思想【DQN、A3C、UNREAL简介】
    - http://blog.csdn.net/mmc2015/article/details/55271605
  - Google Deepmind大神David Silver带你认识强化学习
    - https://www.leiphone.com/news/201608/If3hZy8sLqbn6uvo.html
    - http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf
- NLP
  - 综述 | 一文读懂自然语言处理NLP（附学习资料）
    - http://www.pinlue.com/article/2017/11/1413/074849597604.html
- DL
  - 126 篇殿堂级深度学习论文分类整理 从入门到应用 | 干货9
    - https://www.leiphone.com/news/201702/FWkJ2AdpyQRft3vW.html?utm_source=tuicool&utm_medium=referral
  - DeepLearningBook读书笔记
    - https://github.com/exacity/simplified-deeplearning/blob/master/README.md 
- ML
  - 良心GitHub项目：各种机器学习任务的顶级结果（论文）汇总
    - https://www.ctolib.com/topics-126416.html
- Transfor Learning
  - 14 篇论文为你呈现「迁移学习」研究全貌
    - https://www.ctolib.com/topics-125968.html
- SKLearn(工程用用的较多的模块介绍)
  - http://blog.csdn.net/column/details/scikitlearninaction.html
- Tensorflow 
  - （较好）转TensorFlow实现案例汇集：代码+笔记
    - https://zhuanlan.zhihu.com/p/29128378 
  - 数十种TensorFlow实现案例汇集：代码+笔记
    - http://dy.163.com/v2/article/detail/C3J6JU2U0511AQHO.html
  - 【推荐】TensorFlow/PyTorch/Sklearn实现的五十种机器学习模型
    - https://mp.weixin.qq.com/s/HufdD3OSJIK2yAexM-Wb5w
- Others
  - cs231n课程笔记翻译
    - http://www.cnblogs.com/xialuobo/p/5867314.html
  - 全网AI和机器学习资源大合集（研究机构、视频、博客、书籍...）
    - http://www.sohu.com/a/164766699_468650
- NIPS大会最精彩一日：AlphaZero遭受质疑；史上第一场正式辩论与LeCun激情抗辩；元学习&强化学习亮点复盘
  - https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650734461&idx=1&sn=154e7ff280626bbd6feda4e5607eecc4&chksm=871b3b03b06cb215b3d1d85a08306fa232311d585788d1024ded9305a99dbe4104598df05e29&mpshare=1&scene=1&srcid=1209sY3iZJvGZUUoHQDqWNrA&pass_ticket=VPRBJUnIlp%2BYQtpx6zRWEjZE9o39jBz2mDq5fAz7NkU2RxaP%2BuJhsCR4DDwVHJbm#rd
- 自然语言顶级会议ACL 2016谷歌论文汇集
  - https://www.jiqizhixin.com/articles/2016-08-08-7
- 解决机器学习问题有通法
  - https://www.jiqizhixin.com/articles/2017-09-21-10
- 比AlphaGo Zero更强的AlphaZero来了！8小时解决一切棋类！
  - https://arxiv.org/pdf/1712.01815.pdf
  - https://www.reddit.com/r/chess/comments/7hvbaz/mastering_chess_and_shogi_by_selfplay_with_a/
  - https://zhuanlan.zhihu.com/p/31749249

- DR-BiLSTM: Dependent Reading Bidirectional LSTM for Natural Language Inference
	- https://arxiv.org/abs/1802.05577
	- https://www.arxiv-vanity.com/papers/1802.05577/
- Deep Learning for Sentiment Analysis : A Survey
	- 近年来，深度学习有了突破性发展，NLP 领域里的情感分析任务逐渐引入了这种方法，并形成了很多业内最佳结果。本文中，来自领英与伊利诺伊大学芝加哥分校的研究人员对基于深度学习的情感分析研究进行了详细论述。
- Generating Wikipedia by Summarizing Long Sequences
	- 本文来自 Google Brain，通过长序列摘要生成维基百科。
- MaskGAN: Better Text Generation via Filling in the ______
	- 谷歌大脑提出使用生成对抗网络（GAN）来提高文本质量，它通过显式地训练生成器产生高质量文本，并且已经在图像生成领域取得很大成功。GAN 最初设计用于输出可微分的值，所以生成离散语言是具有挑战性的。作者认为验证复杂度本身不代表模型生成的文本质量。
	- 本文引入条件 actor-critic GAN，它可以基于上下文填充缺失的文本。本文从定性和定量的角度证明，相比最大似然训练的模型，这种方法生成了更真实的有条件和无条件文本样本。
- Investigating the Working of Text Classifiers
	- 文本分类问题，给一段文本指定一个类别，在主题分类和情感分析中都有应用到。它的难点在于如何在具有语义的文本中，对句子之间的内在联系（语义或句法）进行编码。这对文本情感分类很关键，因为比如像“对照”或者“因果”等关系，会直接决定整个文档的性质。
	- 本文并没有提出一套完整的解决方法，而是通过构建新的数据集（训练集和测试集尽可能不包含共同的关键词），验证上面的猜想。此外，作者还设计了一种 ANON 的正则方法，让网络不那么容易记住文档的关键词。
	
- A Hybrid Framework for Text Modeling with Convolutional RNN
	- 本文使用 RNN+CNN 的结构来完成 NLP 中的问答任务，其亮点在于使用 RNN 获取 question 和 answer 的上下文语义，CNN 在语义层面对二者进行操作。
	- http://www.sohu.com/a/165562211_651893
- Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning
	- 本文回顾了用于模型评估、模型选择和算法选择任务中的不同技术，并参考理论和实证研究讨论了每种技术的主要优势和劣势。
- Training Neural Networks by Using Power Linear Units (PoLUs)
	- 本文设计了一种 PoLU 激活函数，在多个数据集上取得超过 ReLU 的性能。
- Learning Continuous User Representations through Hybrid Filtering with doc2vec
	- 本文把用行为使用 item 描述进行串连，构成文档，并使用 doc2vec 训练用户表示向量。
- NDDR-CNN: Layer-wise Feature Fusing in Multi-Task CNN by Neural Discriminative Dimensionality Reduction
	- 本文研究的问题是多任务学习。作者提出了一种对多个网络（对应多个任务）进行逐层特征融合的方法。
- An Attention-based Collaboration Framework for Multi-View Network Representation Learning
  - 本文是网络表示学习大牛 Jian Tang 的工作，论文利用 multi-view 来对网络进行表示学习。各个 view 之间通过共享邻居来保证所有节点表示在同一个空间中，同时，通过引入 attention 机制，可以学到不同节点在不同 view 的权重。

- 2017年度最值得读的AI论文 | NLP篇
	- http://chuansong.me/n/2168901252931
- 总结 | 2016年最值得读的自然语言处理领域Paper 
	- http://chuansong.me/n/1440373152360
- PaperWeekly 第二十七期 | VAE for NLP 
	- http://chuansong.me/n/1628708852431
- 数据开放 | PaperWeekly交流群对话数据 
	- http://chuansong.me/n/1653837452325
- 综述 | 知识图谱研究进展 
	- http://chuansong.me/n/1687512252413

- OpenAI-2018年强化学习领域7大最新研究方向全盘点
	- https://zhuanlan.zhihu.com/p/33630520

- TensorForce Bitcoin Trading Bot
	- 基于深度增强学习的比特币交易机器人
- CakeChat
	- 情感对话机器人
- SentiBridge
	- 中文实体情感知识库

- gradient-checkpointing
	- 神经网络训练省内存神器
	- 本项目是由 OpenAI 提供的内存平衡工具。前馈模型可以在仅增加20%计算时间的基础上，让 GPU 适应十倍大的模型。
	- 项目链接：https://github.com/openai/gradient-checkpointing
- Minigo
	- 基于AlphaGo Zero核心算法的围棋AI
	- 项目链接：https://github.com/tensorflow/minigo

- Synonyms
	- 最好的中文近义词工具包

- (***)「知识表示学习」专题论文推荐 | 每周论文清单
	- https://zhuanlan.zhihu.com/p/33606964

- Deep learning in production with Keras, Redis, Flask, and Apache
	- https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
- MIT course 深度学习导论 by Alexander Amini, Ava Soleimany, Harini Suresh
	- https://www.bilibili.com/video/av19113488/
	- Sequence Modeling with Neural Networks、Convolutional Neural Networks、Deep Generative Modeling
- A Tutorial on Modeling and Inference in Undirected Graphical Models for Hyperspectral Image Analysis
	- https://arxiv.org/abs/1801.08268
	- https://github.com/UBGewali/tutorial-UGM-hyperspectral
---


- word embedding
  - word2vec
  - glove
  - fastText


- represent
  - word2vec(distributed representation)
  - memory neural networks
    - https://github.com/facebook/MemNN
    - https://arxiv.org/abs/1701.08718
- precision 和 accuracy
  -     TP：True Positive，即正确预测出的正样本个数
        FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）
        TN：True Negative，即正确预测出的负样本个数
        FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）
        
        Precision：TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例
        Recall：TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少
        Accuracy：(TP+TN)÷(TP+NP+TN+FN)，分类器对整体的判断能力，即正确预测的比例
- 中文分词指标评价
  - 准确率(Precision)和召回率(Recall)
  	Precision = 正确切分出的词的数目/切分出的词的总数
  	Recall = 正确切分出的词的数目/应切分出的词的总数

  	综合性能指标F-measure
  	Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)
  	β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure
  	F1 = 2*Precisiton*Recall/(Precision+Recall)

  	未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
  	R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
  	R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数

CU-NLP
aspect level情感分析相关工作

1. SemEval-2014 Task 4: Aspect Based Sentiment Analysis**
2. NRC-Canada-2014: Detecting Aspects and Sentiment in Customer Reviews**
3. DCU: Aspect-based Polarity Classification for SemEval Task 4**
4. Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification**
5. Aspect Specific Sentiment Analysis using Hierarchical Deep Learning**
6. PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis**
7. Effective LSTMs for Target-Dependent Sentiment Classification**

1：aspect level情感分析的系统介绍；

2、3：传统分类器方法实现aspect level的情感分析；

4、5、6、7：神经网络方法实现aspect level的情感分析。

---

  - http://aclweb.org/anthology/D16-1058
- 在中文命名实体识别中，现在比较好（准确率和召回率）的算法都有哪些？
  - https://www.zhihu.com/question/19994255


- Attention 
  
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
  - http://www.cnblogs.com/robert-dlut/p/5952032.html

    - Attention机制最早是在视觉图像领域提出来的，应该是在九几年思想就提出来了，但是真正火起来应该算是google mind团队的这篇论文《Recurrent Models of Visual Attention》[14]，他们在RNN模型上使用了attention机制来进行图像分类。随后，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》 [1]中，使用类似attention的机制在机器翻译任务上将翻译和对齐同时进行，他们的工作算是是第一个提出attention机制应用到NLP领域中。接着类似的基于attention机制的RNN模型扩展开始应用到各种NLP任务中。最近，如何在CNN中使用attention机制也成为了大家的研究热点。下图表示了attention研究进展的大概趋势。
