##NLP
- Fast.ai推出NLP最新迁移学习方法「微调语言模型」，可将误差减少超过20%！
	- Fine-tuned Language Models for Text Classification

- FoolNLTK
  - 中文处理工具包
  - 本项目特点：
    - 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
    - 基于 BiLSTM 模型训练而成
    - 包含分词，词性标注，实体识别，都有比较高的准确率
    - 用户自定义词典
  - 项目链接：https://github.com/rockyzhengwu/FoolNLTK
- MUSE
  - 多语言词向量 Python 库
  - 由 Facebook 开源的多语言词向量 Python 库，提供了基于 fastText 实现的多语言词向量和大规模高质量的双语词典，包括无监督和有监督两种。其中有监督方法使用双语词典或相同的字符串，无监督的方法不使用任何并行数据。
  - 无监督方法具体可参考 Word Translation without Parallel Data 这篇论文。
  - 论文链接：https://www.paperweekly.site/papers/1097
  - 项目链接：https://github.com/facebookresearch/MUSE

- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
  + 本文来自 Facebook AI Research。本文研究监督句子嵌入，作者研究并对比了几类常见的网络架构（LSTM，GRU，BiLSTM，BiLSTM with self attention 和 Hierachical CNN）, 5 类架构具很强的代表性。
  + 论文链接：https://www.paperweekly.site/papers/1332
  + 代码链接：https://github.com/facebookresearch/InferSent
- Multilingual Hierarchical Attention Networks for Document Classification
  + 本文使用两个神经网络分别建模句子和文档，采用一种自下向上的基于向量的文本表示模型。	首先使用 CNN/LSTM 来建模句子表示，接下来使用双向 GRU 模型对句子表示进行编码得到文档表示。
  + 论文链接：https://www.paperweekly.site/papers/1152
  + 代码链接：https://github.com/idiap/mhan
---
  Temp Paper Repository

- NLP Task
  - pos tagging, word segmentation, NER
  - semantic analysis, machine translate, machine reading comprehension, QA system, natural language generation
  - 一文概述2017年深度学习NLP重大进展与趋势
    - http://www.qingpingshan.com/bc/jsp/361202.html
    - Tools
      - AllenNLP
      - ParlAI
      - OpenNMT
  - https://www.wxwenku.com/d/100329482
    - 但是由于语言本身已经是一种高层次的表达，深度学习在 NLP 中取得的成绩并不如在视觉领域那样突出。尤其是在 NLP 的底层任务中，基于深度学习的算法在正确率上的提升并没有非常巨大，但是速度却要慢许多，这对于很多对 NLP 来说堪称基础的任务来说，是不太能够被接受的，比如说分词
    - 在完形填空类型的阅读理解（cloze-style machine reading comprehension）上，基于 attention 的模型也取得了非常巨大的突破（在 SQuAD 数据集上，2016 年 8 月的 Exact Match 最好成绩只有 60%，今年 3 月已经接近 77%，半年时间提升了接近 20 个点，这是极其罕见的）
    - 深度学习的不可解释的特性和对于数据的需求，也使得它尚未在要求更高的任务上取得突破，比如对话系统（虽然对话在 2016 年随着 Echo 的成功已经被炒得火热）
    - 在大多数端到端的 NLP 应用中，在输入中包括一些语言学的特征（例如 pos tag 或 dependency tree）并不会对结果有重大影响。我们的一些粗浅的猜测，是因为目前的 NLP 做的这些特征，其实对于语义的表示都还比较差，某种程度来说所含信息还不如 word embedding 来的多
    - 关于阅读理解（Open-domain QA）
      - 幸好 Stanford 的 Chen Danqi 大神的 Reading Wikipedia to Answer Open-Domain Questions 打开了很多的方向。通过海量阅读（「machine reading at scale」），这篇文章试图回答所有在 wikipedia 上出现的 factoid 问题。其中有大量的工程细节，在此不表，仅致敬意。
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
  -     准确率(Precision)和召回率(Recall)
        Precision = 正确切分出的词的数目/切分出的词的总数
        Recall = 正确切分出的词的数目/应切分出的词的总数
        
        综合性能指标F-measure
        Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)
        β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure
        F1 = 2*Precisiton*Recall/(Precision+Recall)
        
        未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
        R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
        R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数
        

---

- 《Aspect Level Sentiment Classification with Deep Memory Network》阅读笔记
  - 
- 《Attention-based LSTM for Aspect-level Sentiment Classification》阅读笔记
  - http://aclweb.org/anthology/D16-1058
- 在中文命名实体识别中，现在比较好（准确率和召回率）的算法都有哪些？
  - https://www.zhihu.com/question/19994255
- ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
  - 第一种方法ABCNN0-1是在卷积前进行attention，通过attention矩阵计算出相应句对的attention feature map，然后连同原来的feature map一起输入到卷积层
  - 第二种方法ABCNN-2是在池化时进行attention，通过attention对卷积后的表达重新加权，然后再进行池化
  - 第三种就是把前两种方法一起用到CNN中

---

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



- Data 20171222
- Sockeye: A Toolkit for Neural Machine Translation**
  - 一个开源的产品级神经机器翻译框架，构建在 MXNet 平台上。
  - 论文链接：https://www.paperweekly.site/papers/1374**
  - 代码链接：https://github.com/awslabs/sockeye**
- Multilingual Hierarchical Attention Networks for Document Classification
  - 本文使用两个神经网络分别建模句子和文档，采用一种自下向上的基于向量的文本表示模型。首先使用 CNN/LSTM 来建模句子表示，接下来使用双向 GRU 模型对句子表示进行编码得到文档表示。
  - 论文链接：https://www.paperweekly.site/papers/1152**
  - 代码链接：https://github.com/idiap/mhan**
- (+++)Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
  - 本文来自 Facebook AI Research。本文研究监督句子嵌入，作者研究并对比了几类常见的网络架构（LSTM，GRU，BiLSTM，BiLSTM with self attention 和 Hierachical CNN）, 5 类架构具很强的代表性。
  - 论文链接：https://www.paperweekly.site/papers/1332**
  - 代码链接：https://github.com/facebookresearch/InferSent**
- Recurrent Neural Networks for Semantic Instance Segmentation
  - 本项目提出了一个基于 RNN 的语义实例分割模型，为图像中的每个目标顺序地生成一对 mask 及其对应的类概率。该模型是可端对端 + 可训练的，不需要对输出进行任何后处理，因此相比其他依靠 object proposal 的方法更为简单。
  - 论文链接：https://www.paperweekly.site/papers/1355**
  - 代码链接：https://github.com/facebookresearch/InferSent

---

- Neural Text Generation: A Practical Guide
  #Seq2Seq
  本文是一篇 Practical Guide，讲了很多用端到端方法来做文本生成问题时的细节问题和技巧，值得一看。
- End-to-End Optimization of Task-Oriented Dialogue Model with Deep Reinforcement Learning
  #Dialog Systems
  一篇基于强化学习的端到端对话系统研究工作，来自 CMU 和 Google。
  论文链接：http://www.paperweekly.site/papers/1257
- Machine Translation Using Semantic Web Technologies: A Survey
  #Neural Machine Translation
  本文是一篇综述文章，用知识图谱来解决机器翻译问题。
  论文链接：http://www.paperweekly.site/papers/1229
- Reinforcement Learning for Relation Classification from Noisy Data**
  - 将强度学习应用于关系抽取任务中，取得了不错的效果。本文已被 AAAI2018 录用。作者团队在上期 PhD Talk 中对本文做过在线解读。
    实录回顾：清华大学冯珺：基于强化学习的关系抽取和文本分类**
    论文链接：http://www.paperweekly.site/papers/1260
- Learning Structured Representation for Text Classification via Reinforcement Learning
  - 将强化学习应用于文本分类任务中，已被 AAAI2018录用。作者团队在上期 PhD Talk 中对本文做过在线解读。
    实录回顾：清华大学冯珺：基于强化学习的关系抽取和文本分类**
    论文链接：http://www.paperweekly.site/papers/1261
- Adversarial Ranking for Language Generation
  - 本文提出了一种 RankGAN 模型，来解决如何生成高质量文本的问题。
  论文链接：https://www.paperweekly.site/papers/1290
- Benchmarking Multimodal Sentiment Analysis
  - 多模态情感分析目前还有很多难点，该文提出了一个基于 CNN 的多模态融合框架，融合表情，语音，文本等信息做情感分析，情绪识别。
    论文链接：https://www.paperweekly.site/papers/1306
- End-to-end Learning for Short Text Expansion
  - 本文第一次用了 end to end 模型来做 short text expansion 这个 task，方法上用了 memory network 来提升性能，在多个数据集上证明了方法的效果；Short text expansion 对很多问题都有帮助，所以这篇 paper 解决的问题是有意义的。
    通过在多个数据集上的实验证明了 model 的可靠性，设计的方法非常直观，很 intuitive。
    论文链接：https://www.paperweekly.site/papers/1313
- DisSent: Sentence Representation Learning from Explicit Discourse Relations
  - 借助文档中一些特殊的词训练句子 embedding。使用文档中 but、because、although 等词，以及其前后或关联的句子构成语义模型。也就是，使用这些词和句子的关系，约束了句子向量的生成空间（使用句子向量，预测关联词），从而达到训练句子向量目的。
    文章只对英文语料进行了测试，实际中文这样的结构也很多，如：因为、所以、虽然、但是，可以参考。
    论文链接：https://www.paperweekly.site/papers/1324
- Deep AND-OR Grammar Networks for Visual Recognition
  - AOG 的全称叫 AND-OR graph，是一种语法模型（grammer model）。在人工智能的发展历程中，大体有两种解决办法：一种是自底向上，即目前非常流形的深度神经网络方法，另一种方法是自顶向下，语法模型可以认为是一种自顶向下的方法。
  - 把语法模型和深度神经网络模型结合起来，设计的模型同时兼顾特征的 exploration and exploitation（探索和利用），并在网络的深度和宽度上保持平衡；
    设计的网络结构，在分类任务和目标检测任务上，都比基于残差结构的方法要好。
- skorch
  - 兼容 Scikit-Learn 的 PyTorch 神经网络库
- FlashText
  - 关键字替换和抽取
- MatchZoo 
  - MatchZoo is a toolkit for text matching. It was developed to facilitate the designing, comparing, and sharing of deep text matching models.
- Geek-AI
  - MAgent
    - MAgent is a research platform for many-agent reinforcement learning. Unlike previous research platforms that focus on reinforcement learning research with a single agent or only few agents, MAgent aims at supporting reinforcement learning research that scales up from hundreds to millions of agents.
    - AAAI 2018 demo paper: MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence
  - 1m-agents
  - irgan
- Tensor Input
  - check  emnlp 
- Attention 
  - Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
  - http://www.cnblogs.com/robert-dlut/p/5952032.html
    - Attention机制最早是在视觉图像领域提出来的，应该是在九几年思想就提出来了，但是真正火起来应该算是google mind团队的这篇论文《Recurrent Models of Visual Attention》[14]，他们在RNN模型上使用了attention机制来进行图像分类。随后，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》 [1]中，使用类似attention的机制在机器翻译任务上将翻译和对齐同时进行，他们的工作算是是第一个提出attention机制应用到NLP领域中。接着类似的基于attention机制的RNN模型扩展开始应用到各种NLP任务中。最近，如何在CNN中使用attention机制也成为了大家的研究热点。下图表示了attention研究进展的大概趋势。
- 12 papers to understand QA system with Deep Learning
  - http://blog.csdn.net/abcjennifer/article/details/51232645
- Neural CRF
  - http://nlp.cs.berkeley.edu/pubs/Durrett-Klein_2015_NeuralCRF_paper.pd

---



