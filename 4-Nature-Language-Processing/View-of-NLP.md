# View of NLP

- Speech and Language Processing(3rd ed. draft)
- http://www.deeplearningbook.org/

- Meka
	- 多标签分类器和评价器
	- MEKA 是一个基于 Weka 机器学习框架的多标签分类器和评价器。本项目提供了一系列开源实现方法用于解决多标签学习和评估。
- Quick NLP
	- Quick NLP 是一个基于深度学习的自然语言处理库，该项目的灵感来源于 Fast.ai 系列课程。它具备和 Fast.ai 同样的接口，并对其进行扩展，使各类 NLP 模型能够更为快速简单地运行。

### CoNLL X
- CoNLL是一个由SIGNLL(ACL's Special Interest Group on Natural Language Learning: 计算语言学协会的自然语言学习特别兴趣小组）组织的顶级会议。CoNLL X (如CoNLL 2006)是它定义的语言学数据格式

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
        
### Transfer 
- Fast.ai推出NLP最新迁移学习方法「微调语言模型」，可将误差减少超过20%！
    - Fine-tuned Language Models for Text Classification

### oxford-cs-deepnlp
- https://github.com/oxford-cs-deepnlp-2017/lectures
- http://study.163.com/course/introduction/1004336028.htm





# View of DL4NLP

- https://zhuanlan.zhihu.com/p/28710886
  - 信息抽取
  - NER
    - 命名实体识别（NER）的主要任务是将诸如Guido van Rossum，Microsoft，London等的命名实体分类为人员，组织，地点，时间，日期等预定类别。许多NER系统已经创建，其中最好系统采用的是神经网络。
    - 在《Neural Architectures for Named Entity Recognition》文章中，提出了两种用于NER模型。这些模型采用有监督的语料学习字符的表示，或者从无标记的语料库中学习无监督的词汇表达[4]。使用英语，荷兰语，德语和西班牙语等不同数据集，如CoNLL-2002和CoNLL-2003进行了大量测试。该小组最终得出结论，如果没有任何特定语言的知识或资源（如地名词典），他们的模型在NER中取得最好的成绩。
  - POS
    - 在《Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network》工作中，提出了一个采用RNN进行词性标注的系统[5]。该模型采用《Wall Street Journal data from Penn Treebank III》数据集进行了测试，并获得了97.40％的标记准确性。
  - CLF
    - Siwei Lai, Liheng Xu, Kang Liu, and Jun Zhao在论文《Recurrent Convolutional Neural Networks for Text Classification》中，提出了一种用于文本分类的循环卷积神经网络，该模型没有人为设计的特征。该团队在四个数据集测试了他们模型的效果，四个数据集包括：20Newsgroup（有四类，计算机，政治，娱乐和宗教），复旦大学集（中国的文档分类集合，包括20类，如艺术，教育和能源），ACL选集网（有五种语言：英文，日文，德文，中文和法文）和Sentiment Treebank数据集（包含非常负面，负面，中性，正面和非常正面的标签的数据集）。测试后，将模型与现有的文本分类方法进行比较，如Bag of Words，Bigrams + LR，SVM，LDA，Tree Kernels，RecursiveNN和CNN。最后发现，在所有四个数据集中，神经网络方法优于传统方法，他们所提出的模型效果优于CNN和循环神经网络。
  - 语义分析和问题回答
    - 问题回答系统可以自动回答通过自然语言描述的不同类型的问题，包括定义问题，传记问题，多语言问题等。神经网络可以用于开发高性能的问答系统。
    - 在《Semantic Parsing via Staged Query Graph Generation Question Answering with Knowledge Base》文章中，Wen-tau Yih, Ming-Wei Chang, Xiaodong He, and Jianfeng Gao描述了基于知识库来开发问答语义解析系统的框架框架。作者说他们的方法早期使用知识库来修剪搜索空间，从而简化了语义匹配问题[6]。他们还应用高级实体链接系统和一个用于匹配问题和预测序列的深卷积神经网络模型。该模型在WebQuestions数据集上进行了测试，其性能优于以前的方法。
  - 释义检测:  释义检测确定两个句子是否具有相同的含义。
    - 《Detecting Semantically Equivalent Questions in Online User Forums》文中提出了一种采用卷积神经网络来识别语义等效性问题的方法
    - 《 Paraphrase Detection Using Recursive Autoencoder》文中提出了使用递归自动编码器的进行释义检测的一种新型的递归自动编码器架构。
  - 语言生成和多文档总结
    - 《 Natural Language Generation, Paraphrasing and Summarization of User Reviews with Recurrent Neural Networks》中，描述了基于循环神经网络（RNN）模型，能够生成新句子和文档摘要的。
  - 机器翻译
  - 语音识别
    - 在《Convolutional Neural Networks for Speech Recognition》文章中，科学家以新颖的方式解释了如何将CNN应用于语音识别，使CNN的结构直接适应了一些类型的语音变化，如变化的语速
  - 字符识别
    - 字符识别系统具有许多应用，如收据字符识别，发票字符识别，检查字符识别，合法开票凭证字符识别等。文章《Character Recognition Using Neural Network》提出了一种具有85％精度的手写字符的方法
  - 拼写检查
    - 大多数文本编辑器可以让用户检查其文本是否包含拼写错误。神经网络现在也被并入拼写检查工具中。
    - 在《Personalized Spell Checking using Neural Networks》，作者提出了一种用于检测拼写错误的单词的新系统。

### DRL4NLP

- https://github.com/ganeshjawahar/drl4nlp.scratchpad
  - Policy Gradients
    - buck_arxiv17: Ask the Right Questions: Active Question Reformulation with Reinforcement Learning [arXiv]
    - dhingra_acl17: Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access [arXiv] [code]
    - paulus_arxiv17: A Deep Reinforced Model for Abstractive Summarization [arXiv]
    - nogueira_arxiv17: Task-Oriented Query Reformulation with Reinforcement Learning [arXiv] [code]
    - li_iclr17: Dialog Learning with Human-in-the-loop [arXiv] [code]
    - li_iclr17_2: Learning through dialogue interactions by asking questions [arXiv] [code]
    - yogatama_iclr17: Learning to Compose Words into Sentences with Reinforcement Learning [arXiv]
    - dinu_nips16w: Reinforcement Learning for Transition-Based Mention Detection [arXiv]
    - clark_emnlp16: Deep Reinforcement Learning for Mention-Ranking Coreference models [arXiv] [code]
  - Value Function
    - narasimhan_emnlp16: Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning [arXiv] [code]
  - Misc
    - bordes_iclr17: Learning End-to-End Goal-Oriented Dialog [arXiv]
    - weston_nips16: Dialog-based Language Learning [arXiv] [code]
    - nogueira_nips16: End-to-End Goal-Driven Web Navigation [arXiv] [code]

