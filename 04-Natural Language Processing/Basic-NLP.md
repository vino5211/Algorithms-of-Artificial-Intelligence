# NLP
## Data
+　[相关NLP数据集](https://github.com/Apollo2Mars/View/blob/master/List-Data.md)

## Tmp Paper (未添加到下边的子分类中)
- QA相关资源/数据集/论文列表
	- https://blog.csdn.net/amds123/article/details/72758936
- 2017年值得读的NLP论文
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
- Neural CRF
    - http://nlp.cs.berkeley.edu/pubs/Durrett-Klein_2015_NeuralCRF_paper.pd
- ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
    - 第一种方法ABCNN0-1是在卷积前进行attention，通过attention矩阵计算出相应句对的attention feature map，然后连同原来的feature map一起输入到卷积层
    - 第二种方法ABCNN-2是在池化时进行attention，通过attention对卷积后的表达重新加权，然后再进行池化
    - 第三种就是把前两种方法一起用到CNN中

## Tmp Project and Tool (未添加到下边的子分类中)
- 语义分析
    - https://bosonnlp.com/
- NLPIR
    - https://github.com/NLPIR-team/NLPIR
- AllenNLP
- ParlAI
- OpenNMT
- MUSE
    - 多语言词向量 Python 库
    - 由 Facebook 开源的多语言词向量 Python 库，提供了基于 fastText 实现的多语言词向量和大规模高质量的双语词典，包括无监督和有监督两种。其中有监督方法使用双语词典或相同的字符串，无监督的方法不使用任何并行数据。
    - 无监督方法具体可参考 Word Translation without Parallel Data 这篇论文。
    - 论文链接：https://www.paperweekly.site/papers/1097
    - 项目链接：https://github.com/facebookresearch/MUSE
- FoolNLTK
    - 中文处理工具包
    - 特点：
        - 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
        - 基于 BiLSTM 模型训练而成
        - 包含分词，词性标注，实体识别，都有比较高的准确率
        - 用户自定义词典
    - 项目链接：https://github.com/rockyzhengwu/FoolNLTK 
- skorch
    - 兼容 Scikit-Learn 的 PyTorch 神经网络库
- FlashText
    - 关键字替换和抽取
- MatchZoo 
    - MatchZoo is a toolkit for text matching. It was developed to facilitate the designing, comparing, and sharing of deep text matching models.
- Sockeye: A Toolkit for Neural Machine Translation
    - 一个开源的产品级神经机器翻译框架，构建在 MXNet 平台上。
    - 论文链接：https://www.paperweekly.site/papers/1374**
    - 代码链接：https://github.com/awslabs/sockeye**

## Tasks
![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibt9rkyqib37KkCF45lBNmGXgc2QrxlrYtKxR8JPIWd4iaicPtQrcSWibmVodtGKttv91H6AwvJZGxbvA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)
- NLP 主要任务 ： 分类、匹配、翻译、结构化预测、与序贯决策过程
- 对于前四个任务，深度学习方法的表现优于或显著优于传统方法

### stem and lemma
+ 词干提取(stemming)和词型还原(lemmatization)
	+ 词形还原（lemmatization），是把一个任何形式的语言词汇还原为一般形式（能表达完整语义）
	+ 而词干提取（stemming）是抽取词的词干或词根形式（不一定能够表达完整语义）
    	```
    	# 词干提取(stemming) :基于规则
		from nltk.stem.porter import PorterStemmer
		porter_stemmer = PorterStemmer()
		porter_stemmer.stem('wolves')
        output is :'wolv'
        # 词性还原(lemmatization) : 基于字典，速度稍微慢一点
		from nltk.stem import WordNetLemmatizer
		lemmatizer = WordNetLemmatizer()
		lemmatizer.lemmatize('wolves')
        output is :'wolf'
        ```

### Grammer Model
- Deep AND-OR Grammar Networks for Visual Recognition
	- AOG 的全称叫 AND-OR graph，是一种语法模型（grammer model）。在人工智能的发展历程中，大体有两种解决办法：一种是自底向上，即目前非常流形的深度神经网络方法，另一种方法是自顶向下，语法模型可以认为是一种自顶向下的方法。
	- 把语法模型和深度神经网络模型结合起来，设计的模型同时兼顾特征的 exploration and exploitation（探索和利用），并在网络的深度和宽度上保持平衡；
	- 设计的网络结构，在分类任务和目标检测任务上，都比基于残差结构的方法要好。

### Dialog Systems
- Feudal Reinforcement Learning for Dialogue Management in Large Domains
	- 本文来自剑桥大学和 PolyAI，论文提出了一种新的强化学习方法来解决对话策略的优化问题
	- https://www.paperweekly.site/papers/1756
- End-to-End Optimization of Task-Oriented Dialogue Model with Deep Reinforcement Learning
	- 一篇基于强化学习的端到端对话系统研究工作，来自 CMU 和 Google。
	- 论文链接：http://www.paperweekly.site/papers/1257

### Reading and Comprehension (Information Retrieval, Question and Answering)
+ 【学习】QA相关资源/数据集/论文列表 (2017-5-25)
	+ https://blog.csdn.net/amds123/article/details/72758936
+ IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
	+ 在现代信息检索领域一直是两大学派之争的局面。一方面，经典思维流派是假设在文档和信息需求（由查询可知）之间存在着一个独立的随机生成过程。另一方面，现代思维流派则充分利用机器学习的优势，将文档和搜索词联合考虑为特征，并从大量训练数据中预测其相关性或排序顺序标签。
	+ 本篇 SIGIR2017 的满分论文则首次提出将两方面流派的数据模型通过一种对抗训练的方式统一在一起，使得两方面的模型能够相互提高，最终使得检索到的文档更加精准。文章的实验分别在网络搜索、推荐系统以及问答系统三个应用场景中实现并验证了结果的有效性。

- LSTM 中文
	- https://github.com/S-H-Y-GitHub/QA
	- 本项目通过建立双向长短期记忆网络模型，实现了在多个句子中找到给定问题的答案所在的句子这一功能。在使用了互联网第三方资源的前提下，用training.data中的数据训练得到的模型对develop.data进行验证，MRR可达0.75以上
	- MRR
		- 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和

+ `基于知识的智能问答技术　冯岩松
+ 基于深度学习的阅读理解　冯岩松
+ 机器这次击败人之后，争论一直没平息 | SQuAD风云
	+ https://zhuanlan.zhihu.com/p/33124445
+ **检索式问答系统的语义匹配模型（神经网络篇） **
	+ https://zhuanlan.zhihu.com/p/26879507
- 12 papers to understand QA system with Deep Learning
    - http://blog.csdn.net/abcjennifer/article/details/51232645
- A Question-Focused Multi-Factor Attention Network for Question Answering
	- https://www.paperweekly.site/papers/1597
	- https://github.com/nusnlp/amanda
- PaperWeekly 第37期 | 论文盘点：检索式问答系统的语义匹配模型（神经网络篇）
	- https://zhuanlan.zhihu.com/p/26879507
- Fast and Accurate Reading Comprehension by Combining Self-Attention and Convolution
	-《Fast and Accurate Reading Comprehension by Combining Self-attention and Convolution》阅读笔记
	 
	- 本文是 CMU 和 Google Brain 发表于 ICLR 2018 的文章，论文改变了以往机器阅读理解均使用 RNN 进行建模的习惯，使用卷积神经网络结合自注意力机制，完成机器阅读理解任务。
	- 其中作者假设，卷积神经网络可建模局部结构信息，而自注意力机制可建模全文互动（Interaction）关系，这两点就足以完成机器阅读理解任务。
	- 论文链接
	- https://www.paperweekly.site/papers/1759
- Attentive Recurrent Tensor Model for Community Question Answering
	- 社区问答有一个很主要的挑战就是句子间词汇与语义的鸿沟。本文使用了 phrase-level 和 token-level 两个层次的 attention 来对句子中的词赋予不同的权重，并参照 CNTN 模型用神经张量网络计算句子相似度的基础上，引入额外特征形成 3-way 交互张量相似度计算。
	- 围绕答案选择、最佳答案选择、答案触发三个任务，论文提出的模型 RTM 取得了多个 state-of-art 效果。
	- 论文链接 : https://www.paperweekly.site/papers/1741
 
- 百度 2018 机器阅读理解竞赛
- 搜狗 问答竞赛
- 机器阅读理解相关论文汇总（截止2017年底）
	- https://www.zybuluo.com/ShawnNg/note/622592
- 百度NLP团队登顶MARCO阅读理解测试排行榜
	- http://tech.qq.com/a/20180222/008569.htm
	- 使用了一种新的多候选文档联合建模表示方法，通过注意力机制使不同文档产生的答案之间能够产生交换信息，互相印证，从而更好的预测答案。据介绍，此次百度只凭借单模型（single model）就拿到了第一名，并没有提交更容易拿高分的多模型集成（ensemble）结果

### Knowledge base
- 「知识表示学习」专题论文推荐 | 每周论文清单
	- https://zhuanlan.zhihu.com/p/33606964
- 知识图谱与知识表征学习系列
	- https://zhuanlan.zhihu.com/p/27664263
- 怎么利用知识图谱构建智能问答系统？
    + https://www.zhihu.com/question/30789770/answer/116138035
    + https://zhuanlan.zhihu.com/p/25735572
+ 揭开知识库问答KB-QA的面纱1·简介篇
	+ 什么是知识库（knowledge base, KB）
	+ 什么是知识库问答（knowledge base question answering, KB-QA）
	+ 知识库问答的主流方法
	+ 知识库问答的数据集
- 经典论文解读 | 基于Freebase的问答研究
	- 本文给出了一种 end-to-end 的系统来自动将 NL 问题转换成 SPARQL 查询语言。
	- 作者综合了**实体识别**以及**距离监督**和 **learning-to-rank** 技术，使得 QA 系统的精度提高了不少，整个过程介绍比较详细，模型可靠接地气。
	- 本文要完成的任务是根据 **KB** 知识来回答自然语言问题，给出了一个叫 Aqqu 的系统，首先为问题生成一些备选 query，然后使用学习到的模型来对这些备选 query 进行排名，返回排名最高的 query
	- 论文 | More Accurate Question Answering on Freebase
	- 链接 | https://www.paperweekly.site/papers/1356
	- 源码 | https://github.com/ad-freiburg/aqqu
- Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks
	- 传统 QA 问题的解决方法是从知识库或者生文本中推测答案，本文将通用模式扩展到自然语言 QA 的应用当中，采用记忆网络来关注文本和 KB 相结合的大量事实。
	- 论文链接
		- https://www.paperweekly.site/papers/1734
	- 代码链接
		- https://github.com/rajarshd/TextKBQA

### NMT
- Machine Translation Using Semantic Web Technologies: A Survey
    - 本文是一篇综述文章，用知识图谱来解决机器翻译问题。
    - 论文链接：http://www.paperweekly.site/papers/1229

### Relation Extract
- Reinforcement Learning for Relation Classification from Noisy Data**
    - 将强度学习应用于关系抽取任务中，取得了不错的效果。本文已被 AAAI2018 录用。作者团队在上期 PhD Talk 中对本文做过在线解读。
        - 实录回顾：清华大学冯珺：基于强化学习的关系抽取和文本分类**
        - 论文链接：http://www.paperweekly.site/papers/1260

### Senmentic
- Recurrent Neural Networks for Semantic Instance Segmentation
    - 本项目提出了一个基于 RNN 的语义实例分割模型，为图像中的每个目标顺序地生成一对 mask 及其对应的类概率。该模型是可端对端 + 可训练的，不需要对输出进行任何后处理，因此相比其他依靠 object proposal 的方法更为简单。
    - 论文链接：https://www.paperweekly.site/papers/1355**
    - 代码链接：https://github.com/facebookresearch/InferSent

### Short Text Expand
- End-to-end Learning for Short Text Expansion
    - 本文第一次用了 end to end 模型来做 short text expansion 这个 task，方法上用了 memory network 来提升性能，在多个数据集上证明了方法的效果；Short text expansion 对很多问题都有帮助，所以这篇 paper 解决的问题是有意义的。
        - 通过在多个数据集上的实验证明了 model 的可靠性，设计的方法非常直观，很 intuitive。
        - 论文链接：https://www.paperweekly.site/papers/1313

### Sentiment
- Benchmarking Multimodal Sentiment Analysis
    - 多模态情感分析目前还有很多难点，该文提出了一个基于 CNN 的多模态融合框架，融合表情，语音，文本等信息做情感分析，情绪识别。
        - 论文链接：https://www.paperweekly.site/papers/1306
- Aspect Level Sentiment Classification with Deep Memory Network
    - 《Aspect Level Sentiment Classification with Deep Memory Network》阅读笔记
- Attention-based LSTM for Aspect-level Sentiment Classification
    - 《Attention-based LSTM for Aspect-level Sentiment Classification》阅读笔记

### Text Classification
- Hierarchical Attention Networks for Document Classification
    - Learning Structured Representation for Text Classification via Reinforcement Learning
    - 将强化学习应用于文本分类任务中，已被 AAAI2018录用。作者团队在上期 PhD Talk 中对本文做过在线解读。
        - 实录回顾：清华大学冯珺：基于强化学习的关系抽取和文本分类**
        - 论文链接：http://www.paperweekly.site/papers/1261

### Text Generation
- Adversarial Ranking for Language Generation
    - 本文提出了一种 RankGAN 模型，来解决如何生成高质量文本的问题。
    - 论文链接：https://www.paperweekly.site/papers/1290
- Neural Text Generation: A Practical Guide
    - 本文是一篇 Practical Guide，讲了很多用端到端方法来做文本生成问题时的细节问题和技巧，值得一看。

### Transfer 
- Fast.ai推出NLP最新迁移学习方法「微调语言模型」，可将误差减少超过20%！
    - Fine-tuned Language Models for Text Classification


---

## DL4NLP
- https://zhuanlan.zhihu.com/p/28710886
	- 信息抽取/
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
		-  《Detecting Semantically Equivalent Questions in Online User Forums》文中提出了一种采用卷积神经网络来识别语义等效性问题的方法
		-  《 Paraphrase Detection Using Recursive Autoencoder》文中提出了使用递归自动编码器的进行释义检测的一种新型的递归自动编码器架构。
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

## DRL4NLP
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

# ViewPoint
+ https://www.wxwenku.com/d/100329482
	- 但是由于语言本身已经是一种高层次的表达，深度学习在 NLP 中取得的成绩并不如在视觉领域那样突出。尤其是在 NLP 的底层任务中，基于深度学习的算法在正确率上的提升并没有非常巨大，但是速度却要慢许多，这对于很多对 NLP 来说堪称基础的任务来说，是不太能够被接受的，比如说分词
	- 在完形填空类型的阅读理解（cloze-style machine reading comprehension）上，基于 attention 的模型也取得了非常巨大的突破（在 SQuAD 数据集上，2016 年 8 月的 Exact Match 最好成绩只有 60%，今年 3 月已经接近 77%，半年时间提升了接近 20 个点，这是极其罕见的）
	- 深度学习的不可解释的特性和对于数据的需求，也使得它尚未在要求更高的任务上取得突破，比如对话系统（虽然对话在 2016 年随着 Echo 的成功已经被炒得火热）
	- 在大多数端到端的 NLP 应用中，在输入中包括一些语言学的特征（例如 pos tag 或 dependency tree）并不会对结果有重大影响。我们的一些粗浅的猜测，是因为目前的 NLP 做的这些特征，其实对于语义的表示都还比较差，某种程度来说所含信息还不如 word embedding 来的多
	- 关于阅读理解（Open-domain QA）
		- 幸好 Stanford 的 Chen Danqi 大神的 Reading Wikipedia to Answer Open-Domain Questions 打开了很多的方向。通过海量阅读（「machine reading at scale」），这篇文章试图回答所有在 wikipedia 上出现的 factoid 问题。其中有大量的工程细节，在此不表，仅致敬意。
- ACL 2016：基于深度学习的 NLP 看点（十大优秀论文下载）
	- 2016年NLP深度学习技术的发展趋势
		- 深度学习模型在更多NLP任务上的定制化应用。例如将过去统计机器翻译的成熟成果迁移到神经网络模型上，基于深度学习的情感分析，再例如今年NAACL 2016的最佳论文Feuding Families and Former Friends; Unsupervised Learning for Dynamic Fictional Relationships也利用神经网络模型检测小说中的人物关系。
		- 带有隐变量的神经网络模型。很多NLP任务传统主要基于HMM、CRF方法对标注标签的关联关系建模，而单纯的神经网络模型并不具备这个能力，因此一个重要热点将是在神经网络模型中引入隐变量，增强神经网络的建模能力。
		- 注意力（attention）机制的广泛应用。大量工作已经证明attention机制在文本产生中的重要性，也是继CNN->RNN->LSTM之后的新的论文增长点，相信在2016年会有大量论文提出各种带有attention的神经网络模型。
	- 如何将先验知识引入分布式表示
		- 分布式表示（distributed representation）是深度学习的重要特点；避免特征工程的端对端（End-to-End）框架则是深度学习在NLP的独特优势。然而，现实世界中我们拥有大量人工标注的语言知识库和世界知识库，如何在深度学习框架中引入这些先验知识，是未来的重要挑战性问题，也是极大拓展深度学习能力的重要途径。在这个方面，有很多颇有创见的探索工作，例如来自香港华为Noah实验室Zhengdong Lu团队的Neural Enquirer: Learning to Query Tables [1]，等等。此外，我认为基于深度学习的attention机制也是引入先验知识的重要可能手段。机器学习领域还提供了很多其他可能的手段，等待我们去探索。
	- 探索人类举一反三能力的One-Shot Learning
		- 如2015年在Science发表的轰动论文[2]所述，人类学习机制与目前深度学习的显著差异在于，深度学习利用需要借助大量训练数据才能实现其强大威力，而人类却能仅通过有限样例就能学习到新的概念和类别，这种举一反三的学习机制，是机器学习也是自然语言处理梦寐以求的能力。这需要我们特别关注认知领域的相关进展[3, 4]，机器学习领域也在热切探索one-shot learning任务。在NLP领域，如何应对新词、新短语、新知识、新用法、新类别，都将与该能力密切相关。
	- 从文本理解到文本生成的飞跃
		- 目前取得重要成果的NLP任务大多在文本理解范畴，如文本分类，情感分类，机器翻译，文档摘要，阅读理解等。这些任务大多是对已有文本的“消费”。自然语言处理的飞跃，需要实现从“消费”到“生产”的飞跃，即探索如何由智能机器自动产生新的有用文本。虽然现在有媒体宣称实现了新闻的自动生成，但从技术上并无太多高深之处，更多是给定数据后，对既有新闻模板的自动填充，无论是从可扩展性还是智能性而言，都乏善可陈。我认为，自然语言处理即将面临的一个飞跃，就是智能机器可以汇总和归纳给定数据和信息，自动产生符合相关标准的文本，例如新闻、专利、百科词条[5]、论文的自动生成，以及智能人机对话系统等等。毫无疑问，这个技术飞跃带来的应用拥有无限的想象空间。
	- 大规模知识图谱的构建与应用
		- “知识图谱”是谷歌推出的产品名，现在已经成为对大规模知识库的通用说法。如果说深度学习是机器大脑的学习机制，那么知识图谱可以看做机器大脑的知识库。知识图谱是问答系统的重要信息来源，也是阅读理解、机器翻译、文档摘要等任务进一步发展的重要支撑。目前，知识图谱从构建到应用都仍有很多问题亟待解决，例如新概念、新知识的自动学习，如何基于知识图谱实现智能推理，等等。在这方面，我一直关注知识的分布式表示学习，能够建立统一的语义表示空间，有效解决大规模知识图谱的数据稀疏问题，有望在知识获取、融合和推理方面发挥重要作用[6]。

- 一文概述2017年深度学习NLP重大进展与趋势
    		- http://www.qingpingshan.com/bc/jsp/361202.html
- 李航NSR论文：深度学习NLP的现有优势与未来挑战
	- Deep Learning for Natural Language Processing: Advantages and Challenges
- pass
	- 端到端训练与表征学习是深度学习的核心特征，这使其成为 NLP 的强大工具。但深度学习并非万能，它在对解决**多轮对话**等复杂任务异常关键的**推断和决策**上表现欠佳。此外，如何结合符号处理与神经处理、如何应对长尾现象等问题依然是深度学习 NLP 面临的挑战[1]。
	- 自然语言处理领域有很多复杂任务，这些任务可能无法仅使用深度学习来轻松完成。例如，多轮对话是一个非常复杂的过程，涉及语言理解、语言生成、对话管理、知识库访问和推断。对话管理可以正式作为序贯决策过程，其中强化学习发挥关键作用。很明显，把深度学习和强化学习结合起来可能有利于完成任务。
	- 总之，深度学习 NLP 仍然面临许多待解决的挑战。深度学习与其他技术（强化学习、推断、知识）结合起来将会进一步扩展 NLP 的边界[1]。

