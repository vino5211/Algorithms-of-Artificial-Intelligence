## Reference links
+ 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
+ https://zhuanlan.zhihu.com/p/33124445

# Dataset
## MS MARCO
+ 相比SQuAD，MARCO的挑战难度更大，因为它需要测试者提交的模型具备理解复杂文档、回答复杂问题的能力。
+ 据了解，对于每一个问题，MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到。这对机器阅读理解提出了更高的要求，需要机器具备综合理解多文档信息、聚合生成问题答案的能力。
## NarrativeQA
+ Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
	+ https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
	+ https://github.com/deepmind/narrativeqa
	+ DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。

+ The NarrativeQA Reading Comprehension Challenge
	+ 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
	+ 论文链接：https://www.paperweekly.site/papers/1397
	+ 代码链接：https://github.com/deepmind/narrativeqa

## SQuAD
+ 这个竞赛基于SQuAD问答数据集，考察两个指标：EM和F1。
	+ EM是指精确匹配，也就是模型给出的答案与标准答案一模一样；F1，是根据模型给出的答案和标准答案之间的重合度计算出来的，也就是结合了召回率和精确率。
	+ 目前阿里、微软团队并列第一，其中EM得分微软（r-net+融合模型）更高，F1得分阿里（SLQA+融合模型）更高。但是他们在EM成绩上都击败了“人类表现”
+ EMNLP2016 SQuAD:100,000+ Questions for Machine Comprehension of Text
	+ https://arxiv.org/pdf/1606.05250.pdf
+ SQuAD，斯坦福在自然语言处理的野心
	+ http://blog.csdn.net/jdbc/article/details/52514050
+ 一共有107,785问题，以及配套的 536 篇文章
+ 数据集的具体构建如下：
	1. 文章是随机sample的wiki百科，一共有536篇wiki被选中。而每篇wiki，会被切成段落，最终生成了23215个自然段。之后就对这23215个自然段进行阅读理解，或者说自动问答。
	2. 之后斯坦福，利用众包的方式，进行了给定文章，提问题并给答案的人工标注。他们将这两万多个段落给不同人，要求对每个段落提五个问题。
	3. 让另一些人对提的这个问题用文中最短的片段给予答案，如果不会或者答案没有在文章中出现可以不给。之后经过他们的验证，人们所提的问题在问题类型分布上足够多样，并且有很多需要推理的问题，也就意味着这个集合十分有难度。如下图所示，作者列出了该数据集答案的类别分布，我们可以看到 日期，人名，地点，数字等都被囊括，且比例相当。
	4. 这个数据集的评测标准有两个：
		第一：F1
		第二：EM。
		EM是完全匹配的缩写，必须机器给出的和人给出的一样才算正确。哪怕有一个字母不一样，也会算错。而F1是将答案的短语切成词，和人的答案一起算recall，Precision和F1，即如果你match了一些词但不全对，仍然算分。
	5. 为了这个数据集，他们还做了一个baseline，是通过提特征，用LR算法将特征组合，最终达到了40.4的em和51的f1。而现在IBM和新加坡管理大学利用深度学习模型，均突破了这个算法。可以想见，在不远的将来会有更多人对阅读理解发起挑战，自然语言的英雄也必将诞生。甚至会有算法超过人的准确度。

## 对比
- 当前的公开数据集对比如下，MCTest，Algebra和Science是现在的三个公开的阅读理解数据集，
- 我们可以看到Squad在数量上远远超过这三个数据集，这使得在这个数据集上训练大规模复杂算法成为可能。
- 同时，相比于WikiQA和TrecQA这两个著名问答数据集，Squad也在数量上远远超过。
- 而CNN Mail和CBT虽然大，但是这两个数据集都是挖空猜词的数据集，并不是真正意义上的问答。

+ ## 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
1. On Generating Characteristic-rich Question Sets for QA Evaluation
	文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。
	+ 数据集特点：
		1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
		2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
		3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
		4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
		5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。

	+ 论文链接：http://www.paperweekly.site/papers/906
	+ 数据集链接：https://github.com/ysu1989/GraphQuestions
2. LSDSem 2017 Shared Task: The Story Cloze Test
	+ Story Cloze Test：人工合成的完形填空数据集。
	+ 论文链接：http://www.paperweekly.site/papers/917
	+ 数据集链接：http://cs.rochester.edu/nlp/rocstories/
3. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering
	+ 百度深度学习实验室创建的中文开放域事实型问答数据集。
	+ 论文链接：http://www.paperweekly.site/papers/914
	+ 数据集链接：http://idl.baidu.com/WebQA.html
4. Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems
	+ DeepMind 和牛津大学共同打造的代数问题数据集 AQuA（Algebra Question Answering）。
	+ 论文链接：http://www.paperweekly.site/papers/913
	+ 数据集链接：https://github.com/deepmind/AQuA
5. Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems
	+ Maluuba 放出的对话数据集。
	+ 论文链接：http://www.paperweekly.site/papers/407
	+ 数据集链接：http://datasets.maluuba.com/Frames
6. Teaching Machines to Read and Comprehend
	+ DeepMind Q&A Dataset 是一个经典的机器阅读理解数据集，分为两个部分：
		+ 1. CNN：~90k 美国有线电视新闻网（CNN）的新闻文章，~380k 问题；
		+ 2. Daily Mail：~197k DailyMail 新闻网的新闻文章（不是邮件正文），~879k 问题。
	+ 论文链接：http://www.paperweekly.site/papers/915
	+ 数据集链接：http://cs.nyu.edu/~kcho/DMQA/
7. Semantic Parsing on Freebase from Question-Answer Pairs
	+ 文章发表在 EMNLP-13，The Stanford NLP Group 是世界领先的 NLP 团队。
	+ 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的
	+ 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣）
	+ 有 6642 个问答对
	+ 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system。
	+ 论文链接：http://www.paperweekly.site/papers/827
	+ 数据集链接：http://t.cn/RWPdQQO
8. A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories
	+ ROCStories dataset for story cloze test.
	+ 论文链接：http://www.paperweekly.site/papers/918
	+ 数据集链接：http://cs.rochester.edu/nlp/rocstories/
9. MoleculeNet: A Benchmark for Molecular Machine Learning
	+ 一个分子机器学习 benchmark，最喜欢看到这种将机器学习应用到传统学科领域了。
	+ 论文链接：http://www.paperweekly.site/papers/862
	+ 数据集链接：http://t.cn/RWPda8r

---

## Current QA Plan
### 语义分析类方案
	+ 语义分析
		+ 利用形式化方法表达问题语义
	+ 语义表示
		+ $\lambda$-Calculus
		+ $\lambda$-DCSs
		+ CCG
		+ Simple Query Graph
		+ Query Graph
		+ Phrase Dependency Graph
	+ 基于状态转移系统的解析器
	+ 联合消解
	+ 短语依存图和知识库映射
	+ 与知识库的Grounding
	+ Grounding


+ ### 信息抽取类方案
	+ IEQA


+ ### 基于深度学习的解决方案
	+ 对现有模块的改进
		+ 关系抽取
		+ 候选评分
	+ Neural End-to-End 框架
		+ 多数遵循信息抽取类框架
		+ Embedding Everything
		+ Memory Networks
    + 对现有模块的改进
        + STAGG
            + Staged Query Graph Generation
            + 通过搜索，逐步构建查询图（Query Graph）
        + Linking Topic Entity
        + Identifying Core Inferential Chain
        + Argument Constraints
        + Learning
        + 知识问答中的关系抽取(+++)
            + 神经网络模型
            + 依存关系结构
            + Multi-Channel Convolutional Neural Networks
            + 关系抽取的结果
                + SemEval-2010 Task 8
                + WebQuestions
        + 改进候选评分模块
    + Neural End-to-End 框架
        + End-to-End
        + Simple Matching
        + Multi-Column CNNs
        + Attention + Global Knowledge
        + Memory Networks
        + Key-Value Memory Networks
        + Neural Symbolic Machines
    + 新的应用场景
        + 生产自然语言回复
            + 输入：事实类自然语言问题
            + 输出：生成自然语言回答
        + COREQA
        + GenQA
        	+ GenQA: Automated Addition of Architectural Quality Attribute Support for Java Software
        	+ http://selab.csuohio.edu/~nsridhar/research/Papers/PDF/genqa.pdf
    + 基于实体关系的问答技术
    + 单独依靠知识库是不够的
        + 实体与关系的联合消解
        + 其他文本的作用：利用维基正文清洗候选答案
        + Hybrid-QA:基于混合资源的知识库问
+ ### SQuAD
	+ Hybrid AoA Reader (ensemble)
		+ Joint Laboratory of HIT and iFLYTEK Research
	+ r-net + 融合模型
		+ Microsoft Research Asia
	+ SLQA + 融合模型
		+ Alibaba iDST NLP
				
---		

## 检索式问答系统的语义匹配模型（神经网络篇） 
+ https://zhuanlan.zhihu.com/p/26879507
+ 实现方式
	+ 问答系统可以基于规则实现
	+ 可以基于检索实现
	+ 还可以通过对 query 进行解析或语义编码来生成候选回复
	+ 如通过解析 query并查询知识库后生成，或通过 SMT 模型生成，或通过 encoder-decoder 框架生成，有些 QA 场景可能还需要逻辑推理才能生成回复
+ 检索式问答系统典型场景
	+ 1）候选集先离线建好索引；
	+ 2）在线服务收到 query 后，初步召回一批候选回复；
	+ 3）matching 和 ranking 模型对候选列表做 rerank 并返回 top K。
+ NN 实现语义匹配的典型工作
1. Po-Sen Huang, et al., 2013, Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
+ 相似性问题： 计算 Quary 和 Doc 的相似性
+ **这篇博客讲的很好： https://cloud.tencent.com/developer/article/1005600**
+ From UIUC 和 Microsoft Research
+ 针对搜索引擎 query/document 之间的语义匹配问题 ，提出了基于 MLP 对 query 和 document 做深度语义表示的模型（Deep Structured SemanticModels, DSSM）
+ structure : 
	![](https://pic1.zhimg.com/80/v2-0187cc3483ec2a2f88453576eef61cc5_hd.jpg)
+ Step
	+ 先把 query 和 document 转换成 BOW 向量形式
	+ 然后通过 word hashing 变换做降维得到相对低维的向量（备注：除了降维，word hashing 还可以很大程度上解决单词形态和 OOV 对匹配效果的影响）
		+ word hashing
			+ http://blog.csdn.net/washiwxm/article/details/19838595
			+ 举个例子，假设用 letter-trigams 来切分单词（3 个字母为一组，#表示开始和结束符），boy 这个单词会被切为 #-b-o, b-o-y, o-y-#
			+ 这样做的好处有两个：首先是压缩空间，50 万个词的 one-hot 向量空间可以通过 letter-trigram 压缩为一个 3 万维(27*27*27)的向量空间。其次是增强范化能力，三个字母的表达往往能代表英文中的前缀和后缀，而前缀后缀往往具有通用的语义。
			+ 选择三字母的原因
	+ 喂给 MLP 网络，输出层对应的低维向量就是 query 和 document 的语义向量（假定为 Q 和 D）
	+ 计算(D, Q)的 cosinesimilarity 后
	+ 用 softmax 做归一化得到的概率值是整个模型的最终输出，该值作为监督信号进行有监督训练

2. Yelong Shen, et al, 2014, A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval
+ 这篇文章出自 Microsoft Research，是对上述 DSSM 模型的改进工作
+ 在 DSSM 模型中，输入层是文本的 bag-of-words 向量，**丢失词序特征，无法捕捉前后词的上下文信息**
+ 基于此，本文提出一种基于卷积的隐语义模型（convolutional latent semantic model, CLSM）
+ Structure :
	![](https://pic1.zhimg.com/80/v2-29ffcff7590aea70e85df0deb3d71abe_hd.jpg)
+ Step:
	+ 先用滑窗构造出 query 或 document 的一系列 n-gram terms（图中是 trigram），
	+ 然后通过 word hashing 变换将 word trigram terms 表示成对应的 letter-trigram 向量形式（主要目的是降维）
	+ 接着对每个 letter-trigram 向量做卷积，由此得到「Word-n-gram-Level Contextual Features」
	+ 接着借助 max pooling 层得到「Sentence-Level Semantic Features」
	+ 最后对 max pooling 的输出做 tanh 变换，得到一个固定维度的向量作为文本的隐语义向量
	+ Query 和 document 借助 CLSM 模型得到各自的语义向量后，构造损失函数做监督训练
	+ 训练样本同样是通过挖掘搜索点击日志来生成
+ Experiment:
	+ BM25、PLSA、LDA、DSSM
	+ NDCG@N 指标表明，CLSM 模型在语义匹配上达到了新的 SOTA 水平
	+ 文中的实验和结果分析详细且清晰，很赞的工作
3. Zhengdong Lu & Hang Li, 2013, A Deep Architecture for Matching Short Texts
+ From : 这篇文章出自华为诺亚方舟实验室
+ Scenario : 针对短文本匹配问题
+ 提出一个被称为 DeepMatch 的神经网络语义匹配模型
+ 该模型的提出基于文本匹配过程的两个直觉：
	+ 1）Localness，也即，两个语义相关的文本应该存在词级别的共现模式（co-ouccurence pattern of words）；
	+ 2）Hierarchy，也即，共现模式可能在不同的词抽象层次中出现。
4. Zongcheng Ji, et al., 2014, An Information Retrieval Approach to Short Text Conversation
+ From : 这篇文章出自华为诺亚方舟实验室
+ Scenario : 针对的问题是基于检索的短文本对话，但也可以看做是基于**检索的问答系统**
+ Step:
	+ 主要思路是，从不同角度构造 matching 特征，作为 ranking 模型的特征输入。
	+ 构造的特征包括：
		+ 1）Query-ResponseSimilarity；
		+ 2）Query-Post Similarity；
		+ 3）Query-Response Matching in Latent Space；
		+ 4）Translation-based Language Model；
		+ 5）Deep MatchingModel；
		+ 6）Topic-Word Model；
			+ 7）其它匹配特征。
5. Baotian Hu, et al., 2015, Convolutional Neural Network Architectures for Matching Natural Language Sentences
+ From : 华为诺亚方舟实验室
+ 采用 CNN 模型来解决语义匹配问题，文中提出 2 种网络架构，分别为 ARC-I 和 ARC-II
+ ARC-I
	+ Structure
	![](https://pic1.zhimg.com/80/v2-bf0d6e2b0040fa995b1d3cadf3b8bb56_hd.jpg)
	+ Step :
		+ 上图所示的 ARC-I 比较直观，待匹配文本 X 和 Y 经过多次一维卷积和 MAX 池化，得到的固定维度向量被当做文本的隐语义向量，
		+ 这两个向量继续输入到符合 Siamese 网络架构的 MLP 层，最终得到文本的相似度分数。
		+ 需要说明的是，MAX POOLING 层在由同一个卷积核得到的 feature maps 之间进行两两 MAX 池化操作，起到进一步降维的作用。
		+ 作者认为 ARC-I 的监督信号在最后的输出层才出现，**在这之前，X 和 Y 的隐语义向量相互独立生成，可能会丢失语义相关信息，于是提出 ARC-II 架构**。
+ ARC-II
	+ (to be continued)
6. Lei Yu, et al., 2014, Deep Learning for Answer Sentence Selection
+ From : University of Oxford 和 DeepMind
+ 提出基于 unigram 和 bigram 的语义匹配模型
+ Step :
	+ unigram :
		+ 其中，unigram 模型通过累加句中所有词（去掉停用词）的 word vector，
		+ 然后求均值得到句子的语义向量；
	+ bigram :
		+ bigram 模型则先构造句子的 word embedding 矩阵，
		+ 接着用 bigram 窗口对输入矩阵做 1D 卷积，
		+ 然后做 average 池化，
		+ 用 n 个 bigram 卷积核对输入矩阵分别做「1D 卷积+average 池化」后，会得到一个 n 维向量，作为文本的语义向量
	+ 对(question,answer)文本分别用上述 bigram 模型生成语义向量后，计算其语义相似度并用 sigmoid 变换成 0~1 的概率值作为最终的 matching score。该 score 可作为监督信号训练模型。
	+ Structure
	![](https://pic3.zhimg.com/80/v2-cd4c9f238689d0412754b3761b84a6af_hd.jpg)
+ 文中用 TREC QA 数据集测试了提出的 2 个模型，实验结果的 MAP 和 MRR 指标表明，unigram 和 bigram 模型都有不错的语义匹配效果，其中 bigram 模型要优于 unigram 模型。
+ 特别地，在语义向量基础上融入 idf-weighted word co-occurence count 特征后，语义匹配效果会得到明显提升。文中还将提出的 unigram 和 bigram 模型与几个已有模型进行了效果对比，结果表明在同样的数据集上，融入共现词特征的 bigram 模型达到了 SOTA 的效果。
7. Aliaksei Severyn, et al., 2015, Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks
8. Ryan Lowe, et al., 2016, The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems
+ From : McGill 和 Montreal 两所大学
+ Scenario : 针对基于检索的多轮对话问题，提出了 **dual-encoder** 模型对 context 和 response 进行语义表示，该思路也可用于检索式问答系统
+ Structure :
![](https://pic1.zhimg.com/80/v2-c4be342ff33fbefc8d0953a7d7bfd1ed_hd.jpg)
+ Step :
	+ 通过对偶的 RNN 模型分别把 context 和 response 编码成语义向量，
	+ 然后通过 M 矩阵变换计算语义相似度，
	+ 相似度得分作为监督信号在标注数据集上训练模型。
		+ 文中在 Ubuntu 对话语料库上的实验结果表明，dual-encoder 模型在捕捉文本语义相似度上的效果相当不错。
9. 从上面 8 篇论文可知，与关键词匹配（如 TF-IDF 和 BM25）和浅层语义匹配（如隐语义模型，词向量直接累加构造的句向量）相比，基于深度学习的文本语义匹配模型在问答系统的匹配效果上有明显提升。
