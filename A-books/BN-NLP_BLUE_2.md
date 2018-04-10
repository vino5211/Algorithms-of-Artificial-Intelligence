# 基于知识的智能问答技术
## 背景
+ 任务：利用知识回答自然语言问题
	+ 输入：自然语言语句
	+ 资源：结构化知识库，文本知识，表格，结构化/半结构化记录
	+ 输出：答案
+ 大规模知识库
	+ wikidata : 66G 下载链接(+++)
	+ Others:
		+ Free917
		+ WebQuestions
		+ QALD
		+ Simple Questions
+ 现有技术
	+ 语义分析(SP)
	+ 信息抽取(IE)
+ 技术挑战
	+ 如何更恰当的表示语义
		+ 语义表示/语义分析
	+ 如何利用(大规模)(开放式)知识库来表示问题的语义
		+ 知识库映射
		+ 实体连接/关系抽取
	+ 需要什么样的知识来解答
		+ 知识融合

## 传统解决方案
+ ### 语义分析类方案
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

## 基于深度学习的解决方案
+ 提纲
	+ 背景
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
	+ GenQA
	+ COREQA
+ 基于实体关系的问答技术
+ 单独依靠知识库是不够的
	+ 实体与关系的联合消解
	+ 其他文本的作用：利用维基正文清洗候选答案
	+ Hybrid-QA:基于混合资源的知识库问
---
# 基于深度学习的阅读理解
## 背景
+ 核心:检验机器是否能恰当的处理文档，从不同侧面理解并回答问题(change)
+ 形式：给定文档作为输入，根据文档回答问题
	+ 文档形式
		+ 新闻(get by dataEngine)
		+ Wiki(wikidata, download from office site)
	+ 问题形式
		+ 选择题
		+ 字符串:找到字符串
		+ 完形填空
	+ 挑战
	+ 典型数据集
		|名称|描述|类型|规模|创建者|
        |---|
		|MCTest|
        |bAbi|
        |CNN/DailyMail|新闻|完形填空|93K/220K|DeepMind|
        |SQuAD|维基百科|问答|10W|Standford|
        |bAbi|
        |MCTest|
    	+ bAbi
    		+ 由Facebook创建，验证实现语言理解所需的推理能力
    		+ 主要是虚拟场景下的动作情节
    		+ 训练1000个问题，测试1000个问题
    		+ 答案：一个词或几个词
    	+ MCTest
    		+ 由MSR构建(Microsoft Resrerch)
    		+ http://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/
    		+ 短片小说 + 选择题
    			+ 共650篇
    			+ 单选+多选/盲选
    		+ 利用众包平台进行标注
    			+ Amazon Mechanical Turk
    	+ SQuAD
    		+ 由斯坦福大学从维基百科中构建(LeaderBorad)
    + 传统方法
    	+ 典型的两步框架
    	+ 片段检索
    		+ P(片段|问题，文档)
    	+ 答案生成
    		+ 通常定义为文本蕴含(???)
    		+ P(答案|问题，片段)
    	+ 最终
    		+ P(片段|问题，文档)×P(答案|问题，片段)、
    		+ 首先确定片段，在确定片段中寻找答案
    + 特征
    	+ 挖掘隐性文本蕴含的特征，如词级别的对应特征
    		+ Sachan,et al.ACL 2015
    + 传统方法蕴含的困难
    	+ 对**篇章**理解建模能力有限
    	+ 对深层次的推理需求无能为力
    	+ 外部资源和工具带来的错误传递和积累
    
##深度学习
+ 方法
	+ LSTM
	+ Attention Mechanism
	+ Memory Network
	+ Hierarchical CNN
	+ Hierarchical Attention
	+ Pointer Network
	+ Attention Over Attention
	+ Self-Matching Network
+ LSTM
	+ LSTM 对较长词串具有抽象(浓缩)能力
	+ 缺点：对距离较远的关键词缺乏足够的关联建模
+ LSTM + Attention
	+ Attentive Reader
	+ 双向LSTM + Attention
	+ 找到最有支持度的句子
+ LSTM + Attentions
	+ Impatient Reader
	+ 在处理文档中的单词时，通过注意力机制令模型能重新阅读文档句子
	+ 逐步处理问题，反复阅读句子，产生更好的文档表示
	+ 例子在CNN/DialyMail
+ **Memory** Networks
	+ I(Input feature map):将输入转化为**内部特征**表示
	+ G(Generalization):根据输入**更新**当前Memory
	+ O(Oputput feature map):根据**输入**和**当前Memory状态**，生成**输出向量**
	+ R(Response):根据**输出向量**，产生答案
	+ 例子：Memory Network for bAbi
	+ 改进：
		+ 自适应记忆单元
		+ 记忆单元使用Ngram
		+ 匹配函数非线性化
		+ End2End MN
+ Attention over Attention
+ Match LSTM
+ Bi-direction Attention Flow
+ DocRetriever-DocReader
	+ Open-domain QA
+ Mnemonic Reader
+ R-Net

---
# Part I : Network Embedding: Recent Progress and Applications
+ Traditional Network Representation
+ Concepts
	+ Representation learning
	+ Distributed representation
	+ Embedding
+ Network Embedding
	+ Map the nodes in a network **into** a low dimensional space
		+ Distributed representation for nodes
		+ Similarity between nodes indicate the link strength(节点间的相似性表示链路强度)
		+ Encode network information and generate node representation
+ Problems with previous Models
	+ Classical graph embedding algorithms
		+ MDS,IsoMap,LLE,Laplacian Eigenmap(???)
		+ Most of them follow a matrix factorization(矩阵分解) or computation approach(???)
		+ Hard to scale up(难以扩展)
		+ Diffcult to extend to new settings
+ Outline
	+ Preliminaries(初步措施)
		+ word2vec
	+ Basic Network Embedding Models
		+ DeepWalk,Node2Vec,LINE,GrapRep,SDNE
	+ Advanced Network Embedding Models
		+ Beyond embedding,vertex information,edge information(超越嵌入, 顶点信息, 边缘信息)
	+ Applications of Network Embedding
		+ Basic applications
		+ visualization
		+ text classification
		+ recommendation
+ Preliminaries
	+ Softmax functions
		+ sigmoid function
			$$ \phi(x) = \frac{1}{1 + e^{-x}}$$
		+ 
	+ Distributional semantics
	+ Word2Vec
		+ CROW
		+ Skip-gram