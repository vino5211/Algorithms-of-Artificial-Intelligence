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