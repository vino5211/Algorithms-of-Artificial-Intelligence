# View of Sequence Labeling
### 人民日报语料下载

### NCRF++:An Open-source Neural Sequence Labeling Toolkit
+ NCRF++ 被设计用来快速实现带有CRF层的不同神经序列标注模型
+ 可编辑配置文件灵活建立模型
+ 论文笔记:COLING 2018 最佳论文解读:序列标注经典模型复现

### Neural CRF
- http://nlp.cs.berkeley.edu/pubs/Durrett-Klein_2015_NeuralCRF_paper.pd

### FoolNLTK
- 中文处理工具包
- 特点：
	- 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
	- 基于 BiLSTM 模型训练而成
	- 包含分词，词性标注，实体识别，都有比较高的准确率
	- 用户自定义词典
- 项目链接：https://github.com/rockyzhengwu/FoolNLTK 


### Adversarial training for multi-context joint entity and relation extraction
+ 根特大学
+ EMNLP2018
+ 同时执行实体识别和关系抽取的multi-head selection 联合模型
+ 实验证明该文提出的方法在大多数数据集上, 可以不依赖NLP工具,且不使用人工特征设置的情况下,同步解决多关系问题
