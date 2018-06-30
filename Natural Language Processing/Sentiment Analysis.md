# Sentiment Analysis

## Reference
+ 统计自然语言处理 P431
+ https://zhuanlan.zhihu.com/p/23615176
+ Attention-based LSTM for Aspect-level Sentiment Classification

## Diff with Text classification
+ 文本分类更侧重与文本得客观性，情感分类更侧重主观性

## 分类[1]
+ 学习方式
	+ 有监督
	+ 无监督
	+ 半监督
+ 侧重方向
	+ 领域相关性研究
		+ 跨领域时保持一定的分类能力
	+ 数据不平衡研究
		+ 有监督
			+ 将多的类进行内部聚类
			+ 在聚类后进行类内部层次采样，获得同少的类相同数据规模得样本
			+ 使用采样样本，并结合类的中心向量构建新的向量，并进行学习
		+ 不平衡数据的半监督问题
		+ 不平衡数据的主动学习问题