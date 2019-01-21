# Semi supervised learning

## 原理
+ 无标签的样本虽然未直接包含标记信息，但如果他们与有标记得样本是从同样得数据源独立同分布采样而来，则他们所包含的关于数据分布得信息对建立模型将有很大益处
+ 类似聚类与流形学习，都是相似得样本有相似得输出

+ 分类
	+ 纯半监督 pure semi supervised learning
		+ 开发世界假设
		+ train(有标记数据 + 未标记数据A)， predict(预测数据B)
		+ A is not B
	+ 直推学习 transductive learning
		+ 封闭世界假设
		+ train(有标记数据 + 未标记数据C)， predict(未标记数据D)
		+ C is D
## 半监督SVM

## 图半监督

## 半监督聚类

## 集成半监督

---

## Semi-supervised Learning for Generative Model

## Low-density Separation Assumption

## Smoothness Assumption

## Better Representation