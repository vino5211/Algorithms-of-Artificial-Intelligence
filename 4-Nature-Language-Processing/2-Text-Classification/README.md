# Current Solution for Text Classification

## 分类性能评测
+ 正确率，召回率，F-score
+ 微平均
	+ 根据总数据计算 P, R，F
+ 宏平均
	+ 计算出每个类得，再求平均值
+ 平衡点
+ 11点平均正确率
	+ https://blog.csdn.net/u010367506/article/details/38777909

## 相关数据集及SOTA方法
- pass


## Traditional Classifications Algorithms

### 文本关键词
+ TFIDF
+ TextRank
	+ https://my.oschina.net/letiantian/blog/351154
	+ TextRank算法基于PageRank，用于为文本生成关键字和摘要

### 文本表示（Text Represent）
+ VSM
    + n-gram 获得特征
        + https://zhuanlan.zhihu.com/p/29555001
    + TFIDF 获得特征权重

+ 词组表示法
	+ 统计自然语言处理 P419

+ 概念表示法
	+ 统计自然语言处理 P419

### 文本特征选择
+ http://sklearn.apachecn.org/cn/0.19.0/modules/feature_extraction.html
+ 文档频率
	+ 从训练预料中统计包含某个特征的频率（个数），然后设定阈值（两个）
	+ 当特征项的DF小与某个阈值（小）时，去掉该特征：因为该特征项使文档出现的频率太低，没有代表性
	+ 当特征项的DF大于某个阈值（大）时，去掉改特征：因为该特征项使文档出现的频率太高，没有区分度
	+ 优点：简单易行，去掉一部分噪声
	+ 缺点: 借用方法，理论依据不足;根据信息论可知某些特征虽然出现频率低，但包含较多得信息
+ 信息增益
	+ 根据某个特征项能为整个分类所能提供得信息量得多少来衡量该特征项得重要程度，从而决定该特征项得取舍
	+ 某个特征项的信息增益是指有该特征和没有该特征时，为整个分类能提供信息量得差别
+ 卡方统计量
	+ 特征量$t_i$ 和 类别$C_j$ 之间得相关联程度, 且假设 $t_i$ 和 类别$C_j$ 满足具有一阶自由度得卡方分布
+ 互信息
	+ 互信息越大，特征量$t_i$ 和 类别$C_j$ 共现得程度就越大

+ 其他方法


### 特征权重计算
+ 特征权重是衡量某个特征项在文档表示中得重要程度
+ 方法
	+ 布尔权重
	+ 绝对词频
	+ 逆文档频率
	+ TF-IDF
	+ TFC
	+ ITC
	+ 熵权重
	+ TF-IWF

### 文本数据分布

### 传统分类器

## DL Classification Algorithms
### Fasttext
### TextCNN
### HAN
### DPCNN
### 

