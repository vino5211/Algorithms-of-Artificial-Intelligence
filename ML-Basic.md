# ML- Basic
- 【机器学习基本理论】详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解
- 似然与极大似然估计
	- https://zhuanlan.zhihu.com/p/22092462
	- 概率是在特定环境下某件事情发生的可能性
	- 而似然刚好相反，是在确定的结果下去推测产生这个结果的可能环境（参数）
	- ！[](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D%28%5Ctheta%7Cx%29+%3DP%28x%7C%5Ctheta%29)
	- 解释了硬币问题的似然估计方法
- 入门 | 什么是最大似然估计、最大后验估计以及贝叶斯参数估计
	- 机器之心

+ Feature extraction and Feature selection
	+ extraction : 用映射或者变换的方法把原始特征变换为较少的新特征
	+ selection : 从原始特征中挑选出一些最有代表性，分类性能最好的额特征

## Evaluate
+ https://www.zhihu.com/question/30643044
+ accuracy
+ P R F1
+ ROC/AUC
	+ https://www.douban.com/note/284051363/

## 小样本不平衡样本
- 训练一个有效的神经网络，通常需要大量的样本以及均衡的样本比例，但实际情况中，我们容易的获得的数据往往是小样本以及类别不平衡的，比如银行交易中的fraud detection和医学图像中的数据。前者绝大部分是正常的交易，只有少量的是fraudulent transactions；后者大部分人群中都是健康的，仅有少数是患病的。因此，如何在这种情况下训练出一个好的神经网络，是一个重要的问题。
本文主要汇总训练神经网络中解决这两个问题的方法。

Training Neural Networks with Very Little Data - A Draft -arxiv,2017.08
- “Training Neural Networks with Very Little Data”学习笔记



## 判别模型和生成模型
+ Reference 
	+ http://www.cnblogs.com/zhangchaoyang/articles/7100083.html
+ 生成模型：
	+ 直接求联合概率p(x,y)，得到p(x,y)后就可以去生成样本。
	+ HMM、高斯混合模型GMM、LDA、PLSA、Naive Bayes都属于生成模型。
	+ 当我们得到HMM模型后，就可以根据初始状态分布π、状态转移矩阵A和发射矩阵B去生成一个状态序列及相应的观察序列，即拿着生成模型可以去生成样本。
	+ LDA（或PLSA）模型也一样，得到文档下的主题分布p(zk|di)及主题下的词分布p(wj|zk)后，上帝就可以去创作文章了。

+ 判别模型：
	+ 直接求判别（或者是预测）函数y=(f(x)，或者另一种表达：p(y|x)。
	+ 最大熵MaxEnt、人工神经网络ANN、逻辑回归LR、线性判别分析LDA、K-Means、KNN、SVM、决策树都属于判别模型。
	+ 最大熵直接去求p(y|x)，它不会浪费功夫去求p(x,y)
	+ 同样KNN也不关心样本是如何生成的，它只会对样本进行分类。

+ 由生成模型可以得到判别模型，因为p(y|x)=p(x,y)p(x)，但它有2个缺点：
	+ 需要额外地去求p(x)。
	+ 样本量不足的情况下p(x)可能求不准。此时预测p(y|x)没有判别模型准。

+ 生成模型也有它的优点：
	+ 可以生成样本，反应样本本身的相似度
	+ 比如LDA中得到了p(zk|di)就相当于得到了文档向量，可以去计算向量之间的相似度。
+ 当含有隐含变量时，也可以用生成模型，但不能用判别模型。比如HMM、GMM、LDA、PLSA模型都可以用EM算法求解。

## loos function
+ 最小二乘法