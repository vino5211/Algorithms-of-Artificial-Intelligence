# Outline of Ensemble Learning

## Reference
+ [1] http://www.daicoolb.ml/xgboost-vs-lightbgm/
+ [2] GBDT
  + https://blog.csdn.net/w28971023/article/details/8240756
+ GBDT 
  - http://blog.csdn.net/a819825294/article/details/51188740
+ XGB
  - https://www.cnblogs.com/harvey888/p/7203256.html
+  [5] Ensemble learning 概述
  + http://www.datakit.cn/blog/2014/11/02/Ensemble_learning.html#stacking

## Concept
+ Bootstraping: 名字来自成语“pull up by your own bootstraps”，意思是依靠你自己的资源，称为自助法，它是一种有放回的抽样方法，它是非参数统计中一种重要的估计统计量方差进而进行区间估计的统计方法。其核心思想和基本步骤如下：
  + 采用重抽样技术从原始样本中抽取一定数量（自己给定）的样本，此过程允许重复抽样。 
  + 根据抽出的样本计算给定的统计量T。 
  + 重复上述N次（一般大于1000），得到N个统计量T。 
  + 计算上述N个统计量T的样本方差，得到统计量的方差。
+ 应该说Bootstrap是现代统计学较为流行的一种统计方法，在小样本时效果很好。通过方差的估计可以构造置信区间等，其运用范围得到进一步延伸。
+ Jackknife： 和上面要介绍的Bootstrap功能类似，只是有一点细节不一样，即每次从样本中抽样时候只是去除几个样本（而不是抽样），就像小刀一样割去一部分
+ 多模型系统
+ Committee Learning
+ Modular systems
+ 多分类器系统
+ Statictical Ensemble [5]
  + **统计总体，通常是无限的**
  + 机器学习下的Ensemble通常是指有限的模型组合

## Overview[5]

+ Ensemble的方法就是组合许多弱模型(weak learners，预测效果一般的模型) 以得到一个强模型(strong learner，预测效果好的模型)
+ Ensemble中组合的模型可以是同一类的模型，也可以是不同类型的模型
+ Ensemble方法对于大量数据和不充分数据都有很好的效果。因为一些简单模型数据量太大而很难训练，或者只能学习到一部分，而Ensemble方法可以有策略的将数据集划分成一些小数据集，分别进行训练，之后根据一些策略进行组合。相反，如果数据量很少，可以使用bootstrap进行抽样，得到多个数据集，分别进行训练后再组合(Efron 1979)
+ 使用Ensemble的方法在评估测试的时候，相比于单一模型，需要更多的计算。因此，有时候也认为Ensemble是用更多的计算来弥补弱模型。同时，这也导致模型中的每个参数所包含的信息量比单一模型少很多，导致太多的冗余！
+ **理论上来说，Ensemble方法也比单一模型更容易过拟合。但是，实际中有一些方法(尤其是Bagging)也倾向于避免过拟合**
+ 经验上来说，如果待组合的各个模型之间差异性(diversity )比较显著，那么Ensemble之后通常会有一个较好的结果，因此也有很多Ensemble的方法致力于提高待组合模型间的差异性
+ **尽管不直观，但是越随机的算法(比如随机决策树)比有意设计的算法(比如熵减少决策树)更容易产生强分类器。然而，实际发现使用多个强学习算法比那些为了促进多样性而做的模型更加有效。**
+ 下图是使用训练集合中不同的子集进行训练（以获得适当的差异性，类似于合理抽样），得到不同的误差，之后适当的组合在一起来减少误差

![](http://www.datakit.cn/images/machinelearning/EnsembleLearning_Combining_classifiers.jpg)



# Common types of ensembles

## Bayes optimal classifier[5]

+ 贝叶斯最优分类器

+ 假设空间中所有假设的一个Ensemble，通常来说BOC是最优的Ensemble

  + 见Tom M. Mitchell, Machine Learning, 1997, pp. 175 
  + 这里y是预测的类，C是所有可能的类别，H是”假设”空间( 模型 ），P是概率分布, T是训练数据

+ $$
  y = argmax_{\ c_j \in C} \sum_{h_i \in H} P（c_j|h_i)P(T|h_i)P(h_i)
  $$

  

+ 实际中无法使用BOC的原因
  + **绝大多数假设空间都非常大而无法遍历（无法 argmax）**
  + 很多假设给出的是一个类别而不是概率
  + **计算一个概率的无偏估计是非常难的 $$P(T|h_i)$$**
  + 估计各个假设的先验分布 $$P(h_i)$$ 基本不可行

## Bagging（Bootstrap aggregating)

+ 解释
  + 简单model 有较大bias 较小得variance
  + 复杂model 较小bias 较大variance
  + 将多个复杂的model 求平均值，bias 得平均值基本不变，variance 可能接近正确值
  + **为了提高模型的方差(variance, 差异性)，bagging在训练待组合的各个模型的时候是从训练集合中随机的抽取数据**
  + 其主要思想是将弱分类器组装成一个强分类器。在PAC（概率近似正确）学习框架下，则一定可以将弱分类器组装成一个强分类器
+ 步骤：
  + 使用前提：一个 model 很复杂，可能存在overfitting, 例如 Decision Tree(NN 相对不容易Overfitting)
  + A）从原始样本集中抽取训练集。每轮从原始样本集中使用Bootstraping的方法抽取n个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行k轮抽取，得到k个训练集。（k个训练集之间是相互独立的）
  + B）每次使用一个训练集得到一个模型，k个训练集共得到k个模型。（注：这里并没有具体的分类算法或回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树. 感知器等）
  + C）对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题，计算上述模型的均值作为最后的结果。（所有模型的重要性相同）
+ 训练：有放回抽样N份数据，将数据经过一个复杂的分类器（一个分类器，多份数据），将N个结果投票或求平均
+ 测试：测试数据经过N个分类器，将N个结果投票(分类)或求平均（回归）
+ bagging的一个有趣应用是非监督式学习中，图像处理中使用不同的核函数进行bagging
  + Image denoising with a multi-phase kernel principal component approach and an ensemble version
  + Preimages for Variation Patterns from Kernel PCA and Bagging

![](http://www.datakit.cn/images/machinelearning/EnsembleLearning_Bagging.jpg)

## Boosting

+ 一系列模型，新模型会更强调上一轮中被错误分类的样本

+ Boosting(提升法)是通过**不断的建立新模型**,  而新模型更强调上一个模型中被错误分类的样本(新样本），再将这些模型组合起来的方法。在一些例子中，boosting要比bagging有更好的准确率，但是也更容易过拟合

+ 核心问题：
  + 1）在每一轮如何改变训练数据的权值或概率分布？
    通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。

  + 2）通过什么方式来组合弱分类器？

    + 通过加法模型将弱分类器进行线性组合， 比如AdaBoost通过加权多数表决的方式，即增大错误率小的分类器的权值，同时减小错误率较大的分类器的权值

    + 而提升树通过拟合残差的方式逐步减小残差，将每一步生成的模型叠加得到最终模型

## Bagging 和 Boosting 得区别
+ bagging 是一个较复杂模型在一系列抽样数据上获得结果，boosting 是一个一些列教简单得模型在一份会变化得样本上获得结果
+ 1）样本选择上：
	+ Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
	+ Boosting：使用全部数据，不抽样，每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
+ 2）样例权重：
	+ Bagging：使用均匀取样，每个样例的权重相等
	+ Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。
+ 3）预测函数：
	+ Bagging：所有预测函数的权重相等。
	+ Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
+ 4）并行计算：
	+ Bagging：各个预测函数可以并行生成
	+ Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。
+ 不论是Boosting还是Bagging，所使用的多个分类器类型都是一致的。但是在前者当中，不同的分类器是通过串行训练而获得的，每个新分类器都根据已训练的分类器的性能来进行训练。Boosting是通过关注被已有分类器错分的那些数据来获得新的分类器。
- 由于Boosting分类的结果是基于所有分类器的加权求和结果的，因此Boosting与Bagging不太一样，Bagging中的分类器权值是一样的，而Boosting中的分类器权重并不相等，每个权重代表对应的分类器在上一轮迭代中的成功度。

## Stacking[5]

+ 思路
  + 训练一个新模型用于组合其他各个模型（combine)
    + 首先训练多个模型
    + 使用多个模型的输出作为一个新模型的输入，使用新模型的输出作为最终的输出
  + 如果可以选用任意一个组合算法，那么stacking可以表示上面提到的各种emsemble方法
  + 实际中，通常选用单层的logistic回归当作新模型
  + 潜在的一个思想是希望训练数据都得被正确的学习到了，比如某个分类器错误的学习到了特征空间里某个特定区域，因此错误分类就会来自这个区域，但是Tier 2分类器可能根据其他的分类器学习到正确的分类
+ 操作过程
  + 通过bootstrap aggregating（Bagging）的方式  获得各个训练集
  + 每个训练集使用一种算法来训练对应的模型，得到一些列模型（下图中成为 TIER-1）
  + 然后将一系列模型的输出用于训练TIER-2
+ 交叉验证也通常用于训练Tier 1分类器
  + 把这个训练集合分成T个块，Tier 1中的每个分类器根据各自余下的T-1块进行训练，并在T块（该块数据并未用于训练）上测试
  + 之后将这些分类器的输出作为输入，在整个训练集合上训练Tier 2分类器。（这里未提及测试集，测试集是指不在任何训练过程中出现的数据）

![](http://www.datakit.cn/images/machinelearning/EnsembleLearning_Stacked_generalization.jpg)

## Bayesian model averaging

+ 贝叶斯模型平均

## Bayesian model combination

+ 是 BMA的一个校正算法

## Bucket of models

+ 针对具体问题进行最优模型选择的方法
+ 常用的方法是 交叉验证(cross-validation), 又称 bake-off contest

```
For each model m in the bucket:
  Do c times: (where 'c' is some constant)
    Randomly divide the training dataset into two datasets: A, and B.
    Train m with A
    Test m with B
Select the model that obtains the highest average score
```

# Ensemble combination rules

+ Ensemble内的各个模型不仅仅可以是同一个模型根据训练集合的随机子集进行训练（得到不同的参数），也可以不同的模型进行组合、甚至可以是针对不同的特征子集进行训练
+ 方式
  + Abstract-level ： 各个模型只输出一个模型类别
  + Rank-level ：各个模型输出的是目标类别的一个排序
  + Measurement-level：各个模型输出的是目标类别的概率估计或一些相关的信念值，如猫，狗，人的图像识别中，输出人0.7， 狗0.2，猫0.1
+ 组合器
  + Algebraic combiners
  + perceptron
  + logistic

# 演变
+ 1）Bagging + 决策树 = 随机森林  （很多 深度较大得决策树）
+ 2）AdaBoost + 决策树 = 提升树	(很多 深度较小的决策树)
+ 3）Gradient Boosting + 决策树 = GBDT(Gradient Boosting Decision Tree)

# 集成半监督学习的改进

