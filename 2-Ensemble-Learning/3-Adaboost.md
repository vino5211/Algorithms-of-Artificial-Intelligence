# Adaboost(Adaptive Boosting)



## Reference

+ https://www.cnblogs.com/pinard/p/6133937.html

+ https://blog.csdn.net/haidao2009/article/details/7514787
+ [AdaBoost及其他变种](https://zhuanlan.zhihu.com/p/25096501)
+ [Wiki AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)

### Overview

+ 针对的是两类分类问题
+ PAC 定义了学习算法的强弱
  - 弱学习算法---识别错误率小于1/2(即准确率仅比随机猜测略高的学习算法)
  - 强学习算法---识别准确率很高并能在多项式时间内完成的学习算法

### Abstract

+ AdaBoost 本质就是，每次迭代更新样本分布，然后对新的分布下的样本学习一个弱分类器，和它对应的权重。更新样本分布的规则是：减小之前弱分类器分类效果较好的数据的概率，增大之前弱分类器分类效果较差的数据的概率。最终的分类器是弱分类器线性组合

### Framework

![](../../../../Downloads/1042406-20161204194331365-2142863547.png)



![](../../../../Downloads/v2-b1db524dfb651d258928c0e175571555_hd.png)



### 损失函数的优化

### 二分类的算法流程

### 回归的算法流程

### Adaboost 算法的正则化