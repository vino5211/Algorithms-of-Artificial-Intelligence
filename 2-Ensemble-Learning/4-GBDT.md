# GBDT
### Reference
+ https://www.cnblogs.com/pinard/p/6140514.html

### GBDT主要由三个概念组成：
+ Regression Decistion Tree（即DT)
	+ CART回归树
+ Gradient Boosting（即GB)
+ Shrinkage (算法的一个重要演进分枝，目前大部分源码都按该版本实现）

### 原理
- GBDT与传统的Boosting区别较大，它的每一次计算都是为了减少上一次的残差，而为了消除残差，我们可以在残差减小的梯度方向上建立模型,所以说，在GradientBoost中，每个新的模型的建立是为了使得之前的模型的残差往梯度下降的方法，与传统的Boosting中关注正确错误的样本加权有着很大的区别
- 在GradientBoosting算法中，关键就是利用损失函数的负梯度方向在当前模型的值作为残差的近似值，进而拟合一棵CART回归树
- GBDT的会累加所有树的结果，而这种累加是无法通过分类完成的，因此GBDT的树都是CART回归树，而不是分类树（尽管GBDT调整后也可以用于分类但不代表GBDT的树为分类树）

### GBDT 回归

### GBDT 分类

### 优缺点
- GBDT的性能在RF的基础上又有一步提升，因此其优点也很明显， 1. 它能灵活的处理各种类型的数据； 2. 在相对较少的调参时间下，预测的准确度较高。
- **当然由于它是Boosting，因此基学习器之前存在串行关系，难以并行训练数据**。