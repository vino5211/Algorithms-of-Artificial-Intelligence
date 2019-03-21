
# Outline of Regularization

### 过拟合产生
+ ZHH ML P252 P129
	+ 当样本特征较多，而样本数较少时，很容易陷入过拟合


+ L2 岭回归(ridge regression)

+ L1 缓解过拟合的同时，可以进行特征选择(LASSO)
	+ L1 正则化的求解方法
		+ 近端梯度下降

### 权重递减　Weight Decay
+ 正则函数为权重的平方和
+ 与岭回归(L2)使用的技巧一样

### 贝叶斯的思路
+ 将权重的先验分布的对数作为正则项
