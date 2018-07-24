## Reference
+ https://blog.csdn.net/scutjy2015/article/details/72810730
## underfitting
+ 增加网络的复杂度（深度）
+ 降低learning rate
+ 优化数据集
+ 增加网络的非线性度（ReLu）
+ 采用batch normalization
## overfitting
+ 丰富数据
	+ 更多的数据 好于 更好的网络结构
+ 增加网络的稀疏度
	+ ?
+ 降低网络的复杂度（深度）
	+ 添加Dropout
		+ 随机使部分神经元不工作
		+ 测试的时候使用全部神经元, 并对权重进行处理(例如dropout 是 0.7 即 70% 的神经元不工作, 测试的时候要将该层的权重变为 3/10), 因为测试的时候整体神经元数量较多, 权重不变的情况下, loss会变大, 为保证梯度不变, 需要将权重减小
+ 正则项
	+ L1 regularization
	+ L2 regulariztion
		+ $ C = C_0 + \frac{\lambda}{2n} \sum_{w} w^2$

+ Early stopping
	+ https://www.jianshu.com/p/9ab695d91459
+ 适当降低Learning rate
+ 适当减少epoch的次数