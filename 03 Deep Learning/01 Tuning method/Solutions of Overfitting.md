# Solution of Overfitting

### Reference
+ https://blog.csdn.net/scutjy2015/article/details/72810730

### overfitting
+ 数据集扩增
	+ 原有数据增加
	+ 原有数据加随机噪声
	+ 重新采样
+ 增加网络的稀疏度
	+ Drop out
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
+ 交叉验证