## Training and Tuning

## 传播公式推导
+ RNN前向传播与后向传播公式推导
	+ https://zhuanlan.zhihu.com/p/28806793
+ 传统全连接神经网络和RNN的反向传播求导
	+ https://zhuanlan.zhihu.com/p/29731200

## Batch-size
+ batch size 32 相当于 前 32 次反向传播求得参数（w，b）的平均值 做 第33 次输入的初始化参数，即 item1，item2, ..., item 32 的反向传播的得到的参数做 batch 1 的结果，batch 1 的结果做batch 2 的初始化参数
+ batch_size设的大一些，收敛得快，也就是需要训练的次数少，准确率上升得也很稳定，但是实际使用起来精度不高。batch_size设的小一些，收敛得慢，而且可能准确率来回震荡，所以还要把基础学习速率降低一些；但是实际使用起来精度较高。一般我只尝试batch_size=64或者batch_size=1两种情况。

## Optimizer
- Biased Importance Sampling for Deep Neural Network Training
	- Importance Sampling 在凸问题的随机优化上已经得到了成功的应用。但是在 DNN 上的优化方面结合 Importance Sampling 存在困难，主要是缺乏有效的度量importance 的指标。
	- 本文提出了一个基于 loss 的 importance 度量指标，并且提出了一种利用小型模型的 loss 近似方法，避免了深度模型的大规模计算。经实验表明，结合了 Importance Sampling 的训练在速度上有很大的提高。
	- 论文链接：https://www.paperweekly.site/papers/1758
	- 代码链接：https://github.com/idiap/importance-sampling