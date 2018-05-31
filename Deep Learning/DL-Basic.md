[TOC]

# Deep Learning Basic

## Reference

## Some projects
+ https://github.com/fchollet/deep-learning-models

## Some blogs
+ https://blog.csdn.net/u012526120/article/list/2?


## Basic Problems
- Neural Network
	- Fully Connected Layer
	- Recurrent Structure
		- RNN
		- LSTM
		- GRU
		- stack RNN
	- Convolutional/Pooling Layer
- cost function
- Optimization




## Train Setting
- Batch-size
	+ batch size 32 相当于 前 32 次反向传播求得参数（w，b）的平均值 做 第33 次输入的初始化参数，即 item1，item2, ..., item 32 的反向传播的得到的参数做 batch 1 的结果，batch 1 的结果做batch 2 的初始化参数
	+ batch_size设的大一些，收敛得快，也就是需要训练的次数少，准确率上升得也很稳定，但是实际使用起来精度不高。batch_size设的小一些，收敛得慢，而且可能准确率来回震荡，所以还要把基础学习速率降低一些；但是实际使用起来精度较高。一般我只尝试batch_size=64或者batch_size=1两种情况。

## loss function
- Biased Importance Sampling for Deep Neural Network Training
	- Importance Sampling 在凸问题的随机优化上已经得到了成功的应用。但是在 DNN 上的优化方面结合 Importance Sampling 存在困难，主要是缺乏有效的度量importance 的指标。
	- 本文提出了一个基于 loss 的 importance 度量指标，并且提出了一种利用小型模型的 loss 近似方法，避免了深度模型的大规模计算。经实验表明，结合了 Importance Sampling 的训练在速度上有很大的提高。
	- 论文链接：https://www.paperweekly.site/papers/1758
	- 代码链接：https://github.com/idiap/importance-sampling

## 公式推导
+ RNN前向传播与后向传播公式推导
	+ https://zhuanlan.zhihu.com/p/28806793
+ 传统全连接神经网络和RNN的反向传播求导
	+ https://zhuanlan.zhihu.com/p/29731200

## Optimizer
- Adam
- SGD

+ 深度学习-学习优化(Andrew Ng. DL 笔记)
	+ http://www.shuang0420.com/2017/08/15/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-%E5%8A%A0%E5%BF%AB%E5%AD%A6%E4%B9%A0%E9%80%9F%E5%BA%A6/
	+ mini-batch
	+ Momentum
	+ RMSprop
	+ Adam
	+ Learning rate decay

## 正则化
+ http://blog.sina.com.cn/s/blog_8267db980102wryn.html
+ 如何判断LSTM模型中的过拟合和欠拟合
	+ https://www.tuicool.com/articles/VNzqmu6

+ 深度学习-过拟合(Andrew Ng. DL 笔记)
	+ http://www.shuang0420.com/2017/08/29/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-%E8%BF%87%E6%8B%9F%E5%90%88(Andrew%20Ng.%20DL%20%E7%AC%94%E8%AE%B0)/
	+ 深度学习中 --- 解决过拟合问题（dropout, batchnormalization）
		+ 过拟合，在Tom M.Mitchell的《Machine Learning》中是如何定义的：给定一个假设空间H，一个假设h属于H，如果存在其他的假设h’属于H,使得在训练样例上h的错误率比h’小，但在整个实例分布上h’比h的错误率小，那么就说假设h过度拟合训练数据。
		+ 也就是说，某一假设过度的拟合了训练数据，对于和训练数据的分布稍有不同的数据，错误率就会加大。这一般会出现在训练数据集比较小的情况。
		+ 深度学习中避免过拟合的方法：
			+ Dropout
				+ 2012年ImageNet比赛的获胜模型AlexNet论文中提出的避免过拟合的方法。其操作方法如下图所示。
				+ 在训练中以概率P(一般为50%)关掉一部分神经元，如图中的虚线的箭头。那么对于某些输出，并不是所有神经元会参与到前向和反向传播中。
				+ 在预测的时候，将使用所有的神经元，但是会将其输出乘以0.5
     			+ Dropout的意义在于，减小了不同神经元的依赖度。有些中间输出，在给定的训练集上，可能发生只依赖某些神经元的情况，这就会造成对训练集的过拟合。而随机关掉一些神经元，可以让更多神经元参与到最终的输出当中。我觉得dropout方法也可以看成，联合很多规模比较小的网络的预测结果，去获取最终的预测。
			+ Batch Normalization
				+ Batch Normalization: Accelerating Deep Network Training by  Reducing Internal Covariate Shift
				+ Batch Normalization（批量标准化）
					+ CNN 往往有很深的层数，而层数越深，越容易出现梯度消失。这是因为随着网络的传播，数据的分布产生了变化，所以利用 Batch Normalization 让数据的分布保持均值为 0，方差为 1 的正太分布。
				+ 一般放在 relu 之前，可以提高准确性，加快训练速度。
				+ CNN 有应用
					+ https://blog.csdn.net/aichipmunk/article/details/54234646
				+ RNN ?
				+ https://morvanzhou.github.io/tutorials/machine-learning/torch/5-04-A-batch-normalization/

			+ https://zhuanlan.zhihu.com/p/30951658
				1.Early stop
				在模型训练过程中，提前终止
				2.Data expending
   				用更多的数据集
				3.正则
   				众所周知的正则化，有L1 L2 两者，顺便问下两者区别是什么？
				4.Droup Out
   				以一定的概率使某些神经元停止工作，可以从ensemble的角度来看，顺便问下为啥Droup Out效果就是好呢？
				5.BatchNorm
   				对神经元作归一化
				6.REF
				神经网络简介-防止过拟合 - CSDN博客
				深度学习（二十九）Batch Normalization 学习笔记
                理解droupout - CSDN博客
                神经网络中的Early Stop
                L2正则项与早停止(Early Stop)之间的数学联系
                【CV知识学习】early stop、regularation、fine-tuning and some other trick to be known
