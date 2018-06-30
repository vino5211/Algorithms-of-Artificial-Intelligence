# Tensorflow 
## Reference
+ 常用函数
	+ https://blog.csdn.net/wuqingshan2010/article/details/71056292

## 常用函数
+ TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU。一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测。
+ 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.
+ 并行计算能让代价大的算法计算加速执行，TensorFlow也在实现上对复杂操作进行了有效的改进。大部分核相关的操作都是设备相关的实现，比如GPU
+ 下面是一些重要的操作/核：
	+ Maths
		+ Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
	+ Array	
		+ Concat, Slice, Split, Constant, Rank, Shape, Shuffle
	+ Matrix
		+ MatMul, MatrixInverse, MatrixDeterminant
	+ Neuronal Network	
		+ SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool
	+ Checkpointing	
		+ Save, Restore
	+ Queues and syncronizations
		+ Enqueue, Dequeue, MutexAcquire, MutexRelease
	+ Flow control	
		+ Merge, Switch, Enter, Leave, NextIteration

## 算术操作
+ tf.add(x, y, name=None)
	+ 求和
+ tf.sub(x, y, name=None)
	+ 减法
+ tf.mul(x, y, name=None)
	+ 乘法
+ tf.div(x, y, name=None)
	+ 除法
+ tf.mod(x, y, name=None)
	+ 取模
+ tf.abs(x, name=None)	
	+ 求绝对值
+ tf.neg(x, name=None)	
	+ 取负 (y = -x).
+ tf.sign(x, name=None)	
	+ 返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
+ tf.inv(x, name=None)	
	+ 取反
+ tf.square(x, name=None)	
	+ 计算平方 (y = x * x = x^2).
+ tf.round(x, name=None)
	+ 舍入最接近的整数
	```
	# ‘a’ is [0.9, 2.5, 2.3, -4.4]
	tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
	```
+ tf.sqrt(x, name=None)	
	+ 开根号 (y = \sqrt{x} = x^{1/2}).
+ tf.pow(x, y, name=None)
	+ 幂次方 
	```
	# tensor ‘x’ is [[2, 2], [3, 3]]
	# tensor ‘y’ is [[8, 16], [2, 3]]
	tf.pow(x, y) ==> [[256, 65536], [9, 27]]
	```
+ tf.exp(x, name=None)	
	+ 计算e的次方
+ tf.log(x, name=None)	
	+ 计算log，一个输入计算e的ln，两输入以第二输入为底
+ tf.maximum(x, y, name=None)	
	+ 返回最大值 (x > y ? x : y)
+ tf.minimum(x, y, name=None)	
	+ 返回最小值 (x < y ? x : y)
+ tf.cos(x, name=None)	
	+ 三角函数cosine
+ tf.sin(x, name=None)	
	+ 三角函数sine
+ tf.tan(x, name=None)	
	+ 三角函数tan
+ tf.atan(x, name=None)	
	+ 三角函数ctan

## 张量操作
## 矩阵操作
## 复数操作
## 规约计算
## 分割
## 序列比较与索引提取
## 神经网络
## 保存与恢复变量
