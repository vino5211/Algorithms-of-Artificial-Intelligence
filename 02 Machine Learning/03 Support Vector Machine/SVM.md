# Machine-Learing SVM

### key idea
+ Hinge loss
+ Kernel

## Reference
+ LHY ML 2017 Support Vector Macnhine
+ (U)用一张图理解SVM 的脉

## Hinge Loss

+ A. square loss
	+ $l(f(x^n),\ {\hat y}^n) = (1 - f(x^n) \times {\hat y}^n)^2$
+ B. square loss + sigmoid
	+ $l(f(x^n),\ {\hat y}^n) = (1 - \sigma(f(x^n) \times {\hat y}^n))^2$
+ C. square loss + cross entropy
	+ $l(f(x^n),\ {\hat y}^n) = (1 + exp(- f(x^n) \times {\hat y}^n))$
+ D. hinge loss
	+ $l(f(x^n),\ {\hat y}^n) = max(0, 1 - f(x^n) \times {\hat y}^n)$

<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM1.png" width="500px" height="500px" />

+ 对比
	+ A 不适合做loss 函数， 如图红线当横坐标取极大时，loss也趋向极大
	+ C 比 B 数值变化明显，效果较好
	+ D 和 C 各有千秋, 主要区别体现在 当横坐标大于 1 时 hinge loss 认为结果已经够好了，所以loss的值全是0

## Linear SVM
+ step 1
	+ 公式有待添加
+ step 2
+ step 3
	+ loss function 和 正则项都是convex的，所以整体也是convex的 
	+ 尽管不是全局可微分的，但可以分段计算导数，之后进行gradient descent
	+ 类似 relu, maxout
	
        <img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM2.png" width="300px" height="300px" />
	
	+ 注意公式中最后一个i
	+ loss 对 $w_i$ 的偏导数 可以看成所有输入样本第i维的线性组合

## 推到至常见SVM的解释

<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM3.png" width="300px" height="300px" />

+ (???) QP 问题，二次规划

## Dual Representation
+ (???) KKT
+ (周志华 P123) 由 拉格朗日乘数法推导
+ (LHY ML 2017)
	+ 如下图 $x^n$ 为训练数据中的某一样本，x 为输入样本

	<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM4.png" width="300px" height="300px" />

	+ K 表示 核函数, 当前图中仅仅表示内积
        
        <img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM5.png" width="300px" height="300px" />

## Kernel
+ 在一些情况下，在低维空间中无法完成分类操作（不可分），需要将低维空间映射到高维空间，即在内积之前可以对 z 和 x 进行变化, 即$[\phi(z),\phi(x)]$。(中括号表示内积)
+ 但是经过变化后一般都是将低维空间映射到高维空间(甚至是无穷维)，在高维空间中进行内积计算较为困难（或者无法计算），故希望找到一些特殊映射（核函数）能满足$[\phi(z),\phi(x)] = \phi([x,z])$。(中括号表示内积), 这些特殊得映射称为核函数

+ Demo
	+ Radial Basis Function Kernel (RBF)
	
        <img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM6.png" width="300px" height="300px" />
	
	+ Sigmoid Kernel
	
        <img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/SVM7.png" width="300px" height="300px" />

+ 核函数的判定

## 模型参数
+ https://blog.csdn.net/lujiandong1/article/details/46386201
+ SVM模型有两个非常重要的参数C与gamma。其中 C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
+ gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。

## 软间隔和正则化
+ L1 稀疏解，特征选择
+ L2 就只是一种规则化而已

### LR 与 SVM 的区别与联系
+ 联系:
	+ 一般都是处理二分类问题
	+ 都可以添加不同的正则项
+ 区别:
	+ loss
		+ LR : logistical loss 
		+ SVM : hinge loss, 只考虑support vectors
	+ LR 
		+ 简单,好理解,大规模线性分类时比较方便
		+ LR 能做的 SVM 也可以做, 有点准确率可能不然SVM
	+ SVM 
		+ 使用复杂核函数计算支持向量时, 简化模型和计算
		+ SVM能做的, 有的LR做不了