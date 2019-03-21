# Activation Function
## Reference
+ https://zhuanlan.zhihu.com/p/22142013
+ https://blog.csdn.net/qq_20909377/article/details/79133981

## Define
+ 在人工神经网络中，神经元节点的激活函数定义了对神经元输出的映射，简单来说，神经元的输出（例如，全连接网络中就是输入向量与权重向量的内积再加上偏置项）经过激活函数处理后再作为输出。加拿大蒙特利尔大学的Bengio教授在 ICML 2016 的文章[1]中给出了激活函数的定义：激活函数是映射 h:R→R，且几乎处处可导。
## 主要作用
+ 神经网络中激活函数的**主要作用**是提供网络的**非线性建模能力**，如不特别说明，激活函数一般而言是非线性函数。假设一个示例神经网络中仅包含线性卷积和全连接运算，那么该网络仅能够表达线性映射，即便增加网络的深度也依旧还是线性映射，难以有效建模实际环境中非线性分布的数据。
+ 加入（非线性）激活函数之后，深度神经网络才具备了分层的非线性映射学习能力。因此，激活函数是深度神经网络中不可或缺的部分。
+ 一个只有线性关系隐含层的多层网络不会比只有输入输出两层的神经网络更强大,因为线性函数的函数依然是一个线性函数

## Demo:线性变非线性

## 分类
+ sigmoid : logistic,  tanh,  高斯, softmax(梯度消失)
	+ 优点:
		+ 连续可微,参数一点的变化会带来输出的变化, 有利于判断参数的变化是否有利于目标函数的优化
	+ 缺点:
		+ 梯度消失(梯度随着隐藏层的增加而成指数减少)
			+ 产生原因:在反向传播算法中, 对梯度的计算使用链式法则,因此第n层的梯度都需要前面各层梯度的相乘,但是由于sigmoid函数的值域在(-1,1)或者(0,1)之间, 因此多个很小的数相乘后得到的第n层梯度就会接近于0, 造成训练的困难
+ threshold : relu, Hard Max
	+ 优点:
		+ 不会梯度消失
			+ threshold 的值域不在(-1,1)之间, 比如ReLU的值域是[0, +inf], 因此没有梯度消失的问题
			+ 另外一些激活函数, 比如 Hard Max : max(0,x), 在隐藏层中引入了稀疏性, 有助于模型的训练
+ 选择技巧 : 根据因变量选择激活函数
	+ 只有0,1取值的双值因变量  ===>>> logistic sigmoid (因为 logistic sigmoid 的输出就是 0, 1)
	+ 对于具有多个取值的离散因变量, 比如0到9的数字识别 ===>>> softmax
	+ 对于有限值域的连续型因变量, sigmoid 或者 tanh 激活函数都可以用, 但是需要将因变量的值域伸缩到sigmoid 或者tanh 的值域中
	+ 如果因变量为正, 且没有上限  ===>>> 指数函数
	+ 如果因变量没有值域, 或者虽然有值域但是边界未知 ===>>> 线性函数作为激活函数

## logistic(sigmoid)
+ https://www.jianshu.com/p/fcbc6983cc9a
+ 由于函数图像很像一个“S”型，所以该函数又叫 sigmoid 函数

![](https://pic3.zhimg.com/80/d2d1335c99df1923da4f228da3da9bdd_hd.jpg)

## fast sigmoid
![](https://img-blog.csdn.net/20170628153035514)

## tanh
![](https://pic3.zhimg.com/80/adbf164a53f8341cbf3a08c301b99b59_hd.jpg)

可见，tanh(x)=2sigmoid(2x)-1，也具有软饱和性。Xavier在文献[2]中分析了sigmoid与tanh的饱和现象及特点，具体见原论文。此外，文献 [3] 中提到tanh 网络的收敛速度要比sigmoid快。因为 tanh 的输出均值比 sigmoid 更接近 0，SGD会更接近 natural gradient[4]（一种二次优化技术），从而降低所需的迭代次数。

## ReLU
虽然2006年Hinton教授提出通过分层无监督预训练解决深层网络训练困难的问题，但是深度网络的直接监督式训练的最终突破，最主要的原因是采用了新型激活函数ReLU[5, 6]。与传统的sigmoid激活函数相比，ReLU能够有效缓解梯度消失问题，从而直接以监督的方式训练深度神经网络，无需依赖无监督的逐层预训练，这也是2012年深度卷积神经网络在ILSVRC竞赛中取得里程碑式突破的重要原因之一。

![](https://pic4.zhimg.com/80/0effc747d9b2fee78c14e390743fab69_hd.jpg)

可见，ReLU 在x<0 时硬饱和。由于 x>0时导数为 1，所以，ReLU 能够在x>0时保持梯度不衰减，从而缓解梯度消失问题。但随着
训练的推进，部分输入会落入硬饱和区，导致对应权重无法更新。这种现象被称为“神经元死亡”。
ReLU还经常被“诟病”的一个问题是输出具有偏移现象[7]，即输出均值恒大于零。偏移现象和 神经元死亡会共同影响网络的收敛性。本文作者公开在arxiv的文章[8]中的实验表明，如果不采用Batch Normalization，即使用 MSRA 初始化30层以上的ReLU网络，最终也难以收敛。相对的，PReLU和ELU网络都能顺利收敛，这两种改进的激活函数将在后面介绍。实验所用代码见GitHub - Coldmooon/Code-for-MPELU: Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units 。

ReLU另外一个性质是提供神经网络的稀疏表达能力，在Bengio教授的Deep Sparse Rectifier Neural Network[6]一文中被认为是ReLU带来网络性能提升的原因之一。但后来的研究发现稀疏性并非性能提升的必要条件，文献 RReLU [9]也指明了这一点。

PReLU[10]、ELU[7]等激活函数不具备这种稀疏性，但都能够提升网络性能。本文作者在文章[8]中给出了一些实验比较结果。首先，在cifar10上采用NIN网络，实验结果为 PReLU > ELU > ReLU，稀疏性并没有带来性能提升。其次，在 ImageNet上采用类似于[11] 中model E的15 层网络，实验结果则是ReLU最好。为了验证是否是稀疏性的影响，以 LReLU [12]为例进一步做了四次实验，负半轴的斜率分别为1，0.5，0.25, 0.1，需要特别说明的是，当负半轴斜率为1时，LReLU退化为线性函数，因此性能损失最大。实验结果展现了斜率大小与网络性能的一致性。综合上述实验可知，ReLU的稀疏性与网络性能之间并不存在绝对正负比关系。

![](https://pic1.zhimg.com/80/1248c4bdbc1285fc1ab1586c25f65c51_hd.jpg)

## PReLU

## RReLU

## Maxout
Maxout[13]是ReLU的推广，其发生饱和是一个零测集事件（measure zero event）。正式定义为：

![](https://pic4.zhimg.com/80/8358833218327d45cdd3f19a33d5479b_hd.jpg)

Maxout网络能够近似任意连续函数，且当w2,b2,…,wn,bn为0时，退化为ReLU。 其实，Maxout的思想在视觉领域存在已久。例如，在HOG特征里有这么一个过程：计算三个通道的梯度强度，然后在每一个像素位置上，仅取三个通道中梯度强度最大的数值，最终形成一个通道。这其实就是Maxout的一种特例。

Maxout能够缓解梯度消失，同时又规避了ReLU神经元死亡的缺点，但增加了参数和计算量。

## ELU
## Noisy Activation Functions
## CReLU
## MPELU

## Summary
+ 深度学习的快速发展，催生了形式各异的激活函数。面对琳琅满目的成果，如何做出选择目前尚未有统一定论，仍需依靠实验指导。
+ 一般来说，在分类问题上建议首先尝试 ReLU，其次ELU，这是两类不引入额外参数的激活函数。然后可考虑使用具备学习能力的PReLU和本文作者提出的MPELU，并使用正则化技术，例如应该考虑在网络中增加Batch Normalization层。
+ 围绕深度卷积神经网络结构，对十余种激活函数进行了总结，相关代码可在作者的github主页上下载：GitHub - Coldmooon/Code-for-MPELU: Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units。