# Simple RNN

### Reference
+ Deep　Learning Chapter 10 Recurrent and Recursive Network
+ https://www.jianshu.com/p/32d3048da5ba

### Outline of Simple RNN
+ 用于处理序列
+ 结构
	+ 从结构上与BP一脉相承, 都有前馈和反馈层, 但simple RNN 引入时间循环机制
	+ 神经网络为A，通过读取某个t时间(状态)的输入x_t，然后输出一个值h_t。循环可以使得从当前时间步传递到下一个时间步
	
        ![](https://upload-images.jianshu.io/upload_images/2666154-68f7ea029d4626fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/150)
	+ 这些循环使得RNN可以被看做同一个网络在不同时间步的多次循环，每个神经元会把更新的结果传递到下一个时间步，为了更清楚的说明，将这个循环展开，放大该神经网络A，看一下网络细节

	![](https://upload-images.jianshu.io/upload_images/2666154-fa24f52a330198f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/348)
+ 展开计算图
	+ 展开计算图将导致深度网络结构中的参数共享
	+ 经典动态系统
		+ $ｓ^{(t)} = f(s^{(t-1)};\theta)$
	+ 由外部信号驱动的动态系统
		+ $ｓ^{(t)} = f(s^{(t-1)}, x^{t};\theta)$
+ 参数
	+ 递归网络的输入是一整个序列，也就是x=[ x_0, ... , x_t-1, x_t, x_t+1, x_T ]，对于语言模型来说，每一个x_t将代表一个词向量，一整个序列就代表一句话。h_t代表时刻t的隐含状态，y_t代表时刻t的输出。
	+ 其中：
		+ U：输入层到隐藏层直接的权重
		+ W：隐藏层到隐藏层的权重
		+ V： 隐藏层到输出层的权重

+ 参数共享
	+ 使得模型能够拓展到不同形式的样本(这里指不同长度)进行泛化
	+ 假设要训练一个处理固定长度句子的前馈网络
		+ 全连接:会给每个特征分配一个单独的参数, 所以需要学习句子每个位置的语言规则
		+ Recurrent NN : 在几个时间步内共享权重, 不需要学习句子每个位置的语言规则
+ 正向传播(Forward Propagation)
	+ 依次按照时间的顺序计算一次即可
	+ 首先在t=0的时刻，U、V、W都被随机初始化好了，h_0通常初始化为0，然后进行如下计算：
	
        ![](https://upload-images.jianshu.io/upload_images/2666154-a460042bd502dfac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/155)

	+ 其中，f ，g 是激活函数，g通常是Softmax
	+ RNN有记忆能力，正是因为这个 W，记录了以往的输入状态，作为下次的输出。这个可以简单的理解为 
	$$h = f(现在的输入 +过去的记忆）$$
	+ 全局误差为
	$$ E = \sum{e} = \sum_{i=1}^{t} f_e (y_i - d_i)$$
	+ E是全局误差，$e_i$是第i个时间步的误差，y是输出层预测结果，d是实际结果。误差函数$f_e$可以为交叉熵( Cross Entropy ) ,也可以是平方误差项等


+ 反向传播(Back Propagation)
	+ 从最后一个时间将累积的残差传递回来即可
	+ 利用输出层的全局误差，求解各个权重$\bigtriangledown_V$、$\bigtriangledown_U$、$\bigtriangledown_W$，然后梯度下降更新各个权重
	+ 更新公式如下
	　
	$$ U(t+1) = U(t) + \alpha \cdot \bigtriangledown_U$$
	$$ V(t+1) = V(t) + \alpha \cdot \bigtriangledown_V$$
	$$ W(t+1) = W(t) + \alpha \cdot \bigtriangledown_W$$
	$$ \bigtriangledown_U = \frac{\delta E}{\delta U} = \sum_{i} \frac{\delta_{e_i}}{\delta_U}$$
	$$ \bigtriangledown_V = \frac{\delta E}{\delta V} = \sum_{i} \frac{\delta_{e_i}}{\delta_V}$$
	$$ \bigtriangledown_W = \frac{\delta E}{\delta W} = \sum_{i} \frac{\delta_{e_i}}{\delta_W}$$
	+ $\bigtriangledown_V$不依赖之前的状态
	+ $\bigtriangledown_U$、$\bigtriangledown_W$不能直接求导，需定义中间变量
		+　https://www.jianshu.com/p/32d3048da5ba 

### 与全连接的区别
+　在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，**每层之间的节点是无连接的**。但是这种普通的神经网络对于很多问题却无能无力
+　例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的

### 有效长度
+　RNNs之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。
+　理论上，RNNs能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关

![](https://img-blog.csdn.net/20150921225357857)
![](https://img-blog.csdn.net/20170109194713802?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc29mdGVl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 缺点
a：如果出入越长的话，展开的网络就越深，对于“深度”网络训练的困难最常见的是“梯度爆炸( Gradient Explode )” 和 “梯度消失( Gradient Vanish )” 的问题。

b：Simple-RNN善于基于先前的词预测下一个词，但在一些更加复杂的场景中，例如，“我出生在法国......我能将一口流利的法语。” “法国”和“法语”则需要更长时间的预测，而随着上下文之间的间隔不断增大时，Simple-RNN会丧失学习到连接如此远的信息的能力。

### 与一维卷积的区别

### RNN 梯度爆炸/梯度消失
+ https://www.zhihu.com/question/34878706
+ LSTM只能避免RNN的梯度消失（gradient vanishing）
	+ 将求导链式法则中的连乘部分变成累加部分
+ 梯度膨胀(gradient explosion)不是个严重的问题，一般靠裁剪后的优化算法即可解决，比如gradient clipping，果梯度的范数大于某个给定值，将梯度同比收缩

+ 有关避免梯度消失的方法，可以看下He Kaiming的这篇论文：[1603.05027] Identity Mappings in Deep Residual Networks说的是Residual Net，但是和LSTM的原理一致， 都是利用一个独立的加法通道（所谓的shortcut connection），把梯度保持下去。

+ 对比

![](https://pic2.zhimg.com/80/v2-8d64e83943e31fb95af6b1845e174b49_hd.jpg)
    
+ LSTM相对普通RNN多了加和，为避免梯度消散提供了可能。线性自连接的memory是关键。
+ RNN中为什么要采用tanh而不是ReLu作为激活函数？
	+ https://www.zhihu.com/question/61265076
+ 为什么在CNN等结构中将原先的sigmoid、tanh换成ReLU可以取得比较好的效果？
  为什么在RNN中，将tanh换成ReLU不能取得类似的效果？
  + https://www.zhihu.com/question/61265076