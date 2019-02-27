# Loss Function
## Reference
+ https://blog.csdn.net/u014595019/article/details/52562159
+ keras-loss
	+ https://blog.csdn.net/lien0906/article/details/78429768
	+ https://blog.csdn.net/lien0906/article/details/78429768
+ center-loss
	+ https://cloud.tencent.com/developer/article/1032585
+ 用sigmoid作为激活函数，为什么往往损失函数选用binary_crossentropy 
	+ 参考地址:https://blog.csdn.net/wtq1993/article/details/51741471
+ softmax与categorical_crossentropy的关系，以及sigmoid与bianry_crossentropy的关系。 
	+ 参考地址:https://www.zhihu.com/question/36307214
+ 各大损失函数的定义:MSE,MAE,MAPE,hinge,squad_hinge,binary_crossentropy等 
	+ 参考地址:https://www.cnblogs.com/laurdawn/p/5841192.html
+ https://blog.csdn.net/cherrylvlei/article/details/53038603
+ 交叉熵的推导
	+ https://www.cnblogs.com/baiting/p/6101969.html
+ 交叉熵代价函数(损失函数)及其求导推导
	+ https://blog.csdn.net/jasonzzj/article/details/52017438

## 二次代价函数(均方差MSE) 的缺点 与 交叉熵代价函数的改进
+ 针对s型激活函数, CE 比 MSE 会更快收敛
+ https://www.bilibili.com/video/av20542427/?p=11
	+ 推导过程有待说明
+ 交叉熵不包括激活函数的导数，更为合理

## 均方差（MSE）
+  $ C = \frac{1}{2n} \sum_{x} {|| y(x) - a^L(x)||}^2$
+  $ \frac{\partial C}{\partial w} = (a-y) \delta^{'}(z)x$
+  $ \frac{\partial C}{\partial b} = (a-y) \delta^{'}(z)$
+ 应用在实数值域连续变量的回归问题上,并且对参差较大的情况给予更多权重

## 平均绝对差(MAE)
+　应用在实数值域连续变量的回归问题上,　*在时间序列预测问题上也较为常用 *
+　在误差函数中, 每个误差对总体误差的贡献与其误差的绝对值成线性比例关系

## 交叉熵损失(Cross Entorpy)  针对 S 型激活函数
+ $ C = - \frac{1}{n} \sum_{x} [ylna + (1-y)ln(1-a)]$
+ $ \frac{\partial C}{\partial w_j} = \frac{1}{n} \sum_{x} x_j(\delta(z)-y)$
+  $ \frac{\partial C}{\partial b} = \frac{1}{n} \sum_{x}( \delta(z) - y)$
+  与  $\delta^{'}(z)$ 无关, $ \delta(z)-y$ 表示输出值与实际值的误差, 相比MSE对参数的偏导数更合理, 梯度较大, 训练速度也就越快
+  误差大时梯度变大, 调整的快, 加速训练; 误差小时梯度变小, 调整的慢, 防止震荡
+ 交叉熵的解释:映射到最可能的类别的概率的对数, 因此, 当预测值的分布和实际因变量的分布尽可能一致时,交叉熵最小

### tensorflow API
+ tf.nn.sigmoid_cross_entropy_with_logits()
	+ 和sigmoid一起使用的交叉熵
	+ 二分类
+ tf.nn.softmax_cross_entropy_with_logits()
	+ 和softmax 一起使用的交叉熵
	+ 多分类

### keras API
+ binary_crossentropy（亦称作对数损失，logloss）
+ categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
+ sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需

### demo Error
- Cross Entropy Loss（此处记录可能有误，问题在于　交叉熵是否是由信息熵推导出来的）
	- 预测概率分布与真实概率分布的相似程度
	- 对文本三分类问题：
		- 输入文本为ｘ, $y=f(x)$，输出对应类别
		- $y_i$ 为预测的类别，$\hat y_i$ 为真实的类别
		- $y_1$, $y_2$, $y_3$ 为预测到１, 2, 3 这几个类的概率, $\hat y_1$, $\hat y_2$, $\hat y_3$为真实概率   
		- 预测的信息熵为$y_1 * log \frac{1}{y_1} + y_2 * log \frac{1}{y_2} + y_3 * log \frac{1}{y_3}$
		- 真实的信息熵为$\hat y_1 * log \frac{1}{\hat y_1} + \hat y_2 * log \frac{1}{\hat y_2} + \hat y_3 * log \frac{1}{\hat y_3}$
		- $CE = - \sum_{i=1}^{3} y_i * log(\hat y_i) = y_1 * log \frac{1}{y_1} + y_2 * log \frac{1}{y_2} + y_3 * log \frac{1}{y_3} - ( \hat y_1 * log \frac{1}{\hat y_1} + \hat y_2 * log \frac{1}{\hat y_2} + \hat y_3 * log \frac{1}{\hat y_3} ) $
	- 对文本二分类问题：
		-  $y_1$, $y_2$, $\hat y_1$, $\hat y_2$
		- $CE = y_1 * log \frac{1}{y_1} + y_2 * log \frac{1}{y_2} - \hat y_1 * log \frac{1}{\hat y_1} - \hat y_2 * log \frac{1}{\hat y_2} $
	- 简化公式
		- $CE = - \sum_{i=1}^{n} y_i * log(\hat y_i) $


## tensorflow loss

## keras loss
+ mean_squared_error或mse
+ mean_absolute_error或mae
+ mean_absolute_percentage_error或mape
+ mean_squared_logarithmic_error或msle
+ squared_hinge
+ hinge
+ binary_crossentropy（亦称作对数损失，logloss）
+ categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
+ sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：np.expand_dims(y,-1)
+ kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
+ cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数