# Recurrent Neural Network

## Papers
- Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN
	- 本文使用 ReLU 等非饱和激活函数使网络变得更具有鲁棒性，可以处理很长的序列（超过 5000 个时间步），可以构建很深的网络（实验中用了 21 层）。在各种任务中取得了比 LSTM 更好的效果。
	- 论文链接：https://www.paperweekly.site/papers/1757
	- 代码链接：https://github.com/batzner/indrnn

## Recurrent Network
## Feedforward v.s. Recurrent

## GRU
## GRU -> Highway Network 
+ Training very deep networks
## Residual Network
+ Deep Residual Learning for image recoognition

## LSTM
+ 有详细的数学推到：零基础入门深度学习(6) - 长短时记忆网络(LSTM)
	+ https://www.zybuluo.com/hanbingtao/note/581764
## AT-LSTM
+ Attention LSTM
+ https://zhuanlan.zhihu.com/p/23615176
## Grid LSTM
## Tree LSTM

## Recursive Network
## Matrix-Vector Recursive Network

## Application : Sentiment Analysis
+ Recurrent Structure
	+ word sequence as input
	+ use rnn represent the sentence embedding
+ Recursive Structure
	+ 类似 syntatic structure(文法结构)
	+ 按照文法结构的顺序两两结合
	+ Recursive Neural Tensro Network

## Application : Sentence relatedness(句子是不是同样的意思)
+ Recursive Neural Network

---

## Detail
### RNN
![](https://pic4.zhimg.com/80/2a37bd4e9b12bcc19e045eaf22fea4e5_hd.jpg)

### RNN 梯度爆炸/梯度消失
+ https://www.zhihu.com/question/34878706
+ LSTM只能避免RNN的梯度消失（gradient vanishing）
	+ 将求导链式法则中的连乘部分变成累加部分
+ 梯度膨胀(gradient explosion)不是个严重的问题，一般靠裁剪后的优化算法即可解决，比如gradient clipping，果梯度的范数大于某个给定值，将梯度同比收缩

+ 有关避免梯度消失的方法，可以看下He Kaiming的这篇论文：[1603.05027] Identity Mappings in Deep Residual Networks说的是Residual Net，但是和LSTM的原理一致， 都是利用一个独立的加法通道（所谓的shortcut connection），把梯度保持下去。

+ 对比
	![](https://pic2.zhimg.com/80/v2-8d64e83943e31fb95af6b1845e174b49_hd.jpg)
    LSTM相对普通RNN多了加和，为避免梯度消散提供了可能。线性自连接的memory是关键。
+ RNN中为什么要采用tanh而不是ReLu作为激活函数？
	+ https://www.zhihu.com/question/61265076
+ 为什么在CNN等结构中将原先的sigmoid、tanh换成ReLU可以取得比较好的效果？
  为什么在RNN中，将tanh换成ReLU不能取得类似的效果？
  + https://www.zhihu.com/question/61265076

### units/输出维度/隐藏大小
+ 输入：每个时刻的输入都是一个向量，它的长度是输入层神经元的个数（units）。在你的问题中，这个向量就是embedding向量。它的长度与时间步的个数（即句子的长度）没有关系。
+ 输出：每个时刻的输出是一个概率分布向量，其中最大值的下标决定了输出哪个词。
+ units 含义:Keras中使用LSTM层时设置的units参数是什么
	+ https://www.cnblogs.com/bnuvincent/p/8280541.html
	```
    keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
    ```
    ```
    model = Sequential()

    model.add(LSTM(32, return_sequences=True, stateful=True,batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(num_classes, activation='softmax'))

 	类似上述代码中，数字'32'的含义。
    ```