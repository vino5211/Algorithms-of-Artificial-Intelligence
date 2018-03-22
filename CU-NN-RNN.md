# Recurrent Neural Network

## papers
- Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN
	- 本文使用 ReLU 等非饱和激活函数使网络变得更具有鲁棒性，可以处理很长的序列（超过 5000 个时间步），可以构建很深的网络（实验中用了 21 层）。在各种任务中取得了比 LSTM 更好的效果。
	- 论文链接：https://www.paperweekly.site/papers/1757
	- 代码链接：https://github.com/batzner/indrnn

---
## RNN

## RNN 梯度爆炸/梯度消失
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

## LSTM

## BiLSTM

## Tree LSTM

## Grid LSTM

## GRU

## Stacked(???)

## AT-LSTM
+ Attention LSTM
+ https://zhuanlan.zhihu.com/p/23615176
