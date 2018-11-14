# Markov networks - CRF
- Define and Probability calculation
    + $P(x,y)\ \epsilon \ exp(w\ \cdot\ \phi (x,y) )$
        + $\phi$ (x,y) is a feature vector
        + w is the weight vector to be learned from training data
    + $P(y\ |\ x) = \frac{P(x,y)} { \sum_{y'} P(x, y')}$ 

    + Diff from HMM (P(x,y) for CRF)
        + In HMM $P(x,y) = P(y_1|\ start)\ \prod_{l=1}^{L-1} P(y_{l+1}|y_l)\ P(end|y_L)\ \prod^{L}_{l=1}P(x_l|y_l)$
        + $log P(x,y) = logP(y_1 | start) + \sum_{l=1}^{L-1}logP(y_{l+1}|y_l) + log P(end|y_L) + \sum_{l=1}^{L} logP(x_l|y_l)$
        + the last item of last formula
            + $\sum_{l=1}^{L} logP(x_l|y_l) = \sum_{s,t} log P(s|t) \times N_{s,t}(x,y)$
            + $log P(s|t)$ : Log probability of word given tag s ()
            + $N_{s,t}(x,y)$ : Number fo tag s and word t appears together in (x,y)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM1.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM2.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM3.png)
        + Define Feature Vector
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM4.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM5.png)
            ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM6.png) 

- Training
    - cost function like crosss entropy
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM7.png)
        - Maximize what we boserve in training data
        - Minimize what we dont observe in training data
    - gredient Assent
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM8.png)
    - process
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM9.png)
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM10.png)
    - right - wrong
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM11.png)


- Inference
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM12.png)
- CRF v.s. HMM
    - adjust P(a|V) -> 0.1
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM13.png)
- Synthetic Data
    - First paper purpose CRF
    - comparing HMM and CRF
    ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM14.png)

- CRF Summary
   ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM15.png) 

- CRF的定义与形式
	- 定义
	- 参数化形式
	- 简化形式
	- 矩阵形式
- 概率计算问题
	- 前向后向算法
	- 概率计算
	- 期望值计算
- 学习算法
	- 改进的迭代尺度法
	- 拟牛顿法
- 预测算法（根据观测序列预测状态序列）

## Diff
- https://www.zhihu.com/question/46688107/answer/117448674
	- LSTM：像RNN、LSTM、BILSTM这些模型，它们在序列建模上很强大，它们能够capture长远的上下文信息，此外还具备神经网络拟合非线性的能力，这些都是crf无法超越的地方，对于t时刻来说，输出层y_t受到隐层h_t（包含上下文信息）和输入层x_t（当前的输入）的影响，但是y_t和其他时刻的y_t`是相互独立的，感觉像是一种point wise，对当前t时刻来说，我们希望找到一个概率最大的y_t，但其他时刻的y_t`对当前y_t没有影响，如果y_t之间存在较强的依赖关系的话（例如，形容词后面一般接名词，存在一定的约束），LSTM无法对这些约束进行建模，LSTM模型的性能将受到限制。
	- CRF：它不像LSTM等模型，能够考虑长远的上下文信息，它更多考虑的是整个句子的局部特征的线性加权组合（通过特征模版去扫描整个句子）。关键的一点是，CRF的模型为p(y | x, w)，注意这里y和x都是序列，它有点像list wise，优化的是一个序列y = (y1, y2, …, yn)，而不是某个时刻的y_t，即找到一个概率最高的序列y = (y1, y2, …, yn)使得p(y1, y2, …, yn| x, w)最高，它计算的是一种联合概率，优化的是整个序列（最终目标），而不是将每个时刻的最优拼接起来，在这一点上CRF要优于LSTM。
	- **HMM：CRF不管是在实践还是理论上都要优于HMM，HMM模型的参数主要是“初始的状态分布”，“状态之间的概率转移矩阵”，“状态到观测的概率转移矩阵”，这些信息在CRF中都可以有，例如：在特征模版中考虑h(y1), f(y_i-1, y_i), g(y_i, x_i)等特征。**
	- CRF与LSTM：从数据规模来说，在数据规模较小时，CRF的试验效果要略优于BILSTM，当数据规模较大时，BILSTM的效果应该会超过CRF。从场景来说，如果需要识别的任务不需要太依赖长久的信息，此时RNN等模型只会增加额外的复杂度，此时可以考虑类似科大讯飞FSMN（一种基于窗口考虑上下文信息的“前馈”网络）。
	- CNN＋BILSTM＋CRF：
		- 这是目前学术界比较流行的做法，BILSTM＋CRF是为了结合以上两个模型的优点，CNN主要是处理英文的情况，英文单词是由更细粒度的字母组成，这些字母潜藏着一些特征（例如：前缀后缀特征），通过CNN的卷积操作提取这些特征，在中文中可能并不适用（中文单字无法分解，除非是基于分词后），这里简单举一个例子，例如词性标注场景，单词football与basketball被标为名词的概率较高， 这里后缀ball就是类似这种特征。
		- BILSTM+CRF的Tensorflow版本：https://github.com/chilynn/sequence-labeling，主要参考了GitHub - glample/tagger: Named Entity Recognition Tool的实现，tagger是基于theano实现的，每一轮的参数更新是基于一个样本的sgd，训练速度比较慢。sequence-labeling是基于tensorflow实现的，将sgd改成mini-batch sgd，由于batch中每个样本的长度不一，训练前需要padding，最后的loss是通过mask进行计算（根据每个样本的真实长度进行计算）。
		- 参考论文：
			- https://arxiv.org/pdf/1603.01360v3.pdf
			- https://arxiv.org/pdf/1603.01354v5.pdf
			- http://arxiv.org/pdf/1508.01991v1.pdf


