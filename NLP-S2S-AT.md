# Encoder-Decoder

# Sequence2Sequence

# Attention
## 注意：
+ 目前Keras官方还没有单独将attention模型的代码开源，下面有一些第三方的实现：
	+ Deep Language Modeling for Question Answering using Keras
	+ Attention Model Available!
	+ Keras Attention Mechanism
	+ Attention and Augmented Recurrent Neural Networks
	+ How to add Attention on top of a Recurrent Layer (Text Classification)
	+ Attention Mechanism Implementation Issue
	+ Implementing simple neural attention model (for padded inputs)
	+ Attention layer requires another PR
	+ seq2seq library
---
## 理解LSTM/RNN中的Attention机制
+ http://www.jeyzhang.com/understand-attention-in-rnn.html
+ 输入序列不论长短都会被编码成一个固定长度的向量表示，而解码则受限于该固定长度的向量表示
+ 当输入序列比较长时，模型的性能会变得很差
+ Attention机制的基本思想是，打破了传统编码器-解码器结构在编解码时都依赖于内部一个固定长度向量的限制。
+ 如果你想进一步地学习如何在LSTM/RNN模型中加入attention机制，可阅读以下论文：
	+ Attention and memory in deep learning and NLP
	+ Attention Mechanism
	+ Survey on Attention-based Models Applied in NLP
	+ What is exactly the attention mechanism introduced to RNN? （来自Quora）
	+ What is Attention Mechanism in Neural Networks?

## 基于attention模型的应用实例
	这部分将列举几个具体的应用实例，介绍attention机制是如何用在LSTM/RNN模型来进行序列预测的。
1.Attention在文本翻译任务上的应用
	文本翻译这个实例在前面已经提过了。

	给定一个法语的句子作为输入序列，需要输出翻译为英语的句子。Attention机制被用在输出输出序列中的每个词时会专注考虑输入序列中的一些被认为比较重要的词。

	我们对原始的编码器-解码器模型进行了改进，使其有一个模型来对输入内容进行搜索，也就是说在生成目标词时会有一个编码器来做这个事情。这打破了之前的模型是基于将整个输入序列强行编码为一个固定长度向量的限制，同时也让模型在生成下一个目标词时重点考虑输入中相关的信息。

	— Dzmitry Bahdanau, et al., Neural machine translation by jointly learning to align and translate, 2015



	Attention在文本翻译任务（输入为法语文本序列，输出为英语文本序列）上的可视化（图片来源于Dzmitry Bahdanau, et al., Neural machine translation by jointly learning to align and translate, 2015）

2. Attention在图片描述上的应用
	与之前启发式方法不同的是，基于序列生成的attention机制可以应用在计算机视觉相关的任务上，帮助卷积神经网络重点关注图片的一些局部信息来生成相应的序列，典型的任务就是对一张图片进行文本描述。

	给定一张图片作为输入，输出对应的英文文本描述。Attention机制被用在输出输出序列的每个词时会专注考虑图片中不同的局部信息。

	我们提出了一种基于attention的方法，该方法在3个标准数据集上都取得了最佳的结果……同时展现了attention机制能够更好地帮助我们理解模型地生成过程，模型学习到的对齐关系与人类的直观认知非常的接近（如下图）。

	— Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, 2016



	Attention在图片描述任务（输入为图片，输出为描述的文本）上的可视化（图片来源于Attend and Tell: Neural Image Caption Generation with Visual Attention, 2016）

3. Attention在语义蕴涵 (Entailment) 中的应用
	给定一个用英文描述的前提和假设作为输入，输出假设与前提是否矛盾、是否相关或者是否成立。

	举个例子：

	前提：在一个婚礼派对上拍照

	假设：有人结婚了

	该例子中的假设是成立的。

	Attention机制被用于关联假设和前提描述文本之间词与词的关系。

	我们提出了一种基于LSTM的神经网络模型，和把每个输入文本都独立编码为一个语义向量的模型不同的是，该模型同时读取前提和假设两个描述的文本序列并判断假设是否成立。我们在模型中加入了attention机制来找出假设和前提文本中词/短语之间的对齐关系。……加入attention机制能够使模型在实验结果上有2.6个点的提升，这是目前数据集上取得的最好结果…

	— Reasoning about Entailment with Neural Attention, 2016

	![](http://i.imgur.com/BTCD2NH.png)

	Attention在语义蕴涵任务（输入是前提文本，输出是假设文本）上的可视化（图片来源于Reasoning about Entailment with Neural Attention, 2016）

4. Attention在语音识别上的应用
	给定一个英文的语音片段作为输入，输出对应的音素序列。

	Attention机制被用于对输出序列的每个音素和输入语音序列中一些特定帧进行关联。

	…一种基于attention机制的端到端可训练的语音识别模型，能够结合文本内容和位置信息来选择输入序列中下一个进行编码的位置。该模型有一个优点是能够识别长度比训练数据长得多的语音输入。

	— Attention-Based Models for Speech Recognition, 2015.



	Attention在语音识别任务（输入是音帧，输出是音素的位置）上的可视化（图片来源于Attention-Based Models for Speech Recognition, 2015）

5. Attention在文本摘要上的应用
	给定一篇英文文章作为输入序列，输出一个对应的摘要序列。

	Attention机制被用于关联输出摘要中的每个词和输入中的一些特定词。

	… 在最近神经网络翻译模型的发展基础之上，提出了一个用于生成摘要任务的基于attention的神经网络模型。通过将这个概率模型与一个生成式方法相结合来生成出准确的摘要。

	— A Neural Attention Model for Abstractive Sentence Summarization, 2015

---

## Attention-based RNN in NLP
- Neural Machine Translation by Jointly Learning to Align and Translate
	- 图中我并没有把解码器中的所有连线画玩，只画了前两个词，后面的词其实都一样。可以看到基于attention的NMT在传统的基础上，它把源语言端的每个词学到的表达（传统的只有最后一个词后学到的表达）和当前要预测翻译的词联系了起来，这样的联系就是通过他们设计的attention进行的
	- 在模型训练好后，根据attention矩阵，就可以得到源语言和目标语言的对齐矩阵了
	- 具体论文的attention设计如下
	- 使用一个感知机公式来将目标语言和源语言的每个词联系了起来，然后通过soft函数将其归一化得到一个概率分布，就是attention矩阵
	- 结果来看相比传统的NMT（RNNsearch是attention NMT，RNNenc是传统NMT）效果提升了不少，最大的特点还在于它可以可视化对齐，并且在长句的处理上更有优势
- Effective Approaches to Attention-based Neural Machine Translation
	- attention在RNN中可以如何进行扩展
	- 提出了两种attention机制，一种是全局（global）机制，一种是局部（local）机制
	- global
    	- local
		- 主要思路是为了减少attention计算时的耗费，作者在计算attention时并不是去考虑源语言端的所有词，而是根据一个预测函数，先预测当前解码时要对齐的源语言端的位置Pt，然后通过上下文窗口，仅考虑窗口内的词
		- 里面给出了两种预测方法，local-m和local-p，再计算最后的attention矩阵时，在原来的基础上去乘了一个pt位置相关的高斯分布。作者的实验结果是局部的比全局的attention效果好
## Attention-based CNN in NLP
	- ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs

## Reference websites
  - http://www.cnblogs.com/robert-dlut/p/5952032.html
