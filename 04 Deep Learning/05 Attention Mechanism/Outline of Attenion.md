# Outline of Attention

### Reference
+ http://www.cnblogs.com/robert-dlut/p/5952032.html
+ Hard Attention
	+ https://blog.csdn.net/malefactor/article/details/50583474
+ self Attention
	+ http://www.cnblogs.com/guoyaohua/p/9429924.html
### Paper List of Attention
+ Google mind : Recurrent Models of Visual Attention
+ Bahdanau : Neural Machine Translation by Jointly Learning to Align and Translate
	+ soft
		+ https://zhuanlan.zhihu.com/p/27766967
+ Effective Approaches to Attention-based Neural Machine Translation
	+ Global
	+ Local
		+ https://zhuanlan.zhihu.com/p/27766967
+ ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs

### Outline
+ Attention机制最早是在视觉图像领域提出来的，应该是在九几年思想就提出来了
+ 但是真正火起来应该算是google mind团队的这篇论文《Recurrent Models of Visual Attention》[14]，他们在RNN模型上使用了attention机制来进行图像分类
+ 随后，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》 [1]中，使用类似attention的机制在机器翻译任务上将翻译和对齐同时进行，他们的工作算是是第一个提出attention机制应用到NLP领域中
+ 接着类似的基于attention机制的RNN模型扩展开始应用到各种NLP任务中。
+ 最近，如何在CNN中使用attention机制也成为了大家的研究热点
+ 下图表示了attention研究进展的大概趋势

![](https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111501343-1669960587.png)

### Reference
https://www.zhihu.com/question/36591394

### Outline 
+ NLP里面有一类典型的natural language generation问题：
	+ 给定一段上下文(context)， 生成一段与context相关的目标文本(target)。
+ 典型的例子包括：
	+ 机器翻译：context是英文，需要生成对应的中文
	+ 摘要生成：context是新闻内容， 需要生成新闻标题或者摘要
	+ 阅读理解：context是一段文章和一个选择题，需要输出答案
+ 这类问题的核心是要对 条件概率 P(target|context) 进行建模

### 相关研究
+ Attention严格意义上讲是一种idea，而不是某一个model的实现。
+ 用到该思路的论文 [2,3,4,7] 的实现方式也可以完全不同。
+ 例如，在Alexander Rush最近的Summarization Paper中 [3]，就完全没有使用RNN对context编码，而是直接将context中的单词作一次linear projection，并smooth一下。其计算attention vector的方式也没有用到neural activations比如tanh， 而是直接bi-linear scoring + softmax normalization。+ 相比之下，[2][4]和[7]的实现更为"deep learning"，甚至attention vector本身也是recurrent的。
+ [1] Sequence to Sequence Learning using Neural Networks
+ [2] Reasoning about Neural Attention
+ [3] A Neural Attention Model for Abstractive Sentence Summarization
+ [4] Neural Machine Translation by Jointly Learning to Align and Translate
+ [5] Recurrent Continuous Translation Models
+ [6] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
+ [7] Teaching Machines to Read and Comprehend

### 相关论文 in 图像
+ Attention在图像领域中，物体识别、主题生成上效果都有提高。
+ Attention可以分成hard与soft两种模型: 
	+ hard: Attention每次移动到一个固定大小的区域
	+ soft: Attention每次是所有区域的一个加权和
+ 相关博客与论文有:  
	+ Survey on Advanced Attention-based Models
	+ Recurrent Models of Visual Attention (2014.06.24)  
	+ Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (2015.02.10) 
	+ DRAW: A Recurrent Neural Network For Image Generation (2015.05.20)
	+ Teaching Machines to Read and Comprehend (2015.06.04) 
	+ Learning Wake-Sleep Recurrent Attention Models (2015.09.22)
	+ Action Recognition using Visual Attention (2015.10.12)
	+ Recursive Recurrent Nets with Attention Modeling for OCR in the Wild (2016.03.09)


### Links
[1] Cho et al., 2014 . Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
[2] Sutskever et al., 2014. Sequence to Sequence Learning with Neural Networks
[3] Bahdanau et al., 2014. Neural Machine Translation by Jointly Learning to Align and Translate
[4] Jean et. al., 2014. On Using Very Large Target Vocabulary for Neural Machine Translation
[5] Vinyals et. al., 2015. A Neural Conversational Model[J]. Computer Science
[6] Effective Approaches to Attention-based Neural Machine Translation

### self attention
+ http://www.cnblogs.com/guoyaohua/p/9429924.html