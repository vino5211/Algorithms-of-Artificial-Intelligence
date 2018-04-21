
## Survey
- A Survey of Word Embeddings Evaluation Methods
	- https://arxiv.org/abs/1801.09536

## fine tune
- http://www.cnblogs.com/iloveai/p/word2vec.html
	- 无监督或弱监督的预训练以word2vec和auto-encoder为代表。这一类模型的特点是，不需要大量的人工标记样本就可以得到质量还不错的embedding向量。不过因为缺少了任务导向，可能和我们要解决的问题还有一定的距离。因此，我们往往会在得到预训练的embedding向量后，用少量人工标注的样本去fine-tune整个模型。

## word2vec 原理
- word2vec原理(一) CBOW与Skip-Gram模型基础
	- http://www.cnblogs.com/pinard/p/7160330.html
- word2vec原理(二) 基于Hierarchical Softmax的模型
	- http://www.cnblogs.com/pinard/p/7243513.html
- word2vec原理(三) 基于Negative Sampling的模型
	- http://www.cnblogs.com/pinard/p/7249903.html

## Gensim 训练 word2vec
- Gensim进阶教程：训练word2vec与doc2vec模型
	- http://www.cnblogs.com/iloveai/p/gensim_tutorial2.html

## Word Embedding
- word2vec
	- (侧重训练过程)http://blog.csdn.net/zhoubl668/article/details/24314769
	- (侧重原理 NNLM )http://www.cnblogs.com/iloveai/p/word2vec.html
- word2vec 与 Glove 的区别
	- https://zhuanlan.zhihu.com/p/31023929
	- word2vec是“predictive”的模型，而GloVe是“count-based”的模型
	- Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram。对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测（这一块有些大公司面试会问到，为什么用Huffuman树，大家可以思考一下）。
	- Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理)。
	- http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf
- CS224n笔记3 高级词向量表示
	- http://www.hankcs.com/nlp/cs224n-advanced-word-vector-representations.html#h3-5


## Document Represent(文档表示)
- DisSent: Sentence Representation Learning from Explicit Discourse Relations
	- 借助文档中一些特殊的词训练句子 embedding。使用文档中 but、because、although 等词，以及其前后或关联的句子构成语义模型。也就是，使用这些词和句子的关系，约束了句子向量的生成空间（使用句子向量，预测关联词），从而达到训练句子向量目的。
  	- 文章只对英文语料进行了测试，实际中文这样的结构也很多，如：因为、所以、虽然、但是，可以参考。
   	- 论文链接：https://www.paperweekly.site/papers/1324
- Multilingual Hierarchical Attention Networks for Document Classification
	- 本文使用两个神经网络分别建模句子和文档，采用一种自下向上的基于向量的文本表示模型。首先使用 CNN/LSTM 来建模句子表示，接下来使用双向 GRU 模型对句子表示进行编码得到文档表示。
  	- 论文链接：https://www.paperweekly.site/papers/1152**
  	- 代码链接：https://github.com/idiap/mhan**
- **Supervised Learning of Universal Sentence Representations from Natural Language Inference Data**
	- 本文来自 Facebook AI Research。本文研究监督句子嵌入，作者研究并对比了几类常见的网络架构（LSTM，GRU，BiLSTM，BiLSTM with self attention 和 Hierachical CNN）, 5 类架构具很强的代表性。
  	- 论文链接：https://www.paperweekly.site/papers/1332**
  	- 代码链接：https://github.com/facebookresearch/InferSent**

## Network Embedding
- Structural Deep Network Embedding
 	- SDNE 是清华大学崔鹏老师组发表在 2016KDD 上的一个工作，目前谷歌学术引用量已经达到了 85，是一篇基于深度模型对网络进行嵌入的方法。
 SDNE 模型同时利用一阶相似性和二阶相似性学习网络的结构，一阶相似性用作有监督的信息，保留网络的局部结构；二阶相似性用作无监督部分，捕获网络的全局结构，是一种半监督深度模型。
 	- 论文链接：https://www.paperweekly.site/papers/1142**
 	- 代码链接：https://github.com/xiaohan2012/sdne-keras
	- 《Structural Deep Network Embedding》阅读笔记
		- https://zhuanlan.zhihu.com/p/24769965?refer=c_51425207
