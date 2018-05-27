[TOC]

# Embedding

## Word2Vec
+ w2v 得推理过程
    + 所有词 都 进行 1 of N Encoding， 得到所有词得one-hot编码
    + 利用上下文进行训练
        + count based
            +  if two words $w_i$ and $w_j$ frequently co-occur, V($w_i$) and V($w_j$) would be close to each other
            +  Eg
                + glove
                    + find $v(w_i)$ and $v(w_i)$, inner product $v(w_i)$ and $v(w_i)$, the result should be Positive correlation to $N_{i,j}$
                    <img src="/home/apollo/Pictures/Emb1.png" width="400px" height="200px" />

        + Perdition based
            + 类比语言模型的 过程 ：Language Modeling(Machine Translate/Speech recognize)
                <img src="/home/apollo/Pictures/Emb2.png" width="400px" height="200px" />
            + 推理过程
                + 不同的词，他们的输入都是 1-of-N encoding(图中的黄色块）
                + 大蓝色块是一个神经网络，绿色块是网络得第一层
                + 黄色块乘以参数$W_i$ 后_， 得到得绿色块应该尽可能相似（因为对蓝色的网络来说，相同得输入才会产生相同得输出）
                + **获得的embedding就是绿色块**
                + 蓝色块的维度是所有词构成得词典Dic的大小，每一维度的值代表预测词是字典中某一个次得概率
                    <img src="/home/apollo/Pictures/Emb3.png" width="400px" height="200px" />
            + Sharing Parameters
                + 类似CNN
                + $W_i$ = w, 输入不同得词得时候，这个值是共享得
                + 不管输入单词数量是多少，参数的个数不会增加
                    <img src="/home/apollo/Pictures/Emb4.png" width="400px" height="200px" />
            + Various Architectures
                + CBOW
                + Skip-gram
            + 建立逻辑 build analogy
                <img src="/home/apollo/Pictures/Emb6.png" width="400px" height="200px" />
            + Multi-lingual Embedding
                + 不同的embedding无法同时使用，不同embedding 的不同维度代表的信息不一样（例如emb1 的第一维代表动物，emb2 的第7维代表动物）
                + 找一些不同语言的匹配词，使用这些匹配词去推导其他词的匹配关系

## Char2Vec

## Doc2Vec
+ Documnet Embedding
	+ the vector represent the meaning of the word sequence
	+ A word sequence can be a document and a paragraph
	+ word sequence with different lengths
	+ Semantic Embedding
		+ Bag-of-word + Auto encoder
		+ Beyond Bag of word
			+ need to be added

## Glove

## FastText
- https://blog.csdn.net/sinat_26917383/article/details/54850933

## WordRank

## Diff
- word2vec 与 Glove 的区别
    - https://zhuanlan.zhihu.com/p/31023929
    - word2vec是“predictive”的模型，而GloVe是“count-based”的模型
    - Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram。对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测（这一块有些大公司面试会问到，为什么用Huffuman树，大家可以思考一下）。
    - Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理)。
    - http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf
- FastText词向量与word2vec对比 
    - FastText= word2vec中 cbow + h-softmax的灵活使用
    - 灵活体现在两个方面： 
        1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用； 
        2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；

    - 两者本质的不同，体现在 h-softmax的使用。 
        - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。 
        - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
    - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#




