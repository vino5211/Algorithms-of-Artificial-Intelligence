# 论文笔记 :A fast and simple algorithm for training neural probabilistic language models
## 摘要
+ NPLMs(neural probabilistic language models) 比 N-Gram 的方法需要更多的训练次数,  中等大小的数据集也可能要花几周的时间
+ 训练NPLM在计算上是昂贵的，因为它们被明确标准化，这导致在计算对数似然梯度时不得不考虑词汇表中的所有单词(标准化是什么意思,暂时不理解)
+ 这篇文章提出了一个快速又简单的方法, 基于noise-contrastive estimation
	+ noise-contrastive estimation : 估计非标准化的连续分布(estimating unnormalized continuous distributions)
+ 在 Penn Treebank 语料上进行试验后, 发现训练次数降低了一个数量级,但模型的精度并未降低
+ 比 importance sampling更好, 因为需要的噪声样本更少 (importance sampling 不知道)
+ 在Microsoft Research Sentence Completion Challenge dataset 上取得了SOTA(state of the art)