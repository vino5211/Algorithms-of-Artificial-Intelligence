# 论文笔记：Machine Comprehension Using Match-LSTM and Answer Pointer

## Reference
+ https://blog.csdn.net/u012892939/article/details/80186590

+ 在Machine Comprehension（MC）任务中，早期数据库规模小，主要使用pipeline的方法；后来随着深度学习的发展，2016年，一个比较大规模的数据库出现了，即SQuAD。**该文是第一个在SQuAD数据库上测试的端到端神经网络模型。**
+ 主要结构包括两部分：Match-LSTM和Pointer-Net，并针对Pointer-Net设计了两种使用方法，序列模型（Sequence Model）和边界模型（Boundary Model）。
+ 最终训练效果好于原数据库发布时附带的手动抽取特征+LR模型。
+ Tips
	+ match-LSTM是作者早些时候在文本蕴含（textual entertainment）任务中提出的，可参考《Learning natural language inference with LSTM》
	+ 代码：https://github.com/shuohangwang/SeqMatchSeq

## 模型
![](https://img-blog.csdn.net/20180125113635568?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFkZGllMTMy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
+ 序列模型：使用Ptr-Net网络，不做连续性假设，预测答案存在与原文的每一个位置

+ 边界模型：直接使用Ptr-Net网络预测答案在原文中起始和结束位置

+ 这两种模型都分为三部分：
  + LSTM预处理层：编码原文以及上下文信息
  + match-LSTM层：匹配原文和问题
  + Ans-Ptr层：从原文中选取答案(序列模型,选取答案)

+ LSTM 预处理层

  + 使用单向LSTM:每一个时刻的隐含层向量输出只包含左侧上下文信息

+ Match-LSTM层
  + match-LSTM原先设计是用来解决文版蕴含任务：有两个句子，一个是前提H，另外一个是假设T，match-LSTM序列化地经过假设的每一个词，然后预测前提是否继承自假设。而在问答任务中，将question当做H，passage当做T，则可以看作是带着问题去段落中找答案。这里利用了attention机制（soft-attention）。
  + 对段落p中每个词，计算其关于问题q的注意力分布α，并使用该注意力分布汇总问题表示；将段落中该词的隐层表示和对应的问题表示合并，输入另一个 LSTM 编码，得到该词的 query-aware 表示。具体结构如下： 
  1. 针对passage每一个词语输出一个α向量，这个向量维度是question词长度，故而这种方法也叫做question-aware attention passage representation。 
  2. 将attention向量与原问题编码向量点乘，得到passage中第i个token的question关联信息，再与passage中第i个token的编码向量做concat，粘贴为一个向量 
  3. 最后输出到LSTM网络中。 
  4. 反向同样来一次，最后两个方向的结果拼起来。得到段落的新的表示，大小为2lxP. 

+ Ans-Ptr层

+ Answer Pointer的思想是从Pointer Net得到的， 它将 Hr 作为输入，生成答案有两种方式sequence model 和 boundary model。

+ The sequence model
  + 序列模型不限定答案的范围，即可以连续出现，也可以不连续出现，因此需要输出答案每一个词语的位置。又因答案长度不确定，因此输出的向量长度也是不确定的，需要手动制定一个终结符。假设passage长度为P，则终结符为P+1。 
  + 假设答案序列为： a=(a1,a2,…) ，其中ai为选择出来答案的词在原文passage里面的下标位置，ai∈[1,P+1], 其中第P+1 是一个特殊的字符，表示答案的终止，当预测出来的词是终止字符时，结束答案生成。 
  + 简单的方式是像机器翻译一样，直接利用LSTM做decoder处理： 
  + 这里写图片描述
  + 找到passage里面概率最大的词的就可以了。
  + 对于pointer net网络，实质上仍然是一个attention机制的应用，只不过直接将attention向量作为匹配概率输出。 
  + 这里也利用了Attention机制， 在预测第k个答案的词时，先计算出一个权重向量 βk 用来表示在[1, P+1]位置的词，各个词的权重。 
            1. 对于第k个答案，段落里各个词对应的权重： 
               这里写图片描述
            2. 将上一步得到的编码Hr与权重βk求内积，得到针对第k个答案的表示 
               这里写图片描述
        3. βk,j 就表示第k个位置的答案ak是段落中第j个词的概率。βk,j最大的一个下标就是预测的ak值 
      这里写图片描述

+ The boundary model
  + 边界模型直接假设答案在passage中连续出现，因此只需要输出起始位置s和终止位置e即可。基本结构同Sequence Model，只需要将输出向量改为两个，并去掉终结符。 
  + 答案a=(as，ae) 只有两个值 
  + 作者对于这种模型扩展了一种 search mechanism，在预测过程中，限制span的长度，然后使用全局搜索，找 p(as) × p(ae)最大的。

## 实验结果
![](https://img-blog.csdn.net/20180503220923886?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI4OTI5Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
