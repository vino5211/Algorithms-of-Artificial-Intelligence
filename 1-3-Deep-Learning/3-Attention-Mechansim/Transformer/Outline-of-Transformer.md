# Outline of Transformer

## Reference

+ paper
  + Attention is All you nedd
    + https://arxiv.org/pdf/1706.03762.pdf
  + Universal Transformer
    + https://arxiv.org/pdf/1807.03819.pdf
    + https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer.py

+ https://www.cnblogs.com/jiangxinyang/p/10210813.html
+ https://www.cnblogs.com/jiangxinyang/p/10069330.html
+ https://kexue.fm/archives/4765
+ https://blog.csdn.net/malefactor/article/details/78767781
+ The Illustrated Transformer
  + https://jalammar.github.io/illustrated-transformer/
+ The annotated Transformer
  + http://nlp.seas.harvard.edu/2018/04/03/attention.html
+ https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3
+ https://zhuanlan.zhihu.com/p/42213742



## Seq2Seq + Attention

+ https://img2018.cnblogs.com/blog/1335117/201812/1335117-20181205134951131-1214038638.png



## Overview

+ Transformer 来自 (Attention is all you nedd)

+ 抛弃了之前 Encoder-Decoder 模型中 的CNN/RNN，只用Attention 来实现

+ 引入self-attention

+ Transformer 的整个架构就是层叠的self-attention 和 全连接层

+ 左侧 Encoder， 右侧Decoder

+ 文本分类中一般只使用Encoder

+ Decoder 主要用于 NLG

  ![](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png)



## self-attention

+ 一般的attention中

  $$ Attention(query, source) = \sum_{i=1}^{Len(x)} Similarity(query, key_i) * value_i​$$

+ 在 self-attention 中 认为 query=key=value， 内部做attention，寻找内部联系

+ 优点

  + 可并行化处理，不依赖其他结果
  + 计算复杂度低，self-attention 的计算复杂度是 $n*n*d​$ ,  而RNN 是 $n*d*d​$ ,  这里n 是指序列长度， d指词向量的维度，一般d>n
  + self-Attention可以很好的捕获全局信息，无论词的位置在哪，词之间的距离都是1，因为计算词之间的关系时是不依赖于其他词的。在大量的文献中表明，self-Attention的长距离信息捕捉能力和RNN相当，远远超过CNN（CNN主要是捕捉局部信息，当然可以通过增加深度来增大感受野，但实验表明即使感受野能涵盖整个句子，也无法较好的捕捉长距离的信息）

## Scaled dot-product attention

+ 公式

  $$ Attention(Q,K,V) = softmax(QK^T/\sqrt d_k) * V​$$

+ Q  和 K 的向量维度都是$$d_k$$, V 的向量维度是$$d_v$$

+ 1

+ 使用点积计算相似度

+ 然而点积的方法面临一个问题，当 $$\sqrt{d_k}$$太大时，点积计算得到的内积会太大，这样会导致softmax的结果非0即1，因此引入了$$\sqrt{d_k}$$来对内积进行缩放 

## Multi-Head Attention

+ 公式

  $$ MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h) * W^O$$

  $$where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$

+ 计算过程如下

  + 假设头数是h， 首先按照每一时序上的向量长度（如果是词向量的形式输入，可任务是embedding size）等分成h份
  + 然后将上面等分后h份数据通过权重(W_i)