# View of Attention

### Overview

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

+ https://blog.csdn.net/mpk_no1/article/details/72862348

  ![](https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111501343-1669960587.png)

+ http://www.cnblogs.com/guoyaohua/p/9429924.html

### Attention Mechanism in NLP

- http://www.cnblogs.com/robert-dlut/p/5952032.html

### Attention Mechanism in CV

- Attention可以分成hard与soft两种模型: 
  - hard: Attention每次移动到一个固定大小的区域
  - soft: Attention每次是所有区域的一个加权和
- Reference:  
  - Survey on Advanced Attention-based Models
  - Recurrent Models of Visual Attention (2014.06.24)  
  - Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (2015.02.10) 
  - DRAW: A Recurrent Neural Network For Image Generation (2015.05.20)
  - Teaching Machines to Read and Comprehend (2015.06.04) 
  - Learning Wake-Sleep Recurrent Attention Models (2015.09.22)
  - Action Recognition using Visual Attention (2015.10.12)
  - Recursive Recurrent Nets with Attention Modeling for OCR in the Wild (2016.03.09)

### Hard Attention

- https://blog.csdn.net/malefactor/article/details/50583474

### Soft Attention

+ Pass

### 强制前向AM

### 加性注意力（additive attention）

### 乘法（点积）注意力（multiplicative attention）

### 关键值注意力（key-value attention）

### Google mind : Recurrent Models of Visual Attention

### Bahdanau : Neural Machine Translation by Jointly Learning to Align and Translate

- soft
  - https://zhuanlan.zhihu.com/p/27766967

### Effective Approaches to Attention-based Neural Machine Translation

- Global
- Local
  - https://zhuanlan.zhihu.com/p/27766967

### Self attention

- http://www.cnblogs.com/guoyaohua/p/9429924.html
- http://www.cnblogs.com/guoyaohua/p/9429924.html

### Transformer

- https://www.sohu.com/a/168933829_465914

### Attention is all your need

	+ http://nlp.seas.harvard.edu/2018/04/03/attention.html
	+ https://github.com/harvardnlp/annotated-transformer

### Hierarchical Attention Networks for Document Classification
- Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector

### ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs