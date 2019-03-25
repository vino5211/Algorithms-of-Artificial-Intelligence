# Summary of Machine Reading Comprehension

## Target
+ Clean up current algorithms of Reading Comprehension and Question Answering
+ Read Paper and Write Note

## Reference
+ [2017年　以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
+ [2018年　清华77篇机器阅读理解论文](http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a)
+ 2019 年　有待整理

## LeaderBoard(sort by dataset)
+ SQuAD
    + BiDAF
    + r-NET
    + BERT
+ MARCO
+ LAMBADA
+ Who-did-What(WDW)
+ CNN & DailyMail(close style)
+ Children's Book Test
+ BookTest

## Mertic
+ MAP
+ MRR
    + 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0
    + 最终的分数为所有得分之和
+ NDCG
+ EM
+ F1

## Idea 
+ Open domain
    + 检索
        + 检索得到主要的关键词
    + 答案生成
        + 模型将关键词输出为一句话
        
## Doing

## Problems

## Papers(sort by type, year, dataset)
### Extractive ＱＡ
##### [BiDAF-ICLR2017](https://arxiv.org/pdf/1611.01603.pdf)
![BiDAF](https://img-blog.csdn.net/20181015145727446?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyMTEzMTg5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
+ Abstract
  + 同时计算contex2query 和 query2context, 注意力的计算在时序上是独立的,并会flow到下一层
  + 避免了过早的summary 造成的信息丢失
+ Framework
  + char embedding:使用CNN训练得到,参考Kim(有待添加)

  + word embedding:Glove

  + 字向量和词向量拼接后,过一个Highway, 得到　Context X 和 query Y

  + Contextual Embedding Layer
    + 使用Bi-LSTM整合Highway的输出，表达词间关系
    + 输出contex $ H \in R^{2d*T} $
    + query $ Q \in R^{d*J} $

  + Attention Flow Layer
    + 计算context和query词和词两两之间的相似性 
      	$$ S_{tj} = \alpha(H_{:j}, U_{:j}) ​$$
      	$$ \alpha = w^T_{S} [h;u;h \odot u] ​$$		

    + 计算context-to-query attention, 对于context中的词,按attention系数计算query中的词的 加权和 作为当前词的 **query aware representation**

      ​	$$ \alpha_t = softmax(St:) \in R^J ​$$

      ​	$$ {\widetilde U}_{:t} = \sum \alpha_{ij} U_{:j} R\in^{2d*J}  $$

    + 计算query-to-context attention, 计算 query 和 每个 context 的最大相似度, query和context的相似度是query所有词里面和context相似度最大的, 然后计算context 的加权和
      $$ b = softmax(max_{col}(S)) ​$$
      $$ \widetilde{h} = \sum_t b_t H_{:t}  \in R^{2d}​$$
      $$ \widetilde{H} = tile(\widetilde{h})  ​$$

    + final query-aware-representation of context

      ​	$$ G_{:t} = \beta(H:t, \widetilde{U}_{:t}, \widetilde{H}_{:t} ) $$

      ​	$$ \beta(h;\widetilde{u};\widetilde{h}) = [h;\widetilde{u};h\odot\widetilde{u};h\odot\widetilde{h}] \in R^{8d}​$$	

  + Modeling Layer

    + 过Bi-LSTM 得到 M

  + Output Layer
    $$ p^1 = softmax(w^T_(p1)[G;M]​$$
    $$ p^2 = softmax(w^T_(p1)[G;M_2]​$$
    $$ L(\theta) = - \frac{1}{N} \sum_i^{N} log(p^1_{y^1_i}) + log(p^2_{y^2_i})​$$

  + results

    ![SQuAD 1](https://pic2.zhimg.com/80/v2-12e684f49462f029ed79665913875a6d_hd.jpg)
    ![CNN/Dialy Mail](https://pic1.zhimg.com/80/v2-8b37a915752550f910af352c56bad5b8_hd.jpg)

##### [Matching-LSTM](https://arxiv.org/pdf/1608.07905.pdf)
![M-LSTM](https://img-blog.csdn.net/2018050321103273?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI4OTI5Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
+ Abstract
	+ LSTM 编码原文的上下文信息
	+ Match-LSTM 匹配原文和问题
	+ Answer-Pointer :　使用Ptr网络, 预测答案
		+ Sequence Model :　答案是不连续的
		+ Boundary Model : 答案是连续的, 在SQuAD数据集上 Boundary 比 Sequence 的效果要好
+ Framework
	+ LSTM Preprocessing Layer
		+ 使用embedding表示Question 和 Context, 在使用单向LSTM编码,得到hidden state 表示 
	+ Mathc-LSTM Layer
		+ 类似文本蕴含:前提H, 假设T, M-LSTM序列化的经过假设的每一个词,然后预测前提是否继承自假设
		+ 文本问答中, question 当做 H, context当做T, 可以看成带着问题去段落中找答案(利用soft-attention)
+ Results
	![](https://img-blog.csdn.net/20180503220923886?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI4OTI5Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  	  
+ Others
	+ https://github.com/shuohangwang/SeqMatchSeq

##### R-NET
+ https://blog.csdn.net/jyh764790374/article/details/80247204
+ https://blog.csdn.net/sparkexpert/article/details/79141584

![](https://www.msra.cn/wp-content/uploads/news/blogs/2017/05/images/machine-text-comprehension-20170508-4.jpg)

#### 检索式ＱＡ

#### 生成式ＱＡ

##### Search QA

##### Knowledge QA

##### Factoid QA

##### No-Factoid QA

##### Open Domain QA

##### Close Style QA

##### Language Model QA (文字生成)

##### Community QA 

##### Semantic parsing for QA


