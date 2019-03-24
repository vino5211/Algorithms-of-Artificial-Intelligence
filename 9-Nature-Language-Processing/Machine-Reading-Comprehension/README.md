# Summary of Machine Reading Comprehension

## Target
+ Clean up current algorithms of Reading Comprehension and Question Answering
+ Read Paper and Write Note

## Reference
+ [2017 年以前的论文和数据集整理](https://www.zybuluo.com/ShawnNg/note/622592)
+ 清华77篇机器阅读理解论文
    + http://www.zhuanzhi.ai/document/87418ceee95a21622d1d7a21f71a894a

## Dataset
+ SQuAD
+ LAMBADA
+ Who-did-What(WDW)
+ CNN & DailyMail
+ Children's Book Test
+ BookTest
+ MARCO

## Mertic
+ MAP
+ MRR
    + 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和
+ NDCG
+ EM
+ F1

## Idea 
+ Memory Network
+ Attention

+ Open domain
    + 检索
        + 检索得到主要的关键词
    + 答案生成
        + 模型将关键词输出为一句话
        
## Doing

## Problems


## Papers(sort by year)

#### BiDAF
#### R-NET
+ https://blog.csdn.net/jyh764790374/article/details/80247204
+ https://blog.csdn.net/sparkexpert/article/details/79141584

![](https://www.msra.cn/wp-content/uploads/news/blogs/2017/05/images/machine-text-comprehension-20170508-4.jpg)

#### Matching-LSTM
+ Learning natural language inference with LSTM
+ https://github.com/shuohangwang/SeqMatchSeq
#### Ptr-Net
+ sequence model
+ boundary model

#### FastQA

#### QANet


## Papers(sort by type)
##### Search QA

##### Knowledge QA

##### Factoid QA

##### No-Factoid QA

##### Open Domain QA

##### Close Style QA

##### Language Model QA (文字生成)

##### Community QA 

##### Semantic parsing for QA

## Projects(sort by dataset)
+ SQuAD
    + BiDAF
    + r-NET
    + BERT
+ MARCO
    + 

