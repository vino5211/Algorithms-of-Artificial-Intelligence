# Reference
+ 25个深度学习开源数据集，good luck !(SOTA)
+ https://machinelearningmastery.com/datasets-natural-language-processing/
+ 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
+ https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality
+ wikidata
  + https://www.wikidata.org/wiki/Wikidata:Main_Page
+ project on github
  + Datasets for Natural Language Processing
    - [Question Answering](https://github.com/karthikncode/nlp-datasets#question-answering)
    - [Dialogue Systems](https://github.com/karthikncode/nlp-datasets#dialogue-systems)
    - [Goal-Oriented Dialogue Systems](https://github.com/karthikncode/nlp-datasets#goal-oriented-dialogue-systems)

# Text-CLF

### Kaggle Twitter Sentiment Analysis

### Reuters Newswire Topic Classification(Reuters-21578)

- http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

- 一系列1987年在路透上发布的按分类索引的文档。同样可以看RCV1，RCV2，以及TRC2
  - http://trec.nist.gov/data/reuters/reuters.html

### IMDB Reviews

+ 这是一个电影爱好者的梦寐以求的数据集。它意味着二元情感分类，并具有比此领域以前的任何数据集更多的数据。除了训练和测试评估示例之外，还有更多未标记的数据可供使用。包括文本和预处理的词袋格式。
+ 大小：80 MB
+ 记录数量：25,000个高度差异化的电影评论用于训练，25,000个测试
+ SOTA：Learning Structured Text Representations
+ IMDB Movie Review Sentiment Classification (Stanford)
  - http://ai.stanford.edu/~amaas/data/sentiment/c
+ 一系列从网站imdb.com上摘取的电影评论以及他们的积极或消极的情感。
  - News Group Movie Review Sentiment Classification (cornell)
    - http://www.cs.cornell.edu/people/pabo/movie-review-data/

#### Twenty Newsgroups

+ 顾名思义，该数据集包含有关新闻组的信息。为了选择这个数据集，从20个不同的新闻组中挑选了1000篇新闻文章。这些文章具有一定特征，如主题行，签名和引用。
+ 大小：20 MB
+ 记录数量：来自20个新闻组的20,000条消息
+ DOTA:Very Deep Convolutional Networks for Text Classification

### Sentiment140

+ Sentiment140是一个可用于情感分析的数据集。一个流行的数据集，非常适合开始你的NLP旅程。情绪已经从数据中预先移除。
+ Sentiment140是一个可用于情感分析的数据集。一个流行的数据集，非常适合开始你的NLP旅程。情绪已经从数据中预先移除。最终的数据集具有以下6个特征：
  + 推文的极性
  + 推文的ID
  + 推文的日期
  + 问题
  + 推文的用户名
  + 推文的文本
+ 大小：80 MB（压缩）
+ 记录数量：160,000条推文
+ SOTA:Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets

### 更多的信息，可以从这篇博文中获取：

- Datasets for single-label text categorization
  - http://ana.cachopo.org/datasets-for-single-label-text-categorization

# Language Modeling

语言模型涉及建设一个统计模型来根据给定的信息，预测一个句子中的下一个单词，或者一个单词中的下一个字母。这是语音识别或者机器翻译等任务的前置任务。

### Project Gutenberg

- https://www.gutenberg.org/

### Brown University Standard Corpus of Present-Day American English

- https://en.wikipedia.org/wiki/Brown_Corpus

### Google 1 Billion Word Corpus

- https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark

### Microsoft Research entence Completion Challenge dataset

- SOTA  : A fast and simple algorithm for training neural probabilistic language models



# Sequence Labeling

- 词性标注/命名实体识别 数据集
  - https://www.zhihu.com/question/52756127
- 国内可用免费语料库
  - http://www.cnblogs.com/mo-wang/p/4444858.html
- 一个中文的标注语料库。可用于训练HMM模型。
  - https://github.com/liwenzhu/corpusZh
- 可以提供给题主两份相对较新的中文分词语料
  - 第一份是SIGHAN的汉语处理评测的Bakeoff语料，从03年起首次进行评测，评测的内容针对汉语分词的准确性和合理性，形成Bakeoff 2005评测集，包含简、繁体中文的训练集和测试集，训练集有四个，单句量在1.5W~8W+。内容比较偏向于书面语。后面05 07年分别对中文命名实体识别和词性标注给出了评测。Bakeoff 2005中文分词熟语料传送门：Second International Chinese Word Segmentation Bakeoff
  - 第二份语料来自GitHub作者liwenzhu，于14年发布于GitHub，总词汇量在7400W+，可以用于训练很多模型例如Max Entropy、CRF、HMM......，优点是这份语料在分词基础上还做了词性标注，至于准确性还有待考究。传送门：liwenzhu/corpusZh
- zh seg
  - http://sighan.cs.uchicago.edu/bakeoff2005/
    - icwb2
      - http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.rar

# Information Retrieval/Ranking

- [LETOR, ](http://blog.crackcell.com/posts/2011/12/17/a_short_intro_2_ltr.html#sec-7)<http://research.microsoft.com/en-us/um/beijing/projects/letor/>
- Microsoft Learning to Rank Dataset, <http://research.microsoft.com/en-us/projects/mslr/>
- Yahoo Learning to Rank Challenge, <http://webscope.sandbox.yahoo.com/>

# Question Answering

### Papers with dataset

- **(NLVR)** A Corpus of Natural Language for Visual Reasoning, 2017 [[paper\]](http://yoavartzi.com/pub/slya-acl.2017.pdf) [[data\]](http://lic.nlp.cornell.edu/nlvr)
- **(MS MARCO)** MS MARCO: A Human Generated MAchine Reading COmprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09268) [[data\]](http://www.msmarco.org/)
- **(NewsQA)** NewsQA: A Machine Comprehension Dataset, 2016 [[paper\]](https://arxiv.org/abs/1611.09830) [[data\]](https://github.com/Maluuba/newsqa)
- **(SQuAD)** SQuAD: 100,000+ Questions for Machine Comprehension of Text, 2016 [[paper\]](http://arxiv.org/abs/1606.05250) [[data\]](http://stanford-qa.com/)
- **(GraphQuestions)** On Generating Characteristic-rich Question Sets for QA Evaluation, 2016 [[paper\]](http://cs.ucsb.edu/~ysu/papers/emnlp16_graphquestions.pdf) [[data\]](https://github.com/ysu1989/GraphQuestions)
- **(Story Cloze)** A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories, 2016 [[paper\]](http://arxiv.org/abs/1604.01696) [[data\]](http://cs.rochester.edu/nlp/rocstories)
- **(Children's Book Test)** The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations, 2015 [[paper\]](http://arxiv.org/abs/1511.02301) [[data\]](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz)
- **(SimpleQuestions)** Large-scale Simple Question Answering with Memory Networks, 2015 [[paper\]](http://arxiv.org/pdf/1506.02075v1.pdf) [[data\]](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz)
- **(WikiQA)** WikiQA: A Challenge Dataset for Open-Domain Question Answering, 2015 [[paper\]](http://research.microsoft.com/pubs/252176/YangYihMeek_EMNLP-15_WikiQA.pdf) [[data\]](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx)
- **(CNN-DailyMail)** Teaching Machines to Read and Comprehend, 2015 [[paper\]](http://arxiv.org/abs/1506.03340) [[code to generate\]](https://github.com/deepmind/rc-data) [[data\]](http://cs.nyu.edu/~kcho/DMQA/)
- **(QuizBowl)** A Neural Network for Factoid Question Answering over Paragraphs, 2014 [[paper\]](https://www.cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf) [[data\]](https://www.cs.umd.edu/~miyyer/qblearn/index.html)
- **(MCTest)** MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text, 2013 [[paper\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/MCTest_EMNLP2013.pdf) [[data\]](http://research.microsoft.com/en-us/um/redmond/projects/mctest/data.html)[[alternate data link\]](https://github.com/mcobzarenco/mctest/tree/master/data/MCTest)
- **(QASent)** What is the Jeopardy model? A quasisynchronous grammar for QA, 2007 [[paper\]](http://homes.cs.washington.edu/~nasmith/papers/wang+smith+mitamura.emnlp07.pdf) [[data\]](

### what

+ **http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz)**

### MCTest

### Algebra

### Science

### Stanford Question Answering Dataset (SQuAD)

- official website
  - https://rajpurkar.github.io/SQuAD-explorer/
- reference
  - EMNLP2016 SQuAD:100,000+ Questions for Machine Comprehension of Text
    - https://arxiv.org/pdf/1606.05250.pdf
  - SQuAD，斯坦福在自然语言处理的野心
    - http://blog.csdn.net/jdbc/article/details/52514050
- SOTAs
  - Hybrid AoA Reader (ensemble)
    - Joint Laboratory of HIT and iFLYTEK Research
  - r-net + 融合模型
    - Microsoft Research Asia
  - SLQA + 融合模型
    - Alibaba iDST NLP
- detail
  - 这个竞赛基于SQuAD问答数据集，考察两个指标：EM和F1。
  - EM是指精确匹配，也就是模型给出的答案与标准答案一模一样；
  - F1，是根据模型给出的答案和标准答案之间的重合度计算出来的，也就是结合了召回率和精确率。
  - 目前阿里、微软团队并列第一，其中EM得分微软（r-net+融合模型）更高，F1得分阿里（SLQA+融合模型）更高。但是他们在EM成绩上都击败了“人类表现”
  - 一共有107,785问题，以及配套的 536 篇文章
    - 数据集的具体构建如下：
      ​    1. 文章是随机sample的wiki百科，一共有536篇wiki被选中。而每篇wiki，会被切成段落，最终生成了23215个自然段。之后就对这23215个自然段进行阅读理解，或者说自动问答。
      ​    2. 之后斯坦福，利用众包的方式，进行了给定文章，提问题并给答案的人工标注。他们将这两万多个段落给不同人，要求对每个段落提五个问题。
      ​    3. 让另一些人对提的这个问题用文中最短的片段给予答案，如果不会或者答案没有在文章中出现可以不给。之后经过他们的验证，人们所提的问题在问题类型分布上足够多样，并且有很多需要推理的问题，也就意味着这个集合十分有难度。如下图所示，作者列出了该数据集答案的类别分布，我们可以看到 日期，人名，地点，数字等都被囊括，且比例相当。
      ​    4. 这个数据集的评测标准有两个：
      ​        第一：F1
      ​    ​    第二：EM。
      ​        ​        EM是完全匹配的缩写，必须机器给出的和人给出的一样才算正确。哪怕有一个字母不一样，也会算错。而F1是将答案的短语切成词，和人的答案一起算recall，Precision和F1，即如果你match了一些词但不全对，仍然算分。
      ​    5. 为了这个数据集，他们还做了一个baseline，是通过提特征，用LR算法将特征组合，最终达到了40.4的em和51的f1。而现在IBM和新加坡管理大学利用深度学习模型，均突破了这个算法。可以想见，在不远的将来会有更多人对阅读理解发起挑战，自然语言的英雄也必将诞生。甚至会有算法超过人的准确度。

### MS MARCO

- 相比SQuAD，MARCO的挑战难度更大，因为它需要测试者提交的模型具备理解复杂文档、回答复杂问题的能力。
- 据了解，对于每一个问题，MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到。这对机器阅读理解提出了更高的要求，需要机器具备综合理解多文档信息、聚合生成问题答案的能力。

### NarrativeQA

- Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
  - https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
  - https://github.com/deepmind/narrativeqa
  - DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。

### The NarrativeQA Reading Comprehension Challenge

- 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
- 论文链接：https://www.paperweekly.site/papers/1397
- 代码链接：https://github.com/deepmind/narrativeqa

### SougoQA

- http://task.www.sogou.com/cips-sogou_qa/

### Graph Questions 

- On Generating Characteristic-rich Question Sets for QA Evaluation
- 文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。
  - 数据集特点：
    1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
    2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
    3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
    4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
    5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。
  - 论文链接：http://www.paperweekly.site/papers/906
  - 数据集链接：https://github.com/ysu1989/GraphQuestions

### LSDSem 2017 Shared Task: The Story Cloze Test

- Story Cloze Test：人工合成的完形填空数据集。
- 论文链接：http://www.paperweekly.site/papers/917
- 数据集链接：http://cs.rochester.edu/nlp/rocstories/

### Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering

- 百度深度学习实验室创建的中文开放域事实型问答数据集。
- 论文链接：http://www.paperweekly.site/papers/914
- 数据集链接：http://idl.baidu.com/WebQA.html

### Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems

- DeepMind 和牛津大学共同打造的代数问题数据集 AQuA（Algebra Question Answering）。
- 论文链接：http://www.paperweekly.site/papers/913
- 数据集链接：https://github.com/deepmind/AQuA

### Teaching Machines to Read and Comprehend

- DeepMind Q&A Dataset 是一个经典的机器阅读理解数据集，分为两个部分：
  1. CNN：~90k 美国有线电视新闻网（CNN）的新闻文章，~380k 问题；
  2. Daily Mail：~197k DailyMail 新闻网的新闻文章（不是邮件正文），~879k 问题。
- 论文链接：http://www.paperweekly.site/papers/915
- 数据集链接：http://cs.nyu.edu/~kcho/DMQA/

### Semantic Parsing on Freebase from Question-Answer Pairs

- 文章发表在 EMNLP-13，The Stanford NLP Group 是世界领先的 NLP 团队。
- 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的
- 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣）
- 有 6642 个问答对
- 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system。
- 论文链接：http://www.paperweekly.site/papers/827
- 数据集链接：http://t.cn/RWPdQQO

### A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories

- ROCStories dataset for story cloze test.
- 论文链接：http://www.paperweekly.site/papers/918
- 数据集链接：http://cs.rochester.edu/nlp/rocstories/

### MoleculeNet:  Benchmark for Molecular Machine Learning

- 一个分子机器学习 benchmark，最喜欢看到这种将机器学习应用到传统学科领域了。
- 论文链接：http://www.paperweekly.site/papers/862
- 数据集链接：http://t.cn/RWPda8r

### 关于维基百科文章的问答:Deepmind Question Answering Corpus

- https://github.com/deepmind/rc-data

### 关于亚马逊产品的问答

- Amazon question/answer data
  - http://jmcauley.ucsd.edu/data/amazon/qa/

### Looking Beyond the surface: A Chanllenge Set for Reading Comprehension over Multiple Sentences

- 特点:
  - 多选题
  - 问题的答案来自篇章中的多条语句
  - 数据集来自7个不同的领域
- 基准算法:
  - Random
  - IR
  - SurfaceIR
  - SemanticLP
  - BiDAF
- SOTA
  - SurfaceIR 结构的F1 值 相较人类结果 差 20个百分点

### 更多信息

- Datasets: How can I get corpus of a question-answering website like Quora or Yahoo Answers or Stack Overflow for analyzing answer quality?
  https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality

# Dialogue System

### DSTC

+ The Dialog State Tracking Challenge (DSTC) is an on-going series of research community challenge tasks. Each task released dialog data labeled with dialog state information, such as the user’s desired restaurant search query given all of the dialog history up to the current turn. The challenge is to create a “tracker” that can predict the dialog state for new dialogs. In each challenge, trackers are evaluated using held-out dialog data.

#### DSTC1

+ DSTC1 used human-computer dialogs in the bus timetable domain. Results were presented in a special session at [SIGDIAL 2013](http://www.sigdial.org/workshops/sigdial2013/). DSTC1 was organized by Jason D. Williams, Alan Black, Deepak Ramachandran, Antoine Raux.

+ Data : https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads
+ Project:
  + pass
  + pass

#### DSTC2 and DSTC3

+ DSTC2/3 used human-computer dialogs in the restaurant information domain. Results were presented in special sessions at [SIGDIAL 2014](http://www.sigdial.org/workshops/conference15/) and [IEEE SLT 2014](http://www.slt2014.org/). DSTC2 and 3 were organized by Matthew Henderson, Blaise Thomson, and Jason D. Williams.

+ Data : http://camdial.org/~mh521/dstc/
+ Project:
  + pass
  + pass

#### DSTC4

+ DSTC4 used human-human dialogs in the tourist information domain. Results were presented at [IWSDS 2015](http://www.iwsds.org/). DSTC4 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, and Jason D. Williams.
+ Data:
  + http://www.colips.org/workshop/dstc4/data.html
+ Project:
  + pass

#### DSTC5

+ DSTC5 used human-human dialogs in the tourist information domain, where training dialogs were provided in one language, and test dialogs were in a different language. Results were presented in a special session at [IEEE SLT 2016](http://www.slt2016.org/). DSTC5 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, Jason D. Williams, and Koichiro Yoshino.
+ Data:
  + http://workshop.colips.org/dstc5/data.html
+ Project:
  + Pass

#### DSTC6

+ DSTC6 consisted of 3 parallel tracks: 
  + End-to-End Goal Oriented Dialog Learning
  + End-to-End Conversation Modeling
  + Dialogue Breakdown Detection. 
+ Results will be presented at a workshop immediately after NIPS 2017.
+  DSTC6 is organized by Chiori Hori, Julien Perez, Koichiro Yoshino, and Seokhwan Kim. 
+ Tracks were organized by Y-Lan Boureau, Antoine Bordes, Julien Perez, Ryuichi Higashinaka, Chiori Hori, and Takaaki Hori.

### Ubuntu Dialogue Corpus

- The Ubuntu Dialogue Corpus : A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems, 2015 [[paper\]](http://arxiv.org/abs/1506.08909) [[data\]](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

### Goal-Oriented Dialogue Corpus

- **(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016 [[paper\]](https://arxiv.org/abs/1704.00057) [[data\]](http://datasets.maluuba.com/Frames)
- **(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013 [[paper\]](http://camdial.org/~mh521/dstc/downloads/handbook.pdf) [[data\]](http://camdial.org/~mh521/dstc/)

### Standford

+ A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
  + Mihail Eric and Lakshmi Krishnan and Francois Charette and Christopher D. Manning. 2017. Key-Value Retrieval Networks for Task-Oriented Dialogue. In Proceedings of the Special Interest Group on Discourse and Dialogue (SIGDIAL). https://arxiv.org/abs/1705.05414. [pdf]
  + https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
  + <http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip>
  + calendar scheduling
  + weather information retrieval
  + point-of-interest navigation

### Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems

- Maluuba 放出的对话数据集。
- 论文链接：http://www.paperweekly.site/papers/407
- 数据集链接：http://datasets.maluuba.com/Frames

### Multi WOZ

+ https://www.repository.cam.ac.uk/handle/1810/280608

### Stanford Multi-turn Multi-domain

- 包含三个domain（日程，天气，景点信息），可参考下该数据机标注格式：
- <https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/>
- 论文citation
- Key-Value Retrieval Networks for Task-Oriented Dialogue <https://arxiv.org/abs/1705.05414>

## ### A Survey of Available Corpora for Building Data-Driven Dialogue Systems

+ 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
+ [链接](http://link.zhihu.com/?target=https%3A//docs.google.com/spreadsheets/d/1SJ4XV6NIEl_ReF1odYBRXs0q6mTkedoygY3kLMPjcP8/pubhtml)

## 英文数据集

- Cornell Movie Dialogs：电影对话数据集，下载地址：[http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](http://link.zhihu.com/?target=http%3A//www.cs.cornell.edu/%7Ecristian/Cornell_Movie-Dialogs_Corpus.html)
- Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：[https://arxiv.org/abs/1506.08909](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.08909)
- OpenSubtitles：电影字幕，下载地址：[http://opus.lingfil.uu.se/OpenSubtitles.php](http://link.zhihu.com/?target=http%3A//opus.lingfil.uu.se/OpenSubtitles.php)
- Twitter：twitter数据集，下载地址：[https://github.com/Marsan-Ma/twitter_scraper](http://link.zhihu.com/?target=https%3A//github.com/Marsan-Ma/twitter_scraper)
- Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：[https://github.com/bshao001/ChatLearner](http://link.zhihu.com/?target=https%3A//github.com/bshao001/ChatLearner)

相关数据集的处理代码或者处理好的数据可以参见下面两个github项目：

- [DeepQA](http://link.zhihu.com/?target=https%3A//github.com/Conchylicultor/DeepQA)
- [chat_corpus](http://link.zhihu.com/?target=https%3A//github.com/Marsan-Ma/chat_corpus)

## 中文数据集

- dgk_shooter_min.conv：中文电影台词数据集，下载链接：[https://github.com/rustch3n/dgk_lost_conv](http://link.zhihu.com/?target=https%3A//github.com/rustch3n/dgk_lost_conv)
- 白鹭时代中文问答语料：白鹭时代论坛问答数据，一个问题对应一个最好的答案。下载链接：[https://github.com/Samurais/egret-wenda-corpus](http://link.zhihu.com/?target=https%3A//github.com/Samurais/egret-wenda-corpus)
  - 微博数据集：华为李航实验室发布，也是论文“Neural Responding Machine for Short-Text Conversation”使用的数据集下载链接：[http://61.93.89.94/Noah_NRM_Data/](http://link.zhihu.com/?target=http%3A//61.93.89.94/Noah_NRM_Data/)
- 新浪微博数据集，评论回复短句，下载地址：[http://lwc.daanvanesch.nl/openaccess.php](