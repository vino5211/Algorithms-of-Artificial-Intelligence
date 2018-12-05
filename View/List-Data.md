# Data-list
### Reference
+ 25个深度学习开源数据集，good luck !(SOTA)
+ https://machinelearningmastery.com/datasets-natural-language-processing/
+ 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
+ https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality

### NLP-Basic
+ IMDB Reviews
	+ 这是一个电影爱好者的梦寐以求的数据集。它意味着二元情感分类，并具有比此领域以前的任何数据集更多的数据。除了训练和测试评估示例之外，还有更多未标记的数据可供使用。包括文本和预处理的词袋格式。
	+ 大小：80 MB
	+ 记录数量：25,000个高度差异化的电影评论用于训练，25,000个测试
	+ SOTA：Learning Structured Text Representations
+ Twenty Newsgroups
	+ 顾名思义，该数据集包含有关新闻组的信息。为了选择这个数据集，从20个不同的新闻组中挑选了1000篇新闻文章。这些文章具有一定特征，如主题行，签名和引用。
	+ 大小：20 MB
	+ 记录数量：来自20个新闻组的20,000条消息
	+ DOTA:Very Deep Convolutional Networks for Text Classification
+ Sentiment140
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
+ WordNet
	+ 在上面的ImageNet数据集中提到，WordNet是一个很大的英文同义词集。 同义词集是每个都描述了不同的概念的同义词组。WordNet的结构使其成为NLP非常有用的工具。
	+ 大小：10 MB
	+ 记录数量：117,000个同义词集通过少量“概念关系”与其他同义词集相关联。
	+ SOTA:Wordnets: State of the Art and Perspectives
+ Yelp Reviews
	+ 这是Yelp为了学习目的而发布的一个开源数据集。它包含了由数百万用户评论，商业属性和来自多个大都市地区的超过20万张照片。这是一个非常常用的全球NLP挑战数据集。
	+ 大小：2.66 GB JSON，2.9 GB SQL和7.5 GB照片（全部压缩）
	+ 记录数量：5,200,000条评论，174,000条商业属性，20万张图片和11个大都市区
	+ SOTA：Attentive Convolution
+ The Wikipedia Corpus
	+ 这个数据集是维基百科全文的集合。它包含来自400多万篇文章的将近19亿字。使得这个成为强大的NLP数据集的是你可以通过单词，短语或段落本身的一部分进行搜索。
	+ 这个数据集是维基百科全文的集合。它包含来自400多万篇文章的将近19亿字。使得这个成为强大的NLP数据集的是你可以通过单词，短语或段落本身的一部分进行搜索。
	+ 大小：20 MB
	+ 记录数量：4,400,000篇文章，19亿字
	+ SOTA:Breaking The Softmax Bottelneck: A High-Rank RNN language Model
+ The Blog Authorship Corpus
	+ 这个数据集包含了从blogger.com收集的数千名博主的博客帖子。每个博客都作为一个单独的文件提供。每个博客至少包含200个常用英语单词。
	+ 大小：300 MB
	+ 记录数量：681,288个帖子，超过1.4亿字
	+ SOTA:Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution

### Tasks
### 1. 文本分类（Text Classification）
文本分类指的是标记句子或者文档，比如说垃圾邮件分类和情感分析。
以下是一些对于新手而言非常棒的文本分类数据集：
+ kaggel:Kaggle Twitter Sentiment Analysis
+ Reuters Newswire Topic Classification(Reuters-21578)
    + http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
+ 一系列1987年在路透上发布的按分类索引的文档。同样可以看RCV1，RCV2，以及TRC2
    + http://trec.nist.gov/data/reuters/reuters.html
+ IMDB Movie Review Sentiment Classification (Stanford)
    + http://ai.stanford.edu/~amaas/data/sentiment/c
+ 一系列从网站imdb.com上摘取的电影评论以及他们的积极或消极的情感。
    + News Group Movie Review Sentiment Classification (cornell)
        + http://www.cs.cornell.edu/people/pabo/movie-review-data/
+ 更多的信息，可以从这篇博文中获取：
    + Datasets for single-label text categorization
        + http://ana.cachopo.org/datasets-for-single-label-text-categorization

### 2. 语言模型（Language Modeling）
语言模型涉及建设一个统计模型来根据给定的信息，预测一个句子中的下一个单词，或者一个单词中的下一个字母。这是语音识别或者机器翻译等任务的前置任务。
+ （适合新手）Project Gutenberg
    + https://www.gutenberg.org/
+ （正式）Brown University Standard Corpus of Present-Day American English
    + https://en.wikipedia.org/wiki/Brown_Corpus
+ 大型英语单词示例：Google 1 Billion Word Corpus
    + https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark
+ Microsoft Research entence Completion Challenge dataset
	+ SOTA  : A fast and simple algorithm for training neural probabilistic language models

### 3. 图像字幕（Image Captioning）
图像字幕是为给定图像生成文字描述的任务。
+ （新手）Common Objects in Context (COCO) 
    + http://mscoco.org/dataset/#overview
+ 超过120，000张带描述的图片集合，Flickr 8K
    + 从flickr.com收集的超过8000带描述的图片集合
    + http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
+ Flickr 30K
    + http://shannon.cs.illinois.edu/DenotationGraph/
    + 从flickr.com收集的超过30000带描述的图片集合。
+ 要获得更多的资讯，可以看这篇博客：Exploring Image Captioning Datasets, 2016
    + http://sidgan.me/technical/2016/01/09/Exploring-Datasets

### 4. 机器翻译（Machine Translation）
机器翻译即将一种语言翻译成另一种语言的任务。
+ （新手）Aligned Hansards of the 36th Parliament of Canada
    + https://www.isi.edu/natural-language/download/hansard/
+ 英法对应的句子
    + European Parliament Proceedings Parallel Corpus 1996-2011
        + http://www.statmt.org/europarl/
        + 一系列欧洲语言的成对句子
+ 被用于机器翻译的标准数据集还有很多：
    + Statistical Machine Translation
        + http://www.statmt.org/
+ Machine Translation of Various Languages
	+ 此数据集包含四种欧洲语言的训练数据。这里的任务是改进当前的翻译方法。您可以参加以下任何语言组合：
		+ 英语-汉语和汉语-英语
		+ 英语-捷克语和捷克语-英语
		+ 英语-爱沙尼亚语和爱沙尼亚语-英语
		+ 英语-芬兰语和芬兰语-英语
		+ 英语-德语和德语-英语
		+ 英语-哈萨克语和哈萨克语-英语
		+ 英文-俄文和俄文-英文
		+ 英语-土耳其语和土耳其语-英语

### 5. 问答系统（Question Answering）
+ MCTest
+ Algebra
+ Science
+ Stanford Question Answering Dataset (SQuAD)
    + official website
    	+ https://rajpurkar.github.io/SQuAD-explorer/
    + reference
    	+ EMNLP2016 SQuAD:100,000+ Questions for Machine Comprehension of Text
			+ https://arxiv.org/pdf/1606.05250.pdf
		+ SQuAD，斯坦福在自然语言处理的野心
			+ http://blog.csdn.net/jdbc/article/details/52514050
    + SOTAs
    	+ Hybrid AoA Reader (ensemble)
        	+ Joint Laboratory of HIT and iFLYTEK Research
    	+ r-net + 融合模型
        	+ Microsoft Research Asia
    	+ SLQA + 融合模型
        	+ Alibaba iDST NLP

	+ detail
		+ 这个竞赛基于SQuAD问答数据集，考察两个指标：EM和F1。
		+ EM是指精确匹配，也就是模型给出的答案与标准答案一模一样；
		+ F1，是根据模型给出的答案和标准答案之间的重合度计算出来的，也就是结合了召回率和精确率。
		+ 目前阿里、微软团队并列第一，其中EM得分微软（r-net+融合模型）更高，F1得分阿里（SLQA+融合模型）更高。但是他们在EM成绩上都击败了“人类表现”
		+ 一共有107,785问题，以及配套的 536 篇文章
			+ 数据集的具体构建如下：
                1. 文章是随机sample的wiki百科，一共有536篇wiki被选中。而每篇wiki，会被切成段落，最终生成了23215个自然段。之后就对这23215个自然段进行阅读理解，或者说自动问答。
                2. 之后斯坦福，利用众包的方式，进行了给定文章，提问题并给答案的人工标注。他们将这两万多个段落给不同人，要求对每个段落提五个问题。
                3. 让另一些人对提的这个问题用文中最短的片段给予答案，如果不会或者答案没有在文章中出现可以不给。之后经过他们的验证，人们所提的问题在问题类型分布上足够多样，并且有很多需要推理的问题，也就意味着这个集合十分有难度。如下图所示，作者列出了该数据集答案的类别分布，我们可以看到 日期，人名，地点，数字等都被囊括，且比例相当。
                4. 这个数据集的评测标准有两个：
                第一：F1
                第二：EM。
                    EM是完全匹配的缩写，必须机器给出的和人给出的一样才算正确。哪怕有一个字母不一样，也会算错。而F1是将答案的短语切成词，和人的答案一起算recall，Precision和F1，即如果你match了一些词但不全对，仍然算分。
                5. 为了这个数据集，他们还做了一个baseline，是通过提特征，用LR算法将特征组合，最终达到了40.4的em和51的f1。而现在IBM和新加坡管理大学利用深度学习模型，均突破了这个算法。可以想见，在不远的将来会有更多人对阅读理解发起挑战，自然语言的英雄也必将诞生。甚至会有算法超过人的准确度。
+ MS MARCO
    + 相比SQuAD，MARCO的挑战难度更大，因为它需要测试者提交的模型具备理解复杂文档、回答复杂问题的能力。
    + 据了解，对于每一个问题，MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到。这对机器阅读理解提出了更高的要求，需要机器具备综合理解多文档信息、聚合生成问题答案的能力。
+ NarrativeQA
    + Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
        + https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
        + https://github.com/deepmind/narrativeqa
        + DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。
	+ The NarrativeQA Reading Comprehension Challenge
        + 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
        + 论文链接：https://www.paperweekly.site/papers/1397
        + 代码链接：https://github.com/deepmind/narrativeqa
+ SougoQA
    + http://task.www.sogou.com/cips-sogou_qa/
+ GraphQuestions 
	+ On Generating Characteristic-rich Question Sets for QA Evaluation
	+ 文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。
    + 数据集特点：
        1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
        2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
        3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
        4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
        5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。
    + 论文链接：http://www.paperweekly.site/papers/906
    + 数据集链接：https://github.com/ysu1989/GraphQuestions
+ LSDSem 2017 Shared Task: The Story Cloze Test
    + Story Cloze Test：人工合成的完形填空数据集。
    + 论文链接：http://www.paperweekly.site/papers/917
    + 数据集链接：http://cs.rochester.edu/nlp/rocstories/
+ Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering
    + 百度深度学习实验室创建的中文开放域事实型问答数据集。
    + 论文链接：http://www.paperweekly.site/papers/914
    + 数据集链接：http://idl.baidu.com/WebQA.html
+ Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems
    + DeepMind 和牛津大学共同打造的代数问题数据集 AQuA（Algebra Question Answering）。
    + 论文链接：http://www.paperweekly.site/papers/913
    + 数据集链接：https://github.com/deepmind/AQuA
+ Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems
    + Maluuba 放出的对话数据集。
    + 论文链接：http://www.paperweekly.site/papers/407
    + 数据集链接：http://datasets.maluuba.com/Frames
+ Teaching Machines to Read and Comprehend
    + DeepMind Q&A Dataset 是一个经典的机器阅读理解数据集，分为两个部分：
        1. CNN：~90k 美国有线电视新闻网（CNN）的新闻文章，~380k 问题；
        2. Daily Mail：~197k DailyMail 新闻网的新闻文章（不是邮件正文），~879k 问题。
    + 论文链接：http://www.paperweekly.site/papers/915
    + 数据集链接：http://cs.nyu.edu/~kcho/DMQA/
+ Semantic Parsing on Freebase from Question-Answer Pairs
    + 文章发表在 EMNLP-13，The Stanford NLP Group 是世界领先的 NLP 团队。
    + 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的
    + 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣）
    + 有 6642 个问答对
    + 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system。
    + 论文链接：http://www.paperweekly.site/papers/827
    + 数据集链接：http://t.cn/RWPdQQO
+ A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories
    + ROCStories dataset for story cloze test.
    + 论文链接：http://www.paperweekly.site/papers/918
    + 数据集链接：http://cs.rochester.edu/nlp/rocstories/
+ MoleculeNet:  Benchmark for Molecular Machine Learning
    + 一个分子机器学习 benchmark，最喜欢看到这种将机器学习应用到传统学科领域了。
    + 论文链接：http://www.paperweekly.site/papers/862
    + 数据集链接：http://t.cn/RWPda8r
+ 关于维基百科文章的问答:Deepmind Question Answering Corpus
    + https://github.com/deepmind/rc-data
+ 关于亚马逊产品的问答
    + Amazon question/answer data
        + http://jmcauley.ucsd.edu/data/amazon/qa/
+ 更多信息，参见：
    + Datasets: How can I get corpus of a question-answering website like Quora or Yahoo Answers or Stack Overflow for analyzing answer quality?
        https://www.quora.com/Datasets-How-can-I-get-corpus-of-a-question-answering-website-like-Quora-or-Yahoo-Answers-or-Stack-Overflow-for-analyzing-answer-quality

+ Looking Beyond the surface: A Chanllenge Set for Reading Comprehension over Multiple Sentences
	+ 特点:
		+ 多选题
		+ 问题的答案来自篇章中的多条语句
		+ 数据集来自7个不同的领域
	+ 基准算法:
		+ Random
		+ IR
		+ SurfaceIR
		+ SemanticLP
		+ BiDAF
	+ SOTA
		+ SurfaceIR 结构的F1 值 相较人类结果 差 20个百分点


### 6. 语音识别（Speech Recognition）
+ 语音识别就是将口语语言的录音转换成人类可读的文本。
+ 新手:TIMIT Acoustic-Phonetic Continuous Speech Corpus
    + https://catalog.ldc.upenn.edu/LDC93S1
+ 付费，这里列出是因为它被广泛使用。美语口语以及相关转写
    + VoxForge
        + http://voxforge.org/
+ 为语音识别而建设开源数据库的项目
    + LibriSpeech ASR corpus
        + http://www.openslr.org/12/
+ 从LibriVox获取的英语有声书大型集合
    + https://librivox.org/

### 7. 自动文摘（Document Summarization）
自动文摘即产生对大型文档的一个短小而有意义的描述。
+ 新手：Legal Case Reports Data Set
    + https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports
+ 4000法律案例以及摘要的集合 TIPSTER Text Summarization Evaluation Conference Corpus
    + http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html
+ 将近200个文档以及摘要的集合
    + The AQUAINT Corpus of English News Text
        + https://catalog.ldc.upenn.edu/LDC2002T31

### 8. 序列标注 (Sequence Labeling)
用于序列标注任务
+ 国内可用免费语料库
    + http://www.cnblogs.com/mo-wang/p/4444858.html
+ 一个中文的标注语料库。可用于训练HMM模型。
    + https://github.com/liwenzhu/corpusZh
+ 可以提供给题主两份相对较新的中文分词语料
    + 第一份是SIGHAN的汉语处理评测的Bakeoff语料，从03年起首次进行评测，评测的内容针对汉语分词的准确性和合理性，形成Bakeoff 2005评测集，包含简、繁体中文的训练集和测试集，训练集有四个，单句量在1.5W~8W+。内容比较偏向于书面语。后面05 07年分别对中文命名实体识别和词性标注给出了评测。Bakeoff 2005中文分词熟语料传送门：Second International Chinese Word Segmentation Bakeoff
    + 第二份语料来自GitHub作者liwenzhu，于14年发布于GitHub，总词汇量在7400W+，可以用于训练很多模型例如Max Entropy、CRF、HMM......，优点是这份语料在分词基础上还做了词性标注，至于准确性还有待考究。传送门：liwenzhu/corpusZh
+ zh seg
    + http://sighan.cs.uchicago.edu/bakeoff2005/
        + icwb2
            + http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.rar

### Ranking
+ [LETOR, ](http://blog.crackcell.com/posts/2011/12/17/a_short_intro_2_ltr.html#sec-7)<http://research.microsoft.com/en-us/um/beijing/projects/letor/>
+ Microsoft Learning to Rank Dataset, <http://research.microsoft.com/en-us/projects/mslr/>
+ Yahoo Learning to Rank Challenge, <http://webscope.sandbox.yahoo.com/>

### others
+ 中文文本语料库整理（不定时更新2015-10-24）

### new
*先来个不能错过的数据集网站（深度学习者的福音）：* 
http://deeplearning.net/datasets/**

首先说说几个收集数据集的网站： 
1、Public Data Sets on Amazon Web Services (AWS) 
http://aws.amazon.com/datasets 
Amazon从2008年开始就为开发者提供几十TB的开发数据。

2、Yahoo! Webscope 
http://webscope.sandbox.yahoo.com/index.php

3、Konect is a collection of network datasets 
http://konect.uni-koblenz.de/

4、Stanford Large Network Dataset Collection 
http://snap.stanford.edu/data/index.html

再就是说说几个跟互联网有关的数据集： 
1、Dataset for “Statistics and Social Network of YouTube Videos” 
http://netsg.cs.sfu.ca/youtubedata/

2、1998 World Cup Web Site Access Logs 
http://ita.ee.lbl.gov/html/contrib/WorldCup.html 
这个是1998年世界杯期间的数据集。从1998/04/26 到 1998/07/26 的92天中，发生了 1,352,804,107次请求。

3、Page view statistics for Wikimedia projects 
http://dammit.lt/wikistats/

4、AOL Search Query Logs - RP 
http://www.researchpipeline.com/mediawiki/index.php?title=AOL_Search_Query_Logs

5、livedoor gourmet 
http://blog.livedoor.jp/techblog/archives/65836960.html

海量图像数据集： 
1、ImageNet 
http://www.image-net.org/ 
包含1400万的图像。

2、Tiny Images Dataset 
http://horatio.cs.nyu.edu/mit/tiny/data/index.html 
包含8000万的32x32图像。

3、 MirFlickr1M 
http://press.liacs.nl/mirflickr/ 
Flickr中的100万的图像集。

4、 CoPhIR 
http://cophir.isti.cnr.it/whatis.html 
Flickr中的1亿600万的图像

5、SBU captioned photo dataset 
http://dsl1.cewit.stonybrook.edu/~vicente/sbucaptions/ 
Flickr中的100万的图像集。

6、Large-Scale Image Annotation using Visual Synset(ICCV 2011) 
http://cpl.cc.gatech.edu/projects/VisualSynset/ 
包含2亿图像

7、NUS-WIDE 
http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm 
Flickr中的27万的图像集。

8、SUN dataset 
http://people.csail.mit.edu/jxiao/SUN/ 
包含13万的图像

9、MSRA-MM 
http://research.microsoft.com/en-us/projects/msrammdata/ 
包含100万的图像，23000视频

10、TRECVID 
http://trecvid.nist.gov/

截止目前好像还没有国内的企业或者组织开放自己的数据集。希望也能有企业开发自己的数据集给研究人员使用，从而推动海量数据处理在国内的发展！

2014/07/07 雅虎发布超大Flickr数据集 1亿的图片+视频 
http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for

100多个有趣的数据集 
http://www.csdn.net/article/2014-06-06/2820111-100-Interesting-Data-Sets-for-Statistics

