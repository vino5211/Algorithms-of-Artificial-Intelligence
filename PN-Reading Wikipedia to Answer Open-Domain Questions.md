+ #Reading Wikipedia to Answer Open-Domain Quesitons
+ ## Abstract
+ This paper proposes to tackle open-domain question answering using Wikipedia as the unique knowledge source: the answer to any factoid question is a text span in a Wikipedia article.
	+ 何真实性问题的答案是维基百科文章中的文本跨度)
+ This task of machine reading at scale combines the challenges of document re-trieval (finding the relevant articles) with that of machine comprehension of text (identifying the answer spans from those articles)
	+ 大规模机器阅读的任务将文件重新找到（找到相关文章）和机器理解文本（识别这些文章的答案）的挑战相结合
+ Our approach combines a search component based on bigram hashing and TF-IDF matching with a multi-layer recurrent neural network model trained to detect answers in Wikipedia paragraphs
+ Our experiments on multiple existing QA datasets indicate that:
	+ (1) both modules are highly competitive with respect to existing counterparts(现有的同行)
	+ (2) multitask learning using distant supervision（远距离监督） on their combination is an effective complete system on this challenging task.
+ ## Introduction
+ This paper considers the problem of answering factoid questions in an open-domain setting using Wikipedia as the unique knowledge source,such as one does when looking for answers in an encyclopedia(百科全书)
+ Wikipedia is a constantly evolving source（不断发展的信息源） of detailed information that could facilitate intelligent machines — if they are able to leverage（利用） its power
+ Unlike knowledge bases (KBs) such as Freebase (Bollacker et al., 2008) or DBPedia (Auer et al., 2007), which are easier for computers to process but too **sparsely populated（人口稀少）** for open-domain question answering
	+ Miller at al.2016
+ **Wikipedia contains up-to-date knowledge that humans are interested in**
	+ It is designed, however, for humans – not machines – to read
+ *Using Wikipedia articles as the knowledge source* causes *the task of question answering (QA)* to combine the challenges of both large-scale open-domain QA and of machine comprehension of text
+ In order to answer any question:
	+ one must first retrieve the few relevant articles among more than 5 million items
	+ and then scan them carefully to identify the answer
+ We term this setting,machine reading at scale**(MRS)**
+ Our work treats Wikipedia as a **collection of articles** and does not rely on **its internal graph structure**
+ **As a result, our approach is generic and could be switched to other collections of documents, books, or even daily updated newspapers**
+ Compare to other projects:
	+ Multi information source:
		+ Large-scale QA systems like **IBM’s DeepQA**(Ferrucci et al., 2010) rely on multiple sources to answer: besides Wikipedia, it is also **paired with KBs**, dictionaries, and even news articles,books, etc. As a result, such systems heavily rely on information redundancy among the sources（信息源冗余） to answer correctly.
	+ Single information source:
		+ Having a single knowledge source forces the model to be very precise while searching for an answer as the evidence might appear only once.
		+ This challenge thus encourages research in the ability of a machine to read,a key motivation for the machine comprehension subfield and the creation of datasets such as SQuAD (Rajpurkar et al., 2016), CNN/Daily Mail (Hermann et al., 2015) and CBT (Hill et al.,2016).
			+ SQuAD
			+ CNN/Daily Mail
			+ CBT
+ However, those machine comprehension resources typically assume that a short piece of relevant text is already identified and given to the model, which is not realistic for building an open domain QA system(对开放领域的QA系统是不现实的)
+ In sharp contrast:
	+ methods that use KBs or information retrieval over documents have to employ search as an integral part ofthe solution.
		+ 使用KB或通过文档进行信息检索的方法必须将搜索作为解决方案的一个组成部分
	+ Instead MRS is focused on simultaneously maintaining the challenge of **machine comprehension, which requires the deep understanding of text**, while keeping the realistic constraint(现实的约束) of searching over a large open resource.
+ Key algorithm fo this paper
	+ In this paper, we show how multiple existing QA datasets can be used to evaluate MRS by requiring an open-domain system to perform well on all of them at once.
		+ 评估MRS: 要求在所有数据集上均表现良好
	+  We develop DrQA, a strong system for question answering from Wikipedia composed of:
		+   (1) Document Retriever, a module using bigram hashing and TF-IDF matching designed to, given a question, efficiently return a subset of relevant articles and
		+   (2) Document Reader, a multi-layer recurrent neural network machine comprehension model trained to detect answer spans in those few returned documents.
		```
        （1）文档检索器，使用二元散列和TF-IDF匹配的模块，用于给出问题，有效地返回相关文章的子集，
  		（2）文献阅读器是一种多层递归神经网络机器理解模型，用于检测返回的文档中的回答跨度。
        ```
+ ###Figure 1 gives an illustration of DrQA.
![An overview of our question answering system DrQA.png](/home/apollo/Pictures/An overview of our question answering system DrQA.png)
