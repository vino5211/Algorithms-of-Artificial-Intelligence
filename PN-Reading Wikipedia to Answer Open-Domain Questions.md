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

+ experiment
	+ Our experiments show that Document Retriever outperforms the built-in Wikipedia search engine and that Document Reader reaches state-of-the-art results on the very competitive SQuAD bench-mark (Rajpurkar et al., 2016). 
	+ Finally, our full system is evaluated using multiple benchmarks. In particular, we show that performance is improved across all datasets through the use of **multitask learning** and **distant supervision** compared to single task training.


+ ## Related Work
+ Open-domain QA Define
	+ Open-domain QA was originally defined as finding answers in collections of unstructured documents, following the setting of the annual TREC competitions
		+ TREC competitions
		+ unstructured documents
+ KB
	+ With the development of KBs, many recent innovations have occurred in the context of QA from KBs with the creation of resources like **WebQuestions** (Berant et al., 2013) and **SimpleQuestions** (Bordes et al., 2015) based on the **Freebase KB** (Bollacker et al., 2008), or on automatically extracted KBs, e.g., **OpenIE triples and NELL (Fader et al., 2014)**.
	+ inherent limitations
		+ incompleteness
		+ fixed schemas
+ motivated researchers to return to original setting of **answering from raw text**
	+ KB inherent limitations
	+ deep learning architectures like attention-based and memory augmented neural networks
		+ Bahdanau et al.,2015; Weston et al., 2015; Graves et al., 2014
	+ release of new training and evaluation datasets like QuizBowl (Iyyer et al., 2014), CNN/Daily Mail based on news articles (Hermann et al., 2015), CBT based on children books (Hill et al., 2016), or SQuAD (Rajpurkar et al., 2016) and WikiReading (Hewlett et al., 2016), both based on Wikipedia.
+ Target
	+ An objective of this paper is to test how such new methods can perform in an open-domain QA framework.
+ QA using Wikipedia as a resource has been explored previously
	+ Ryu et al. (2014) perform open-domain QA using a Wikipedia-based knowledge model. They **combine** article content with multiple other answer matching modules based on different types of semi-structured knowledge such as infoboxes, article structure, category structure,and definitions
	+ Ahn et al. (2004) also **combine** Wikipedia as a text resource with other resources, in this case with information retrieval over other documents
	+ Buscaldi and Rosso (2006) also mine knowledge from Wikipedia for QA. Instead of using it as a resource for seeking answers to questions, they focus on validating answers returned by their QA system, and use Wikipedia categories for determining a set of patterns that should fit with the expected answer(找到答案的模式)
+ This work
	+ In our work,we consider the comprehension of text only, and use Wikipedia text documents as the sole resource in order to emphasize the task of machine reading at scale, as described in the introduction.(强调机器阅读)
	+ Comparing against these methods provides a useful datapoint for an “upper bound” benchmark on performance.
		+ 与这些方法进行比较为性能上限基准提供了一个有用的数据点。
+ Highly developed full pipeline QA
	+ AskMSR:Microsoft
	+ DeepQA:IBM
	+ YodaQA
		+ open source
		+ combining websites,information extraction, databases and Wikipedia in particular
+ Multitask learning (Caruana, 1998) and task transfer
	+ **Multitask learning (Caruana, 1998)** and **task transfer** have a rich history in machine learning(e.g., using ImageNet in the computer vision community (Huh et al., 2016)), as well as in NLP in particular (Collobert and Weston, 2008).
	+ Several works have attempted to combine multiple QA training datasets via multitask learning to 
		+ (i)achieve improvement across the datasets via tasktransfer;
		+ (ii) provide a single general system capable of asking different kinds of questions due to the inevitably different data distributions across the source datasets.
	+ Fader et al. (2014) used WebQuestions, TREC and WikiAnswers with four KBs as knowledge sources and reported improvement on the latter two datasets through multitask learning.
	+ Bordes et al. (2015) combined WebQuestions and SimpleQuestions using distant supervision with Freebase as the KB to give slight improvements on both datasets, although poor performance was reported when training on only one dataset and testing on the other, showing that task transfer is indeed a challenging subject
	+ see also (Kadlec et al., 2016) for a similar conclusion.
	+ Our work follows similar themes, but in the setting of having to retrieve and then read text documents(先检索后阅读)，rather than using a KB, with positive results(取得了很好的结果).



+ ## Our System:DrQA
+ In the following we describe our system DrQA for MRS which consists of two components:
	+ (1) the Document Retriever module for finding relevant articles
	+ (2) a machine comprehension model, Document Reader, for extracting answers from a single document or a small collection of documents.
+ ### Document Retriever
+ ### Document Reader
	+ Insired by:
    ```
	Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil 	Blunsom. 2015. Teaching machines to read and comprehend. In Advances in Neural Information Processing Systems (NIPS)
    ```
    + Given a question q consisting of l tokens{q 1 , . . . , q l } and a document or a small set of documents of n paragraphs where a single paragraph p consists of m tokens {p 1 , . . . , p m }, we develop an RNN model that we apply to each paragraph in turn and then finally aggregate the predicted answers. Our method works as follows:
		+ Parapragh encoding
			+ put all tokens $ p_{i} $ in a paragraph $p$ as a sequence of feature vectors $\tilde{p} \in R^{d}$
			+ put feature vectors as the input to a recurrent neural network and thus obtain:
				$$
                	{p_{1},...,p_{n}} = RNN({\tilde{p_{1}},...,\tilde{p_{n}}})
                $$
            + $\tilde{p_{i}}$ is comprised of the following parts：
            	+ word embeddings
            	+ exact match
            	+ token features
            		+ POS
            		+ NER
            		+ TF
            	+ Aligned question embedding (对齐的问题) -- Attention
            		+ ```
            		Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. 2016. The Goldilocks Principle: Reading children’s books with explicit memory representations. In International Conference on Learning Representations (ICLR).
   					```
                    + $f_{align}(p_i) = \sum_j a_{i,j}E(q_j)$
                    + $a_{i,j} = \frac{exp(\alpha (E(p_i)) * \alpha(E(q_j)))}{\sum_{j^`} exp(\alpha(E(p_i)) * \alpha(E(q_{j^`})))}$
                    + $\alpha()$ is single dense layer with ReLU nonlinearity
                    + compared to the exact match features, these features add soft alignments between similar but non(与完全匹配功能相比，这些功能可以在相似但不相同的单词之间添加软对齐)
		+ Question encoding
		+ Prediction

+ ## Data
+ ## Experiments
+ ## Conslusion