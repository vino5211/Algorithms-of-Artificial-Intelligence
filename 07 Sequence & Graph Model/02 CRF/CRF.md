[TOC]

# Markov networks - CRF

### Reference

+ 条件随机场最早由John D. Lafferty提出，其也是[Brown90](http://www.52nlp.cn/strong-author-team-of-smt-classic-brown90)的作者之一，和贾里尼克相似，在离开IBM后他去了卡耐基梅隆大学继续搞学术研究，2001年以第一作者的身份发表了CRF的经典论文 “Conditional random fields: Probabilistic models for segmenting and labeling sequence data”
+ 关于条件随机场的参考文献及其他资料，Hanna Wallach在05年整理和维护的这个页面“[conditional random fields](http://www.inference.phy.cam.ac.uk/hmw26/crf/)”非常不错，其中涵盖了自01年CRF提出以来的很多经典论文（不过似乎只到05年，之后并未更新）以及几个相关的工具包(不过也没有包括CRF++）
+ https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers
+ https://www.bilibili.com/video/av10590361/?p=35
+ https://www.cnblogs.com/pinard/p/7048333.html



### Tutorial

Hanna M. Wallach. [Conditional Random Fields: An Introduction.](http://www.inference.phy.cam.ac.uk/hmw26/papers/crf_intro.pdf) Technical Report MS-CIS-04-21. Department of Computer and Information Science, University of Pennsylvania, 2004.

### Papers by year

#### 2001

John Lafferty, Andrew McCallum, Fernando Pereira. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.](http://www.cs.umass.edu/~mccallum/papers/crf-icml01.ps.gz) In *Proceedings of the Eighteenth International Conference on Machine Learning* (ICML-2001), 2001.

#### 2002

Hanna Wallach. [Efficient Training of Conditional Random Fields.](http://www.cogsci.ed.ac.uk/~osborne/msc-projects/wallach.ps.gz) M.Sc. thesis, Division of Informatics, University of Edinburgh, 2002.

Thomas G. Dietterich. [Machine Learning for Sequential Data: A Review.](http://eecs.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf) In *Structural, Syntactic, and Statistical Pattern Recognition; Lecture Notes in Computer Science, Vol. 2396*, T. Caelli (Ed.), pp. 15–30, Springer-Verlag, 2002.

#### 2003

Fei Sha and Fernando Pereira. [Shallow Parsing with Conditional Random Fields.](http://www.cis.upenn.edu/~feisha/pubs/shallow03.pdf) In *Proceedings of the 2003 Human Language Technology Conference and North American Chapter of the Association for Computational Linguistics* (HLT/NAACL-03), 2003.

Andrew McCallum. [Efficiently Inducing Features of Conditional Random Fields.](http://www.cs.umass.edu/~mccallum/papers/ifcrf-uai2003.pdf) In *Proceedings of the 19th Conference in Uncertainty in Articifical Intelligence* (UAI-2003), 2003.

David Pinto, Andrew McCallum, Xing Wei and W. Bruce Croft. [Table Extraction Using Conditional Random Fields.](http://www.cs.umass.edu/~mccallum/papers/crftable-sigir2003.pdf) In *Proceedings of the 26th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval* (SIGIR 2003), 2003.

Andrew McCallum and Wei Li. [Early Results for Named Entity Recognition with Conditional Random Fields, Feature Induction and Web-Enhanced Lexicons.](http://cnts.uia.ac.be/conll2003/pdf/18891mcc.pdf) In *Proceedings of the Seventh Conference on Natural Language Learning* (CoNLL), 2003.

Wei Li and Andrew McCallum. [Rapid Development of Hindi Named Entity Recognition Using Conditional Random Fields and Feature Induction.](http://www.cs.umass.edu/~mccallum/papers/hindi-talip2003.pdf) In *ACM Transactions on Asian Language Information Processing* (TALIP), 2003.

Yasemin Altun and Thomas Hofmann. [Large Margin Methods for Label Sequence Learning.](http://www.cs.brown.edu/people/altun/pubs/AltunHofmann-EuroSpeech2003.pdf) In *Proceedings of 8th European Conference on Speech Communication and Technology*(EuroSpeech), 2003.

Simon Lacoste-Julien. [Combining SVM with graphical models for supervised classification: an introduction to Max-Margin Markov Networks](http://www.cs.berkeley.edu/~slacoste/school/cs281a/project/M3netReportpdf.pdf). CS281A Project Report, UC Berkeley, 2003.

#### 2004

Andrew McCallum, Khashayar Rohanimanesh and Charles Sutton. [Dynamic Conditional Random Fields for Jointly Labeling Multiple Sequences.](http://www.cs.umass.edu/~mccallum/papers/dcrf-nips03.pdf) Workshop on Syntax, Semantics, Statistics; 16th Annual Conference on Neural Information Processing Systems (NIPS 2003), 2004.

Kevin Murphy, Antonio Torralba and William T.F. Freeman. [Using the forest to see the trees: a graphical model relating features, objects and scenes.](http://web.mit.edu/torralba/www/nips2003.pdf) In *Advances in Neural Information Processing Systems 16* (NIPS 2003), 2004.

Sanjiv Kumar and Martial Hebert. [Discriminative Fields for Modeling Spatial Dependencies in Natural Images.](http://www-2.cs.cmu.edu/~skumar/DRF/modDRF.pdf) In *Advances in Neural Information Processing Systems 16* (NIPS 2003), 2004.

Ben Taskar, Carlos Guestrin and Daphne Koller. [Max-Margin Markov Networks.](http://books.nips.cc/papers/files/nips16/NIPS2003_AA04.pdf) In *Advances in Neural Information Processing Systems 16* (NIPS 2003), 2004.

Burr Settles. [Biomedical Named Entity Recognition Using Conditional Random Fields and Rich Feature Sets.](http://www.cs.cmu.edu/~bsettles/pub/settles.nlpba04.pdf) To appear in *Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications* (NLPBA), 2004.

A demo of the system can be downloaded [here](http://www.cs.wisc.edu/~bsettles/abner/).

Charles Sutton, Khashayar Rohanimanesh and Andrew McCallum. [Dynamic Conditional Random Fields: Factorized Probabilistic Models for Labeling and Segmenting Sequence Data.](http://www.aicml.cs.ualberta.ca/banff04/icml/pages/papers/308.pdf) In *Proceedings of the Twenty-First International Conference on Machine Learning* (ICML 2004), 2004.

John Lafferty, Xiaojin Zhu and Yan Liu. [Kernel conditional random fields: representation and clique selection.](http://portal.acm.org/citation.cfm?id=1015330.1015337) In *Proceedings of the Twenty-First International Conference on Machine Learning* (ICML 2004), 2004.

Xuming He, Richard Zemel, and Miguel Á. Carreira-Perpiñán. [Multiscale conditional random fields for image labelling.](http://www.cs.toronto.edu/pub/zemel/Papers/cvpr04.pdf) In *Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition* (CVPR 2004), 2004.

Yasemin Altun, Alex J. Smola, Thomas Hofmann. [Exponential Families for Conditional Random Fields.](http://www.cs.brown.edu/~th/papers/AltSmoHof-UAI2004.pdf) In *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence* (UAI-2004), 2004.

Michelle L. Gregory and Yasemin Altun. [Using Conditional Random Fields to Predict Pitch Accents in Conversational Speech.](http://www.cs.brown.edu/people/altun/pubs/GregoryAltun.pdf) In *Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics* (ACL 2004), 2004.

Brian Roark, Murat Saraclar, Michael Collins and Mark Johnson. [Discriminative Language Modeling with Conditional Random Fields and the Perceptron Algorithm.](http://www.cslu.ogi.edu/people/roark/ACL04CRFLM.pdf) In *Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics* (ACL 2004), 2004.

Ryan McDonald and Fernando Pereira. [Identifying Gene and Protein Mentions in Text Using Conditional Random Fields.](http://www.pdg.cnb.uam.es/BioLINK/workshop_BioCreative_04/handout/pdf/task1A.pdf) BioCreative, 2004.

Trausti T. Kristjansson, Aron Culotta, Paul Viola and Andrew McCallum. [Interactive Information Extraction with Constrained Conditional Random Fields.](http://http//www.cs.umass.edu/~mccallum/papers/addrie-aaai04.pdf) In *Proceedings of the Nineteenth National Conference on Artificial Intelligence* (AAAI 2004), 2004.

Thomas G. Dietterich, Adam Ashenfelter and Yaroslav Bulatov. [Training Conditional Random Fields via Gradient Tree Boosting.](http://web.engr.oregonstate.edu/~tgd/publications/ml2004-treecrf.pdf) In *Proceedings of the Twenty-First International Conference on Machine Learning* (ICML 2004), 2004.

John Lafferty, Yan Liu and Xiaojin Zhu. [Kernel Conditional Random Fields: Representation, Clique Selection, and Semi-Supervised Learning.](http://www.aladdin.cs.cmu.edu/papers/pdfs/y2004/kernecon.pdf) Technical Report CMU-CS-04-115, Carnegie Mellon University, 2004.

Fuchun Peng and Andrew McCallum (2004). [Accurate Information Extraction from Research Papers using Conditional Random Fields.](http://acl.ldc.upenn.edu/hlt-naacl2004/main/pdf/176_Paper.pdf) In *Proceedings of Human Language Technology Conference and North American Chapter of the Association for Computational Linguistics* (HLT/NAACL-04), 2004.

Yasemin Altun, Thomas Hofmann and Alexander J. Smola. [Gaussian process classification for segmenting and annotating sequences.](http://www.cs.brown.edu/~th/papers/AltHofSmo-ICML2004.pdf) In *Proceedings of the Twenty-First International Conference on Machine Learning* (ICML 2004), 2004.

Yasemin Altun and Thomas Hofmann. [Gaussian Process Classification for Segmenting and Annotating Sequences.](http://www.cs.brown.edu/people/altun/pubs/CS-03-23.ps) Technical Report CS-04-12, Department of Computer Science, Brown University, 2004.

#### 2005

Cristian Smimchisescu, Atul Kanaujia, Zhiguo Li and Dimitris Metaxus. [Conditional Models for Contextual Human Motion Recognition.](http://www.cs.toronto.edu/~crismin/PAPERS/iccv05.pdf) In *Proceedings of the International Conference on Computer Vision*, (ICCV 2005), Beijing, China, 2005.

Ariadna Quattoni, Michael Collins and Trevor Darrel. [Conditional Random Fields for Object Recognition.](http://books.nips.cc/papers/files/nips17/NIPS2004_0810.pdf) In *Advances in Neural Information Processing Systems 17* (NIPS 2004), 2005.

Jospeh Bockhorst and Mark Craven. [Markov Networks for Detecting Overlapping Elements in Sequence Data.](http://books.nips.cc/papers/files/nips17/2004_0745.pdf) In *Advances in Neural Information Processing Systems 17* (NIPS 2004), 2005.

Antonio Torralba, Kevin P. Murphy, William T. Freeman. [Contextual models for object detection using boosted random fields.](http://www.ai.mit.edu/~murphyk/Papers/BRFaimemo.pdf) In *Advances in Neural Information Processing Systems 17* (NIPS 2004), 2005.

Sunita Sarawagi and William W. Cohen. [Semi-Markov Conditional Random Fields for Information Extraction.](http://www-2.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf) In *Advances in Neural Information Processing Systems 17* (NIPS 2004), 2005.

Yuan Qi, Martin Szummer and Thomas P. Minka. [Bayesian Conditional Random Fields.](http://people.csail.mit.edu/u/a/alanqi/public_html/papers/Qi-Bayesian-CRF-AIstat05.pdf) To appear in Proceedings of the Tenth International W\orkshop on Artificial Intelligence and Statistics (AISTATS 2005), 2005.

Aron Culotta, David Kulp and Andrew McCallum. [Gene Prediction with Conditional Random Fields.](http://www.cs.umass.edu/~culotta/pubs/crfgene.pdf)Technical Report UM-CS-2005-028. University of Massachusetts, Amherst, 2005.

Yang Wang and Qiang Ji. [A Dynamic Conditional Random Field Model for Object Segmentation in Image Sequences.](http://www.geocities.com/wang_yang_mr/publication/DCRFcvpr05.pdf) In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2005), Volume 1, 2005.

#### 2010

[An Introduction to Conditional Random Fields](http://arxiv.org/PS_cache/arxiv/pdf/1011/1011.4088v1.pdf). Charles Sutton, Andrew McCallum. Foundations and Trends in Machine Learning. To appear. 2011.



### Software

[MALLET](http://mallet.cs.umass.edu/): A Machine Learning for Language Toolkit.

> MALLET is an integrated collection of Java code useful for statistical natural language processing, document classification, clustering, information extraction, and other machine learning applications to text.

[ABNER](http://www.cs.wisc.edu/~bsettles/abner/): A Biomedical Named Entity Recognizer.

> ABNER is a text analysis tool for molecular biology. It is essentially an interactive, user-friendly interface to a system designed as part of the NLPBA/BioNLP 2004 Shared Task challenge.

[MinorThird](http://minorthird.sourceforge.net/).

> MinorThird is a collection of Java classes for storing text, annotating text, and learning to extract entities and categorize text.

[Kevin Murphy's MATLAB CRF code](http://www.cs.ubc.ca/~murphyk/Software/CRF/crf.html).

> Conditional random fields (chains, trees and general graphs; includes BP code).

[Sunita Sarawagi's CRF package](http://crf.sourceforge.net/).

> The CRF package is a Java implementation of conditional random fields for sequential labeling.

[CRF++:Yet Another CRF toolkit](http://crfpp.sourceforge.net/)

+ 如果读者对于基于字标注的中文分词感兴趣，可以很快的利用该工具包构造一个基于条件随机场的中文分词工具，而且性能也不赖。



### CRF Definition

+ $$ P(Y_v| X, Y_w, w \neq v) =  P(Y_v | X, Y_w, w \sim  v)$$

+ $$ w \sim  v $$ 表示在 G=(V, E) 中与节点v相关的节点
+ $$ w \neq v $$ 表示在 G=(V, E) 中与节点v不同的节点



### Linear-Chain CRF Definition

+ Definition
  + $$ P(I_i | O, I_1, ..., I_{i-1}, I_{i+1}, ..., I_T) = P(I_i | O,  I_{i-1}, I_{i+1})$$
  + 加图

+ HMM formula

  + $ y = arg\ max_{y \in Y}^{} p(y|x) = arg\ max_{y \in Y}^{} \frac{p(x, y)}{p(x)} = arg\ max_{y \in Y}^{} p(x, y)$


### Problem One ：Probability calculation

+ Defination
	+ Given **parameters** and observation sequence $O =(O_1, O_2, ..., O_T)$
	+ Calculate the probability of $O =(O_1, O_2, ..., O_T)$'s occurrence

+ Diff from HMM  ：P(O, I) for CRF

    + In HMM
    	$$
    	P(O,I) = P(I_1|\ start)\ \prod_{t=1}^{T-1} P(I_{t+1}|I_t)\ P(end|I_T)\ \prod^{L}_{t=1}P(O_t|I_t) \tag{1}
    	$$

    	$$
    	log P(O,I) = logP(I_1 | start) + \sum_{t=1}^{T-1}logP(I_{t+1}|I_t) + log P(end|I_T) + \sum_{t=1}^{T} logP(O_t|I_t) \tag{2}
    	$$

+ Feature Vector $$\phi(x,y)$$

    + relation between tags and words 

        + weight : 

            + $N_{s,t}(O, I)$ : Number fo tag s and word w appears together in (O, I)
            + demo
                + O = {北京的中心的位置}，I={B-LOC，I-LOC， O，O，O， O，O， O}
                + $$N_{s='O'\ t='的‘} (O, I ) = 2$$

        + feature

            + $log P(t|s)$ : Log probability of **word w given state s **

            $$
            \sum_{t=1}^{T} logP(O_t|I_t) = \sum_{s,w} log P(w|s) \times N_{s,w}(O,I) \tag{3}
            $$

    + relation between tags

        + weight

            + $$N_{s,s^`}(O, I)$$

        + feature
            $$
            logP(I_1 | start) + \sum_{t=1}^{T-1}logP(I_{t+1}|I_t) + log P(end|I_T) = \sum_{s,s^`} log P(s^`|s) \times N_{s,s^`}(O, I) \tag{4}
            $$


        + if there are T possible tags, all feature number between tags is T\*T + 2\*T  

+ 简化形式


    $$
    P(O,I)\ \epsilon \ exp(w\ \cdot\ \phi (O,I) ) \tag {5}
    $$

+ 参数化形式
    $$
    log P(O,I) = \sum_{s,w} log P(w|s) \times N_{s,w}(O,I) + \sum_{s,s^`} log P(s^`|s) \times N_{s,s^`}(O, I) \tag{6}
    $$

+ 矩阵形式	



### Problem Two ：Training(Doing)

- cost function like crosss entropy
  - $$P(y|x) = \frac{P(x,y)}{\sum_{y^`} P(x,y^{`})}$$
    - Maximize what we boserve in training data
    - Minimize what we dont observe in training data
  - $$logP(\hat{y} ^ {n}|x^n) = log P(x^n, \hat{y}^n) - log\sum_{y^{`}} P(x^n,y^{`})$$
- gredient assent
    - $$\theta \rightarrow \theta + \eta \bigtriangledown O(\theta)$$
- After some math
    - to be add
- 改进的迭代尺度法
- 拟牛顿法
    - 条件随机场的BFGS算法



### Problem Three ：Inference(Doing)

+ $$ y = arg\ max\ P(y|x) = arg\ max\ P(x,y) $$

+ viterbi


### Demo(Doing)

- texts

	- 我在北京的北部的朝阳
	- 北极熊生活在冰川以北

- tags

	- {O, O，B-LOC， I-LOC，O，B-LOC，I-LOC，O，B-LOC，I-LOC}
	- {O，O，O，O，O，O，O，O，O，O}

- text set = { '我'， ‘在’， ‘北’， ‘京’，‘的’，‘部’，‘朝’，‘阳’} 

- Feature Vector

	- relation between words and tags

		|       | 我   | 在   | 北   | 京   | 的   | 部   | 朝   | 阳   |
		| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
		| O     | 1    | 1    |      |      | 2    | 1    |      |      |
		| B-LOC |      |      | 2    |      |      |      | 1    |      |
		| I-LOC |      |      |      | 1    |      |      |      | 1    |

	- relation between tags

		|       | O                | B-LOC                    | I-LOC                     | <END>     |
		| ----- | ---------------- | ------------------------ | ------------------------- | --------- |
		| <BEG> | {'BEG我'}        |                          |                           | NULL      |
		| O     | {‘我在’}         | {‘在北’，‘的北’，‘的朝’} |                           |           |
		| B-LOC |                  |                          | {‘北京’，‘北部’， ‘朝阳’} |           |
		| I-LOC | {‘京的’，‘部的’} |                          |                           | {‘阳END’} |

	- 总的 特征模板数

		- 3*8 = 24

		- 3*3 + 2\*3 = 15

		- 当前的特征模板的大小是 39， 具体值如下：

			| O，我 |      | B-LOC, 我 |      | I-LOC |      |      |      |
			| ----- | ---- | --------- | ---- | ----- | ---- | ---- | ---- |
			| O，在 |      |           |      |       |      |      |      |
			| O，北 |      |           |      |       |      |      |      |
			| O，京 |      |           |      |       |      |      |      |
			| O，的 |      |           |      |       |      |      |      |
			| O，部 |      |           |      |       |      |      |      |
			| O，朝 |      |           |      |       |      |      |      |
			| O，阳 |      |           |      |       |      |      |      |
			|       |      |           |      |       |      |      |      |
			|       |      |           |      |       |      |      |      |

