# View 

### https://blog.csdn.net/irving_zhang/article/details/78113251

+ 本篇要介绍的论文：Ask Me Anything: Dynamic Memory Networks for Natural Language Processing 是DMN（Dynamic Memory Networks）的开端，在很多任务上都实现了state-of-the-art的结果，如：question answering (Facebook’s bAbI dataset), text classification for sentiment analysis (Stanford Sentiment Treebank) and sequence modeling for part-of-speech tagging (WSJ-PTB)。因为问答系统的模型模型可以拓展到情感分析和词性标注。正如简介中所提到的，很多NLP问题都可以看作QA问题，比如

+ 机器翻译machine translation： (What is the translation into French?); 
+ 命名实体识别named entity recognition (NER) ：(What are the named entity tags in this sentence?); 
+ 词性识别part-of-speech tagging (POS) ：(What are the part-of-speech tags?); 
+ 文本分类classification problems like sentiment analysis： (What is the sentiment?); 
+ 指代问题coreference resolution： (Who does ”their” refer to?).

+ 在github上有具体的代码实现，因此本篇文章就该代码进行讲解，而不再具体实现。（本文依然是按照brightmart 项目选择的模型，该项目是做知乎多文本分类的任务。）

+ 数据集 https://research.fb.com/downloads/babi/