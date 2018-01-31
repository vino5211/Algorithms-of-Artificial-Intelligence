# Clear NLTK
+ Installing Third Party Software
	+ https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software
+ corenlp nltk
	+ standford parser and nltk
		+ https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk
+ 常用函数
	+ 关键词的上下文
	+ 相似上下文
	+ 共同上下文
	+ 生成随意文本--generate
	+ 词汇计数
		+ len:文本中出现的词和标点，从文本头到文本尾
		+ set:
			+ sorted(set(text1)) # 获取文本text1的词汇表，并按照英文字母排序
			+ len(set(text1)) # 获取文本text1词汇表的数量（词类型）
		+ 特定词的出现次数和占比：
			+ text3.count("smote") #单词smote在文本中出现次数 
			+ 100 * text3.count("smote") / len(text3) #获取单词的占比 
+ nltk 中文处理方案
	+ 我感觉用nltk 处理中文是完全可用的。
	其重点在于中文分词和文本表达的形式。
    中文和英文主要的不同之处是中文需要分词。
    因为nltk 的处理粒度一般是词，所以必须要先对文本进行分词然后再用nltk 来处理（不需要用nltk 来做分词，直接用分词包就可以了。严重推荐**结巴分词**，非常好用）。
    中文分词之后，文本就是一个由每个词组成的长数组：[word1, word2, word3…… wordn]。之后就可以使用nltk 里面的各种方法来处理这个文本了。
    比如用**FreqDist**统计文本词频，用**bigrams**把文本变成双词组的形式：[(word1, word2), (word2, word3), (word3, word4)……(wordn-1, wordn)]。
    再之后就可以用这些来计算文本词语的信息熵、互信息等。再之后可以用这些来选择机器学习的特征，构建分类器，对文本进行分类（商品评论是由多个独立评论组成的多维数组，网上有很多情感分类的实现例子用的就是nltk 中的商品评论语料库，不过是英文的。但整个思想是可以一致的）。
