# NLP
## Methods
+ Co-occurrence matrix
+ Word bag model
+ TF-IDF

## Seg
+ 常用分词方法总结分析
	+ https://blog.csdn.net/cuixianpeng/article/details/43234235
+ 分词：最大匹配算法（Maximum Matching）
	+ refer:
		+ http://blog.csdn.net/yangyan19870319/article/details/6399871
	+ 算法思想：
		+ 正向最大匹配算法：从左到右将待分词文本中的几个连续字符与词表匹配，如果匹配上，则切分出一个词。但这里有一个问题：要做到最大匹配，并不是第一次匹配到就可以切分的 。我们来举个例子：
           待分词文本：   content[]={"中"，"华"，"民"，"族"，"从"，"此"，"站"，"起"，"来"，"了"，"。"}
           词表：   dict[]={"中华"， "中华民族" ， "从此"，"站起来"}
			(1) 从content[1]开始，当扫描到content[2]的时候，发现"中华"已经在词表dict[]中了。但还不能切分出来，因为我们不知道后面的词语能不能组成更长的词(最大匹配)。
			(2) 继续扫描content[3]，发现"中华民"并不是dict[]中的词。但是我们还不能确定是否前面找到的"中华"已经是最大的词了。因为"中华民"是dict[2]的前缀。
			(3) 扫描content[4]，发现"中华民族"是dict[]中的词。继续扫描下去：
			(4) 当扫描content[5]的时候，发现"中华民族从"并不是词表中的词，也不是词的前缀。因此可以切分出前面最大的词——"中华民族"。
			由此可见，最大匹配出的词必须保证下一个扫描不是词表中的词或词的前缀才可以结束。

## Pos
+ NLP第八篇-词性标注
	+ https://www.jianshu.com/p/cceb592ceda7
	+ 基于统计模型的词性标注方法
	+ 基于规则的词性标注方法
## NER

## stem and lemma
+ 词干提取(stemming)和词型还原(lemmatization)
	+ 词形还原（lemmatization），是把一个任何形式的语言词汇还原为一般形式（能表达完整语义）
	+ 而词干提取（stemming）是抽取词的词干或词根形式（不一定能够表达完整语义）
	```
    # 词干提取(stemming) :基于规则
		from nltk.stem.porter import PorterStemmer
		porter_stemmer = PorterStemmer()
		porter_stemmer.stem('wolves')

        output is :'wolv'

        # 词性还原(lemmatization) : 基于字典，速度稍微慢一点
		from nltk.stem import WordNetLemmatizer
		lemmatizer = WordNetLemmatizer()
		lemmatizer.lemmatize('wolves')

        output is :'wolf'
    ```

