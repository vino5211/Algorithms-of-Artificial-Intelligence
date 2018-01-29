# NLP Clear up
+ ## Basic
  + Seg
  + Pos
  + NER
+ ## 句法理论
  + 依存句法
  + PCFG
+ ## 语料库
  + 语法语料库
    + CTB
+ ## 语义
  + 语义知识库
+ ## 认知 
  + 现代语义学
  + 认知语言学概述
  + 意象图式
  + 隐喻和转喻
  + 构式语法
+ ## 语义计算框架
  + 句子的语义和语法预处理
    + 长句切分与融合
    + 共指消解
  + 语义角色
    + 谓词论元与语义角色
    + PropBank
+ 句子的语义解析
  + 语义依存
  + 完整架构
  + 实体关系抽取

---
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
