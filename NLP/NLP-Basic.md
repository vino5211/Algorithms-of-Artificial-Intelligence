# NLP

## Tasks

![](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibt9rkyqib37KkCF45lBNmGXgc2QrxlrYtKxR8JPIWd4iaicPtQrcSWibmVodtGKttv91H6AwvJZGxbvA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

- NLP 主要任务 ： 分类、匹配、翻译、结构化预测、与序贯决策过程
- 对于前四个任务，深度学习方法的表现优于或显著优于传统方法


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

