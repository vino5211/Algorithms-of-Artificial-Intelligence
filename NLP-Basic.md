# NLP
## Methods
+ Co-occurrence matrix
+ Word bag model
+ TF-IDF

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

