# View of 评价指标

### Reference
+ https://www.cnblogs.com/bentuwuying/p/6690836.html

### MRR

### MAP

### ERR

### NDCG

### 比较
＋　NDCG和ERR指标的优势在于，它们对doc的相关性划分多个（>2）等级，而MRR和MAP只会对doc的相关性划分2个等级（相关和不相关）。并且，这些指标都包含了doc位置信息（给予靠前位置的doc以较高的权重），这很适合于web search。然而，这些指标的缺点是不平滑、不连续，无法求梯度，如果将这些指标直接作为模型评分的函数的话，是无法直接用梯度下降法进行求解的