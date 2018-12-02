# HMM MEMM CRF



这三个模型都可以用来做序列标注模型。但是其各自有自身的特点，HMM模型是对转移概率和表现概率直接建模，统计共现概率。而MEMM模型是对转移 概率和表现概率建立联合概率，统计时统计的是条件概率。MEMM容易陷入局部最优，是因为MEMM只在局部做归一化，而CRF模型中，统计了全局概率，在 做归一化时，考虑了数据在全局的分布，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。

举个例子，对于一个标注任务，“我爱北京天安门“，

​                                  标注为" s s  b  e b c e"

对于HMM的话，其判断这个标注成立的概率为 P= P(s转移到s)*P('我'表现为s)* P(s转移到b)*P('爱'表现为s)* ...*P().训练时，要统计**状态转移概率矩阵和表现矩阵。**

对于MEMM的话，其判断这个标注成立的概率为 P= P(s转移到s|'我'表现为s)*P('我'表现为s)* P(s转移到b|'爱'表现为s)*P('爱'表现为s)*..训练时，要统计**条件状态转移概率矩阵和表现矩阵。**

对于CRF的话，其判断这个标注成立的概率为 P= F(s转移到s,'我'表现为s)....F为一个函数，**是在全局范围统计归一化的概率而不是像MEMM在局部统计归一化的概率。**

优点：

（1）**CRF没有HMM那样严格的独立性假设条件**，因而可以容纳任意的上下文信息。特征设计灵活（与ME一样） ————与HMM比较

（2）同时，**由于CRF计算全局最优输出节点的条件概率，它还克服了最大熵马尔可夫模型标记偏置（Label-bias）的缺点**。 ­­————与MEMM比较

（3）CRF是在给定需要标记的观察序列的条件下，**计算整个标记序列的联合概率分布，**而不是在**给定当前状态条件下，定义下一个状态的状态分布。**————与ME比较

缺点：**训练代价大、复杂度高**

 

**HMM模型中存在两个假设：一是输出观察值之间严格独立，二是状态的转移过程中当前状态只与前一状态有关(一阶马尔可夫模型)。**

**MEMM模型克服了观察值之间严格独立产生的问题，但是由于状态之间的假设理论，使得该模型存在标注偏置问题。**

**CRF模型解决了标注偏置问题，去除了HMM中两个不合理的假设。当然，模型相应得也变复杂了。**





- adjust P(a|V) -> 0.1

	<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM13.png" width = "200" height = "200" div align=center />

- Synthetic Data

	- First paper purpose CRF

	- comparing HMM and CRF

		<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM14.png" width = "200" height = "200" div align=center />

- CRF Summary

	<img src="https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/PGM15.png" width = "200" height = "200" div align=center />

