+ (15) represent learning
 + Share statistical information in diff models 
 + 分布式表示学习
 	+ 聚类
 	+ K-近邻
 	+ 决策树
 	+ 高斯混合体和专家混合体
 	+ 具有高斯核的核机器
 	+ 基于n-gram的语言或翻译模型


+ (16) structured probalilistic model
	+ 非结构化的挑战
		+ 基于表格操作计算量太大

	+ 使用图描述模型结构
		+ 有向模型
			+ 也称为信念网络（belief network）或者贝叶斯网络（Bayesian network）
			+ 有箭头指向的:a 指向 b, 说明b的概率分布依赖a的取值
			+ 正式: x的有向概率模型通过有向无环图和局部条件概率分布来定义
				$$p(t_0,t_1,t_2) = p(t_0)p(t_1|t_0)p(t_2|t_1)
                $$
        + 无向模型
        	+ 也称马尔可夫岁机场(Markov random field)，或马尔可夫网络(Markov network)
        	+ 相互作用间没有明确的方向
        	+ 正式：无向模型是一个定义在无向图(原文是无向模型)上的结构化概率模型，对于图中的每一个团，一个因子(也称团势能)，衡量了团中变量每一种可能的联合状态所对应的密切程度。这些因子都被限制为非负的，他们一起定义了 未归一话概率（unnormalized probability function）:
        		$$
                	\widetilde{p}(x)) = {\prod_{c\epsilon G}^{}}\phi (c)
                $$
        + 配分函数
        + 基于能量的模型
        + 分离和d-分离
        + 在有向图和无向图之间转换
    + 从图模型中采样
    + 结构化建模的优势
+（17）蒙特卡罗方法
