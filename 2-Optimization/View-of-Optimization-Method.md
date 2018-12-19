# View of Optimization Method

### Learning to Warm-start Bayesian Hyperparameter Optimization
+ NIPS 2017 BayesOpt Workshop
+ 论文提出了一种基于迁移学习的初值设定方法,用来提升贝叶斯优化搜索最优超参数的效率
+ 文中通过学习一个基于DNN的距离函数, 找到最相似的K个dataset, 将这些dataset的最优超参数设定为target任务的几个初值,开始迭代优化

### NSGA-NET: A multi-objective Genetic Algorithm for Neural Architecture Search
+ 密歇根大学
+ 提出了一种多目标神经网络搜索架构 NSGA-Net
+ 整个算法框架基于**进化**算法思路, 通过crossover 和 mutation 进行exploration, 然后基于已有的知识,学习一个基于贝叶斯优化模型(BOA), 用于exploitation
+ 此处的BOA和自动调参中常见的贝叶斯优化不同, 是一种EDA算法, 用贝叶斯网络作为概率模型

