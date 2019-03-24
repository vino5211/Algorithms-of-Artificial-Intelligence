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

### On the Convergence of Adam and Beyond
  - Neural Network
  - 本文是 ICLR 2018 最佳论文之一。在神经网络优化方法中，有很多类似 Adam、RMSprop 这一类的自适应学习率的方法，但是在实际应用中，虽然这一类方法在初期下降的很快，但是往往存在着最终收敛效果不如 SGD+Momentum 的问题。
  - 作者发现，导致这样问题的其中一个原因是因为使用了指数滑动平均，这使得学习率在某些点会出现激增。在实验中，作者给出了一个简单的凸优化问题，结果显示 Adam 并不能收敛到最优点。
  - 在此基础上，作者提出了一种改进方案，使得 Adam 具有长期记忆能力，来解决这个问题，同时没有增加太多的额外开销。
- Neural Baby Talk
  - Image Captioning
  - 本文是佐治亚理工学院发表于 CVPR 2018 的工作，文章结合了 image captioning 的两种做法：以前基于 template 的生成方法（baby talk）和近年来主流的 encoder-decoder 方法（neural talk）。
  - 论文主要做法其实跟作者以前的工作"Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning"类似：在每一个 timestep，模型决定生成到底是生成 textual word（不包含视觉信息的连接词），还是生成 visual word。其中 visual word 的生成是一个自由的接口，可以与不同的 object detector 对接。
  - 论文链接
  	- https://www.paperweekly.site/papers/1801
  - 代码链接
  	- https://github.com/jiasenlu/NeuralBabyTalk
