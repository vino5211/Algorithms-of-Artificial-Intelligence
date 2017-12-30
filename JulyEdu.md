微积分与矩阵知识在机器学习中如何应用
逻辑回归如何在海量工业实战数据下应用
如何对商品销量进行高准确率预测
如何使用隐马尔可夫模型（HMM）对中文进行分词
如何使用随机森林和支持向量机（SVM）对数据进行分类
如何使用Tensorflow构建RNN模型进行分类预测
如何使用LDA等对文档进行分类
如何自己构建数据集并使用Caffe进行分类
如何使用xgboost与lightGBM在Kaggle比赛中获胜
如何使用循环神经网络抓取文本特征



第一阶段 夯实数学基础
第1课 机器学习中的微积分与矩阵（管）
Taylor展式、梯度下降和牛顿法初步 
特征向量、对称矩阵对角化、线性方程

第2课 概率与凸优化（邓）
矩估计、极大似然估计
凸集、凸函数、凸优化、KKT条件

第二阶段 掌握基本模型 打开ML大门
第3课 回归问题与应用（寒）
知识内容：线性回归、logistic回归、梯度下降 
实践示例：分布拟合与回归、用LR分类与概率预测 
工程经验：实际工程海量数据下的logistic回归使用，包括样本处理、特征处理、算法调优和背后的原理

第4课 决策树、随机森林、GBDT（加）
知识内容：决策树 随机森林、GBDT 
实践案例：使用随机森林进行数据分类

第5课 SVM（冯）
知识内容：线性可分支持向量机、线性支持向量机、非线性支持向量机、SMO 
实践案例: 使用SVM进行数据分类

第6课 最大熵与EM算法（褚）
熵、相对熵、信息增益、最大熵模型、IIS、GMM

第三阶段 重中之重 特征工程
第7课 机器学习中的特征工程处理（寒）
知识内容：数据清洗、异常点处理、特征抽取、选择与组合策略 
实践动手：特征处理与特征选择工具与模板

第8课 多算法组合与模型最优化（寒）
知识内容：机器学习问题场景分析、算法选择、模型构建、模型性能分析与优化策略 
实践动手：构建模型组合策略工具与模板

第四阶段 工业实战 在实战中掌握一切
第9课 sklearn与机器学习实战（寒）
知识内容：sklearn板块介绍，组装与建模流程搭建 
实践案例：经典Titanic案例，商品销量预测案例等

第10课 高级工具xgboost/lightGBM与建模实战（寒）
知识内容：xgboost与lightGBM使用方法与高级功能 
实践案例：Titanic与商品销量预测进阶，Kaggle案例实战

第11课 用户画像与推荐系统（寒）
知识内容：基于内容的推荐，协同过滤，隐语义模型，learning to rank，推荐系统评估 
实践案例：实际打分数据上的推荐系统构建

第12课 聚类（赵）
K-means/K-Medoid/层次聚类
实践示例：K-means代码实现和实际应用分析

第13课 聚类与推荐系统实战（寒）
案例：用户聚类结合推荐算法，构建推荐系统完整案例（送完整可运行的代码）

第五阶段 高阶知识 深入机器学习
第14课 贝叶斯网络（冯）
朴素贝叶斯、有向分离、马尔科夫模型

第15课 隐马尔科夫模型HMM（冯）
概率计算问题、参数学习问题、状态预测问题 
实践案例：使用HMM进行中文分词

第16课 主题模型（加）
pLSA、共轭先验分布、LDA 
实践案例：使用LDA进行文档分类

第六阶段 迈入深度学习 打开DL大门
第17课 神经网络初步（寒）
知识内容：全连接神经网络、反向传播算法与权重优化，训练注意点 
实践案例：构建神经网络解决非线性切分问题

第18课 卷积神经网络与计算机视觉（彭）
知识内容：卷积神经网络结构分析、过拟合与随机失活，卷积神经网络理解 
实践案例：工业界常用网络结构与搭建

第19课 循环神经网络与自然语言处理（寒）
知识内容：循环神经网络、长时依赖问题与长短时记忆网络，BPTT算法 
实践案例：利用循环神经网络生成文本、学汪峰写歌词

第20课 深度学习实践（寒） 
知识内容：Caffe应用要点、TensorFlow/Keras简介 
实践案例：用Caffe在自己的数据集上完成分类，用Tensorflow构建RNN模型分类预测
第一周 夯实DL必备基础
第1课 夯实深度学习数据基础（管）
1. 必要的微积分、概率统计基础
2. 必要的矩阵、凸优化基础
3. 实战：numpy与高效计算
第2课 DNN与混合网络：google Wide&Deep（寒）
1. 多分类softmax与交叉熵损失
2. 人工神经网络与BP+SGD优化
3. 实战：数据非线性切分+google wide&deep 模型实现分类

第二周 从CNN入手，掌握主流DL框架
第3课 CNN:从AlexNet到ResNet（寒）
1. 卷积神经网络层级结构详解，可视化理解
2. 典型卷积神经网络结构(AlexNet,VGG,GoogLeNet,ResNet)讲解
3. 实战：搭建CNN完成图像分类示例
第4课 NN框架：caffe, tensorflow与pytorch（寒）
1. Caffe的便捷图像应用
2. TensorFlow与搭积木一样方便的Keras
3. facebook的新秀pytorch
4. 实战：用几大框架完成DNN与CNN网络搭建与分类

第三周 CNN延伸：生成对抗与图像风格转化
第5课 造出你要的视界：生成对抗网络GAN（寒）
1. 无监督学习与图像生成
2. 生成对抗网络与原理
3. 实战：DCGAN图像生成
第6课 图像风格转化（加）
1. 秒变文艺：neural style将照片转换成大师佳作
2. 手推公式理解neural style原理
3. 实战：neural-style与fast neural-style代码讲解

第四周 掌握自然语言处理中的神经网络
第7课 RNN/LSTM/Grid LSTM（寒）
1. 序列数据与循环神经网络
2. RNN/LSTM/Grid LSTM
3. 实战：RNN文本分类
第8课 RNN条件生成与attention（寒）
1. RNN条件生成与attention
2. “看图说话”原理
3. google神经网络翻译系统

第五周 迁移学习与增强学习
第9课 增强学习与Deep Q Network（寒）
1. 马尔科夫决策过程
2. 价值函数与策略评价、学习
3. Deep Q network
4. 实战：用Tensorflow搭建Deep Q learning玩Flappy bird
第10课 物体检测与迁移学习（寒）
1. RCNN，Fast-RCNN到Faster-RCNN
2. Fine-tune，保守训练，层转移，多任务学习
3. 领域对抗训练

---
Regression: pdf,pptx,video (2017/03/02)
Where does the error come from?
Gradient Descent: pdf,pptx,video (2017/03/09, recorded at 2016/10/07)
Classification: Probabilistic Generative Model 
Classification: Logistic Regression 
Introduction of Deep Learning 
Backpropagation 
“Hello world” of Deep Learning 
Tips for Deep Learning 
Convolutional Neural Network 
Why Deep? 
Semi-supervised Learning 
Unsupervised Learning: Principle Component Analysis 
Unsupervised Learning: Neighbor Embedding 
Unsupervised Learning: Deep Auto-encoder 
Unsupervised Learning: Word Embedding 
Unsupervised Learning: Deep Generative Model
Transfer Learning 
Recurrent Neural Network 
Matrix Factorization
Ensemble 
Structured Learning 
Reinforcement Learning 