[TOC]

# Maching Learning - Dimension Reduction

## next to do
+ 图片大小

## 为什么要进行降维
+ https://blog.csdn.net/xiongpai1971/article/details/79915047
	+ 降维的目的有二，一个是为了对数据进行可视化，以便对数据进行观察和探索
	+ 另外一个目的是简化机器学习模型的训练和预测。
	+ tsne mnist demo


## Reference
+ LHY ML 2017 Video 24
+ sklearn manifold example
	+ http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
+ t-SNE Laurens van der Maaten
	+ http://lvdmaaten.github.io/tsne/

## Branch Diagram

![](https://camo.githubusercontent.com/fea0f9cf426196493aeb0aafcb6560f133024858/687474703a2f2f696d672e626c6f672e6373646e2e6e65742f3230313530353232313934383031323937)


## Feature selection
![](/home/apollo/Pictures/DR1.png)

## PCA
+ 见纸质笔记
<img src="/home/apollo/Pictures/DR2.png" width="500px" height="500px" />
<img src="/home/apollo/Pictures/DR3.png" width="500px" height="500px" />

+ 左右同时乘以 $(w^1)^T$
<img src="/home/apollo/Pictures/DR4.png" width="500px" height="500px" />


+ 协方差矩阵
	+ 方差
		![](https://images0.cnblogs.com/blog/407700/201307/10144507-9fa45afa80ee4b76929feb47373977ca.gif)
	+ 协方差
		![](https://images0.cnblogs.com/blog/407700/201307/10144507-318040d3955c47f5a2bdd098974bcc79.gif)

    + 这里的cov(x) 是把以上公式中的Y 认为是 $x^T$ 了

+ 拉格朗日乘数法

+ PCA 可以去相关
	+ 不同dimension 间的cov 是 0


+ weakness of PCA
	+ 无法处理label data, 需要 LDA(Supervised)(线性判别分析)
	+ 非线性的无法处理
<img src="/home/apollo/Pictures/DR5.png" width="500px" height="500px" />

+ PCA MNIST
<img src="/home/apollo/Pictures/DR6.png" width="500px" height="500px" />
+ problem 参数是可正可负的，存在多余情况，引入NMF
<img src="/home/apollo/Pictures/DR8.png" width="500px" height="500px" />

+ 笔画清楚很多

<img src="/home/apollo/Pictures/DR9.png" width="500px" height="500px" />

## Word Embedding ( Dimension Reduction for Text)
+ Recorded in Knowledge/Embedding.md



## Neighbor Embedding
+ Manifold Learning(多方面学习,流行学习)
+ LLE(Locally Linear Embedding)
    + Ref
        + http://blog.sina.com.cn/s/blog_82a927880102v2ua.html
        + _先给出一张下面算法得到的图 ，图中第一幅为原始数据，第三个为降维后的数据，可以看出处理后的低维数据保持了原有的拓扑结构。
        + LLE算法可以归结为三步:
            （1）寻找每个样本点的k个近邻点；
            （2）由每个样本点的近邻点计算出该样本点的局部重建权值矩阵；
            （3）由该样本点的局部重建权值矩阵和其近邻点计算出该样本点的输出值

    <img src="http://s6.sinaimg.cn/mw690/002olVxegy6NhaiLyFDc5&690" width="500px" height="500px" />

    + $x^i$ 可以由它的临域内的一些点$x^j$来线性表示，权重是$w_{ij}$_
    <img src="/home/apollo/Pictures/Emb7.png" width="500px" height="500px" />
    + 降维之后
    <img src="/home/apollo/Pictures/Emb8.png" width="500px" height="500px" />
+ Laplacian Eigenmaps(不理解，看完半监督Graph model 之后再整理)
    + Graph based-Model
    + review supervised
        + if connected, $w_{i,j}$ means the similarity of i,j; otherwise,  $w_{i,j}$ is 0
        <img src="/home/apollo/Pictures/Emb9.png" width="500px" height="500px" />
    + Unsupervised
        + use $z^i, z^j$ instead $y^i, y^j$
        + 降维之后的维度空间可以被填满
        + 使用这种方法降维之后再聚类，叫Spectral clustering : clustering on z
        <img src="/home/apollo/Pictures/Emb10.png" width="500px" height="500px" />
+ t-SNE(T-distributed Stochastic Neighbor Embedding)
    + Ref : https://github.com/wepe/MachineLearning/tree/master/ManifoldLearning/DimensionalityReduction_DataVisualizing
    + _problem of previous approaches: similar data are close, but different data may collapse
    + 降维前后的数据有尽可能相同的数据分布
        <img src="/home/apollo/Pictures/Emb11.png" width="500px" height="500px" />
    + Similarity Measure
        + 相似性函数的选择，近的很近，远的很远
        + 在图中，使用t-SNE 比 SNE 产生的相似性差距更大
        <img src="/home/apollo/Pictures/Emb12.png" width="500px" height="500px" />

## Auto Encoder
+ Ref
	+ http://www.cnblogs.com/yangmang/p/7428014.html

## Denosing Auto Encoder
+ http://www.cnblogs.com/yymn/p/4589569.html