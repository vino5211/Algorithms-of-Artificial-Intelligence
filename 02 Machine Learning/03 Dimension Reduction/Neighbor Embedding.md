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