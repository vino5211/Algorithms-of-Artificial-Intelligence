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