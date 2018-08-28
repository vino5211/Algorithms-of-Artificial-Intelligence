# View of Deep Learning

### Reference
+ Deep Learning: A Bayes
	+ https://www.cnblogs.com/dadadechengzi/p/6962414.html
	+ https://arxiv.org/abs/1706.00473
	+ Deep learning is a form of machine learning for nonlinear high dimensional pattern matching and prediction. By taking a Bayesian probabilistic perspective, we provide a number of insights into more efficient algorithms for optimisation and hyper-parameter tuning. Traditional high-dimensional data reduction techniques, such as principal component analysis (PCA), partial least squares (PLS), reduced rank regression (RRR), projection pursuit regression (PPR) are all shown to be shallow learners. Their deep learning counterparts exploit multiple deep layers of data reduction which provide predictive performance gains. Stochastic gradient descent (SGD) training optimisation and Dropout (DO) regularization provide estimation and variable selection. Bayesian regularization is central to finding weights and connections in networks to optimize the predictive bias-variance trade-off. To illustrate our methodology, we provide an analysis of international bookings on Airbnb. Finally, we conclude with directions for future research.ian Perspective 
+ Bayesian Deep Learning
	+ 论文链接：https://arxiv.org/abs/1709.05870
	+ GitHub：https://github.com/thu-ml/zhusuan
	+ 这种结合贝叶斯方法和深度学习各自优势的新研究方向称为贝叶斯深度学习（Bayesian Deep Learning /BDL）
	+ BDL 的组成包括了传统的贝叶斯方法，以概率推断（probabilistic inference）为主的深度学习方法，以及它们的交叉
	+ BDL 的一个独有的特性是，随机变量间的确定性变换可以利用典型的深度神经网络表达的参数化公式从数据中自动学习 (Johnson et al., 2016)，而在传统的贝叶斯方法中，变换通常是一个简单的解析形式，比如指数函数或者内积。
	+ 贝叶斯深度学习的一个关键挑战是后验推断（posterior inference），通常对于这样的模型来说是很难处理的，需要复杂的近似方法。
	+ 虽然利用变分推断和蒙特卡罗方法已经取得了很多进展 (Zhu et al., 2017)，对于开发者来说还是很难上手。此外，虽然变分推断和蒙特卡罗方法有其一般形式，对于一个有经验的研究员来说，为一个特定的模型寻找方程并实现所有的细节仍然是很困难的，过程中很容易出现错误，并需要花费大量时间调试

### 频率学派还是贝叶斯学派？聊一聊机器学习中的MLE和MAP
+ https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/78999639
+ 频率学派 - Frequentist - Maximum Likelihood Estimation (MLE，最大似然估计)
+ 贝叶斯学派 - Bayesian - Maximum A Posteriori (MAP，最大后验估计)

### 变分推断

### 蒙特卡罗方法