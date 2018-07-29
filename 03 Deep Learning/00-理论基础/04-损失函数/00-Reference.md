# View of Loss Function


### Biased Importance Sampling for Deep Neural Network Training
- Importance Sampling 在凸问题的随机优化上已经得到了成功的应用。但是在 DNN 上的优化方面结合 Importance Sampling 存在困难，主要是缺乏有效的度量importance 的指标。
- 本文提出了一个基于 loss 的 importance 度量指标，并且提出了一种利用小型模型的 loss 近似方法，避免了深度模型的大规模计算。经实验表明，结合了 Importance Sampling 的训练在速度上有很大的提高。
- 论文链接：https://www.paperweekly.site/papers/1758
- 代码链接：https://github.com/idiap/importance-sampling
