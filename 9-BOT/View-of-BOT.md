
+ Question answering over Freebase (single-relation)
	+ https://github.com/quyingqi/kbqa-ar-smcnn
- (***)基于CNN的阅读理解式问答模型：DGCNN
  -  Dilate Gated Convolutional Neural Network
  - Ref : 一文读懂「Attention is All You Need」| 附代码实现
  - 本模型——我称之为 DGCNN——是基于 CNN 和简单的 Attention 的模型，由于没有用到 RNN 结构，因此速度相当快，而且是专门为这种 WebQA 式的任务定制的，因此也相当轻量级。
  - SQUAD 排行榜前面的模型，如 AoA、R-Net 等，都用到了 RNN，并且还伴有比较复杂的注意力交互机制，而这些东西在 DGCNN 中基本都没有出现。
  - 这是一个在 GTX1060 上都可以几个小时训练完成的模型！
  - CIPS-SOGOU/WebQA

-WebQA
​	-  WebQA 的参考论文 Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question :
​	- 1. 直接将问题用 LSTM 编码后得到“问题编码”，然后拼接到材料的每一个词向量中；
​	- 2. 人工提取了 2 个共现特征；
​	- 3. 将最后的预测转化为了一个序列标注任务，用 CRF 解决。
