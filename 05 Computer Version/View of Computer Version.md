# View of Computer Version

### Some View 
+ 顺带聊聊CV最近五年的发展史如何？
	+ https://zhuanlan.zhihu.com/p/31337162

### Senmentic Segmantation
- Recurrent Neural Networks for Semantic Instance Segmentation
    - 本项目提出了一个基于 RNN 的语义实例分割模型，为图像中的每个目标顺序地生成一对 mask 及其对应的类概率。该模型是可端对端 + 可训练的，不需要对输出进行任何后处理，因此相比其他依靠 object proposal 的方法更为简单。
    - 论文链接：https://www.paperweekly.site/papers/1355**
    - 代码链接：https://github.com/facebookresearch/InferSent

### Searching for Efficient Multi-Scale Architectures for Dense Image Prediction
+ Google
+ NIPS 2018
+ **首次**提出基于元学习的语义分割模型,可以避免调参时过于玄学,没有规律的特点,受NAS启发设计基于搜索空间的语义分割模型
+ 采用基于随机搜索的策略来找到最优的网络结构参数配置

