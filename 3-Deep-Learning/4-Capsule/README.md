# Summary

## Reference

+ https://zhuanlan.zhihu.com/p/33244896
+ https://zhuanlan.zhihu.com/p/33556066



## Tips

+ Capsule 的革命在于：**它提出了一种新的“vector** **in** **vector out”的传递方案，并且这种方案在很大程度上是可解释的**

+ Demo

  ![](https://pic3.zhimg.com/80/v2-20fb3442f470fdeb59bfa4954e84122e_hd.jpg)

  + 对于测试样本“帆船”，CNN的“三角形”和“四边形”的feature map都会被激活，即该图片中包含了三角形和四边形，就认为这是一个房子。所以说，CNN仅仅考虑了“有没有”的问题，没有考虑feature map的结构关系。这个结构关系包括位置，角度等等

+ Capsule 如何解决这个问题

  + Capsule layer的定义是a nested set of neural layers，即他是多层神经网络的一个集合，而非一个个孤零零的feature map

    ![](https://pic2.zhimg.com/80/v2-42daa6b0a0ae9b8a07fd270854bc4fd9_hd.jpg)

  + 结合房子和帆船的例子，模型中间Primary capsules，我们可以理解为某个capsule表征“房子”，某个capsule表征“帆船”。假设一个“帆船”测试样本输入到网络中，那么Primary capsules中对应“帆船”的capsule会被激活，而对应“房子”的capsule会被抑制，整个模型也就避免了将“帆船”错误地识别为“房子”。某种程度上说，Capsule layer提升了整个模型的表达能力，它比feature maps提取了更多的细节信息

+ 潜在的应用

  + 能建模更多信息，泛化能力比CNN强
  + 图片信息推理
  + 图片描述生成
  + 文本挖掘：替代n-gram 进行建模

