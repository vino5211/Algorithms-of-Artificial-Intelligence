# Teaching Machines to Read and Comprehend

## Reference
+ https://zhuanlan.zhihu.com/p/21343662?refer=paperweekly

## 作者
+ 作者是来自Google DeepMind的科学家Karl Moritz Hermann，是Oxford的博士后

## 主要贡献
+ 一是提出了一种构建用于监督学习的阅读理解大型语料的方法，并开源在Github上，并且给出了两个大型语料，CNN和Daily Mail
+ 二是提出了三种用于解决阅读理解任务的神经网络模型

## 语料构造方法
+ 基本的思路是受启发于自动文摘任务，从两个大型的新闻网站中获取数据源，用abstractive的方法生成每篇新闻的summary，用新闻原文作为document，将summary中去掉一个entity作为query，被去掉的entity作为answer，从而得到阅读理解的数据三元组(document,query,answer)
+ 这里存在一个问题，就是有的query并不需要联系到document，通过query中的上下文就可以predict出answer是什么，这也就失去了阅读理解的意义。因此，本文提出了用entity替换和重新排列的方法将数据打乱，防止上面现象的出现。这两个语料在成为了一个基本的数据集，后续的很多研究都是在数据集上进行训练、测试和对比。处理前和后的效果见下图
![](https://pic3.zhimg.com/80/2220b1ed15fa69bc6cac26abb350bd3e_hd.jpg)

## 三种模型
+ 用神经网络来处理阅读理解的问题实质上是一个多分类的问题，通过构造一些上下文的表示，来预测词表中每个单词的概率，概率最大的那个就是所谓的答案。（任何nlp任务都可以用分类的思路来解决）
+ Deep LSTM Reader
	+ 如下图，其实非常简单，就是用一个两层LSTM来encode query|||document或者document|||query，然后用得到的表示做分类。
	
        ![](https://pic3.zhimg.com/80/55b1bb43845056bdd72ef7679a8cf65a_hd.jpg)

+ Attentive Reader
	+ 这个模型将document和query分开表示
	+ query部分就是用了一个双向LSTM来encode，然后将两个方向上的last hidden state拼接作为query的表示
	+ document这部分也是用一个双向的LSTM来encode，每个token的表示是用两个方向上的hidden state拼接而成，document的表示则是用document中所有token的加权平均来表示，这里的权重就是attention，权重越大表示回答query时对应的token的越重要
	+ 然后用document和query的表示做分类。

![](https://pic1.zhimg.com/80/93c0830072018ea492f0360375d07032_hd.jpg)

+ Impatient Reader
	+ 这个模型在Attentive Reader模型的基础上更细了一步，即每个query token都与document tokens有关联，而不是像之前的模型将整个query考虑为整体。感觉这个过程就好像是你读query中的每个token都需要找到document中对应相关的token。这个模型更加复杂一些，但效果不见得不好，从我们做阅读理解的实际体验来说，你不可能读问题中的每一个词之后，就去读一遍原文，这样效率太低了，而且原文很长的话，记忆的效果就不会很好了。
	![](https://pic1.zhimg.com/80/ce540776aa0edc8da14c8ffca32366ca_hd.jpg)

## 实验部分
+ 作者选了几个baseline作为对比，其中有两个比较有意思，一是用document中出现最多的entity作为answer，二是用document中出现最多且在query中没有出现过的entity作为answer。这个和我们在实际答题遇到不会做的选择题时的应对策略有一点点异曲同工，所谓的不会就选最长的，或者最短的，这里选择的是出现最频繁的。
+ 最终的结果，在CNN语料中，第三种模型Impatient Reader最优，Attentive Reader效果比Impatient Reader差不太多。在Daily Mail语料中，Attentive Reader最优，效果比Impatient Reader好了多一些，见下表：
![](https://pic2.zhimg.com/80/a11ba93eab38bf8041cd669882e64fc9_hd.jpg)

+ (extractive 和 abstractive 的区别)开始在看语料构建方法的时候，我在想应该是用extractive的方法从原文中提取一句话作为query，但看到paper中用的是abstractive的方法。仔细想了一下，可能是因为extractive的方法经常会提取出一些带有指示代词的句子作为摘要，没有上下文，指示代词就会非常难以被理解，从而给后面的阅读理解任务带来了困难，而用abstractive的方法做的话就会得到质量更高的query。
