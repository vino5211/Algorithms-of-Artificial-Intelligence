# Outline of Graph Model

### Reference
+ http://www.cnblogs.com/sea-wind/p/4324394.html

### 概率模型 发展到 概率图模型

+ 就是图使得概率模型可视化了，这样就使得一些变量之间的关系能够很容易的从图中观测出来；同时有一些概率上的复杂的计算可以理解为图上的信息传递，这是我们就无需关注太多的复杂表达式了。最后一点是，图模型能够用来设计新的模型。所以多引入一数学工具是可以带来很多便利的，我想这就是数学的作用吧。
+ 当然，我们也可以从另一个角度考虑其合理性。我们的目的是从获取到的量中得到我们要的信息，模型是相互之间约束关系的表示，而数据的处理过程中运用到了概率理论。而图恰恰将这两者之间联系起来了，起到了一个很好的表示作用

### 加法准则和乘法准则

+ 涉及到概率的相关问题，无论有多复杂，大抵都是基于以下两个式子的——加法准则和乘法准则

$$ p(X) = \sum_Y p(X,Y) $$
$$ p(X,Y) = p(Y|X)p(X) $$

+ 第一个式子告诉我们当知道多个变量的概率分布时如何计算单个变量的概率分布，而第二个式子说明了两个变量之间概率的关系

+ 独立时

$$ p(X,Y) = p(Y)p(X) $$

+ 贝叶斯公式

$$ p(X|Y) =  \frac {p(Y|X)p(X)}{p(Y)} $$

### 图模型

+ 下面这张图片描述的就是图，它是由一些带有数字的圆圈和线段构成的，其中数字只是一种标识。我们将圆圈称为节点，将连接圆圈的节点称为边，那么图可以表示为 $G(V,E)$

![](http://upload.wikimedia.org/wikipedia/commons/5/5b/6n-graf.svg)

### Algorithm of Structure Learning(Probability of Graph Model)

+ directed graphical model (Also known as Bayesian Network)
    + Static Bayesian networks
    + Dynamic Bayesian networks
        + Hidden Markov Model
        + Kalman filter
+ Undirection graphical models
    + Markov networks(Also known as Markov random field)
        + Gibbs Boltzman machine
        + Conditional random field

+ Structure Diagram
+ ![](https://pic2.zhimg.com/v2-48dd591b8bc4775b95dd032983c5e729_r.jpg)

## Define of Structure Learing

+ Input and output are both objects with structures
+ objects : sequence, list, tree, bounding box, **not vectors**
+ find function f 
$$
	f : X \rightarrow Y
$$
+ Example Application
	+ Speech recognition
    ​    + X : speech signal(sequence) $\rightarrow$ Y : text(sequence)
    + Translation
        + X : text(sequence) $\rightarrow$ Y : text(sequence)
    + Syntactic Parsing
        + X : sentence $\rightarrow$  Y : parsing tree(tree structure)
    + Object Detection
        + X : image $\rightarrow$ Y : bounding box
    + Summarization
        + X : long ducument $\rightarrow$ summary (short paragraph)
    + Retrieval
        + X : keyword $\rightarrow$ Y : search result(a list of webpage)

### Basic question of Structure Learning
+ Define
    + What does $ F(X,Y) $ look like ?
    + Also can be considered as probability $P(X,Y)$
        + MRF
        + Belief
        + Energy base model
+ Training
    + Find a function F, input is X, Y, output is a real number(R)
    $$ F : X * Y \rightarrow R$$
+ Inference
    + Given an object x, find $ \hat y $ as follow
    $$ \hat y = arg\ \underset{y \epsilon Y }{max} \  F(X,Y)$$ 

