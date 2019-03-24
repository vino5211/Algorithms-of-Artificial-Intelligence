# Summary of Graph Model

## Reference
+ http://www.cnblogs.com/sea-wind/p/4324394.html

## Outline
+ Probability Theory
+ Bayesian 
+ Sequence Model
+ Graph Model
+ Probability Graph Model（Structure Learning）
+ Network Model
+ Knowledge Base

## Benefit of Graph Model
+ 图使得概率模型可视化了，使得一些变量之间的关系能够很容易的从图中观测出来
+ 一些概率上的复杂的计算可以理解为图上的信息传递

## Bottleneck of Graph Model

## Probability Theory
+ 加法准则和乘法准则
$$ p(X) = \sum_Y p(X,Y) $$
$$ p(X,Y) = p(Y|X)p(X) $$
+ 第一个式子告诉我们当知道多个变量的概率分布时如何计算单个变量的概率分布，而第二个式子说明了两个变量之间概率的关系
+ 独立时
$$ p(X,Y) = p(Y)p(X) $$
+ 贝叶斯公式
$$ p(X|Y) =  \frac {p(Y|X)p(X)}{p(Y)} $$

## Probability of Graph Model
#### Overview of PGM
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

#### Define of PGM
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

#### Basic question of PGM
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

## Sequence Model
+ HMM
+ Chain CRF

## Graph Model
#### Define of Graph
+ 圆圈称为节点，连接圆圈的节点称为边，那么图可以表示为 $G(V,E)$
![](http://upload.wikimedia.org/wikipedia/commons/5/5b/6n-graf.svg)
#### Math of Graph
+ to be added

## Network Model
+ DeepWalk
+ RandomWalk
+ SDNE
+ LINE

## Knowledge Base
#### Dataset
#### Paper
#### Project