# Decision Tree

## Reference

+ https://blog.csdn.net/u012328159/article/details/70184415 

## Tips
+ 决策树算法起源于E.B.Hunt等人于1966年发表的论文“experiments in Induction”，但真正让决策树成为机器学习主流算法的还是Quinlan（罗斯.昆兰）大神（2011年获得了数据挖掘领域最高奖KDD创新奖），昆兰在1979年提出了ID3算法，掀起了决策树研究的高潮。现在最常用的决策树算法是C4.5是昆兰在1993年提出的。（关于为什么叫C4.5，还有个轶事：因为昆兰提出ID3后，掀起了决策树研究的高潮，然后ID4，ID5等名字就被占用了，因此昆兰只好让讲自己对ID3的改进叫做C4.0，C4.5是C4.0的改进）。现在有了商业应用新版本是C5.0link

## Building of Tree

+ 样本集的信息熵(information entropy) : 

  + 样本集D中第k类样本所占的权重比例是$p_k$, K 为总类别数据， Ent(D)越小，则D的纯度越高（数据为1类是，信息熵为0， 纯度最大）
    $$ Ent(D) = - \sum_{k=1}^{K} p_k log_2 {p_k} ​$$

+ 信息增益

  + 假设离散属性a有V个可能的取值 $$a^1, a^2, …, a^V$$, 如果使用a来对D进行划分，则会产生V个分支节点， 第v个节点包含数据集中所有在特征a上的取值为$a^v$的样本数， 记为$D^v$,  根据上面公式计算出信息熵，在考虑到不同分支包含的样本数量不同， 给每个分支赋予权重$\frac{D^v}{D}$,  由此可以计算出 使用a 对 数据集 D 进行划分得到的信息增益

    $$Gain(D,a) = Ent(D) - \sum_{v=1}^{V} \frac{D^v}{D} Ent(D^v)​$$

+ 信息增益率
+ 基尼指数
+ 停止条件



## Type of Tree

### 分类树

### 回归树

### CART(CART,Classification And Regression Tree) 分类回归树
+ http://www.cnblogs.com/zhangchaoyang/articles/2709922.html
+ https://www.jianshu.com/p/b90a9ce05b28

### ID3

### CART4.5

## Drawback of Decision Tree

+ 容易过拟合

