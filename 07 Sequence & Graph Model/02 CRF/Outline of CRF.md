[TOC]

# Markov networks - CRF

### Reference

+ 条件随机场最早由John D. Lafferty提出，其也是[Brown90](http://www.52nlp.cn/strong-author-team-of-smt-classic-brown90)的作者之一，和贾里尼克相似，在离开IBM后他去了卡耐基梅隆大学继续搞学术研究，2001年以第一作者的身份发表了CRF的经典论文 “Conditional random fields: Probabilistic models for segmenting and labeling sequence data”
+ 关于条件随机场的参考文献及其他资料，Hanna Wallach在05年整理和维护的这个页面“[conditional random fields](http://www.inference.phy.cam.ac.uk/hmw26/crf/)”非常不错，其中涵盖了自01年CRF提出以来的很多经典论文（不过似乎只到05年，之后并未更新）以及几个相关的工具包(不过也没有包括CRF++）
+ https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers
+ https://www.bilibili.com/video/av10590361/?p=35
+ https://www.cnblogs.com/pinard/p/7048333.html
+ 李航 统计学习方法 第11章 条件随机场



### CRF Definition

+ $$ P(Y_v| X, Y_w, w \neq v) =  P(Y_v | X, Y_w, w \sim  v)$$

+ $$ w \sim  v $$ 表示在 G=(V, E) 中与节点v相关的节点
+ $$ w \neq v $$ 表示在 G=(V, E) 中与节点v不同的节点



### Linear-Chain CRF Definition

+ Definition
  + $$ P(I_i | O, I_1, ..., I_{i-1}, I_{i+1}, ..., I_T) = P(I_i | O,  I_{i-1}, I_{i+1})$$
  + 加图

+ HMM formula

  + $ y = arg\ max_{y \in Y}^{} p(y|x) = arg\ max_{y \in Y}^{} \frac{p(x, y)}{p(x)} = arg\ max_{y \in Y}^{} p(x, y)$


### Problem One ：Probability calculation

+ Defination
  + Given **parameters** and observation sequence $O =(O_1, O_2, ..., O_T)$
  + Calculate the probability of $O =(O_1, O_2, ..., O_T)$'s occurrence

+ Diff from HMM  ：P(O, I) for CRF

    + In HMM
      $$
      P(O,I) = P(I_1|\ start)\ \prod_{t=1}^{T-1} P(I_{t+1}|I_t)\ P(end|I_T)\ \prod^{L}_{t=1}P(O_t|I_t) \tag{1}
      $$

      $$
      log P(O,I) = logP(I_1 | start) + \sum_{t=1}^{T-1}logP(I_{t+1}|I_t) + log P(end|I_T) + \sum_{t=1}^{T} logP(O_t|I_t) \tag{2}
      $$

+ Feature Vector $$\phi(x,y)$$

    + relation between tags and words 

        + weight : 

            + $N_{s,t}(O, I)$ : Number fo tag s and word w appears together in (O, I)
            + demo
                + O = {北京的中心的位置}，I={B-LOC，I-LOC， O，O，O， O，O， O}
                + $$N_{s='O'\ t='的‘} (O, I ) = 2$$

        + feature

            + $log P(t|s)$ : Log probability of **word w given state s **

            $$
            \sum_{t=1}^{T} logP(O_t|I_t) = \sum_{s,w} log P(w|s) \times N_{s,w}(O,I) \tag{3}
            $$

    + relation between tags

        + weight

            + $$N_{s,s^`}(O, I)$$

        + feature
            $$
            logP(I_1 | start) + \sum_{t=1}^{T-1}logP(I_{t+1}|I_t) + log P(end|I_T) = \sum_{s,s^`} log P(s^`|s) \times N_{s,s^`}(O, I) \tag{4}
            $$


        + if there are T possible tags, all feature number between tags is T\*T + 2\*T  

+ 简化形式
	$$
	P(O,I)\ \epsilon \ exp(w\ \cdot\ \phi (O,I) ) \tag {5}
	$$


+ 参数化形式
    $$
    log P(O,I) = \sum_{s,w} log P(w|s) \times N_{s,w}(O,I) + \sum_{s,s^`} log P(s^`|s) \times N_{s,s^`}(O, I) \tag{6}
    $$

+ 矩阵形式	

    + pass


### Problem Two ：Training(Doing)

- cost function like crosss entropy
  - $$P(y|x) = \frac{P(x,y)}{\sum_{y^`} P(x,y^{`})}$$
    - Maximize what we boserve in training data
    - Minimize what we dont observe in training data
  - $$logP(\hat{y} ^ {n}|x^n) = log P(x^n, \hat{y}^n) - log\sum_{y^{`}} P(x^n,y^{`})$$
- gredient assent
    - $$\theta \rightarrow \theta + \eta \bigtriangledown O(\theta)$$
- After some math
    - to be add
- 改进的迭代尺度法
- 拟牛顿法
    - 条件随机场的BFGS算法



### Problem Three ：Inference(Doing)

+ $$ y = arg\ max\ P(y|x) = arg\ max\ P(x,y) ​$$
+ viterbi
+ Demo


### Demo(Doing)

- texts

  - 我在北京的北部的朝阳
  - 北极熊生活在冰川以北

- tags

  - {O, O，B-LOC， I-LOC，O，B-LOC，I-LOC，O，B-LOC，I-LOC}
  - {O，O，O，O，O，O，O，O，O，O}

- text set = { '我'， ‘在’， ‘北’， ‘京’，‘的’，‘部’，‘朝’，‘阳’} 

- Feature Vector

  - 以下表格中的数值，是预料中统计的特征出现的次数

  - relation between words and tags（通过DL 的方法，实际上是在建模这部分数据，下一个表格）

    |       | 我   | 在   | 北   | 京   | 的   | 部   | 朝   | 阳   |
    | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
    | O     | 1    | 1    |      |      | 2    | 1    |      |      |
    | B-LOC |      |      | 2    |      |      |      | 1    |      |
    | I-LOC |      |      |      | 1    |      |      |      | 1    |

  - relation between tags

    |       | O                | B-LOC                    | I-LOC                     | <END>     |
    | ----- | ---------------- | ------------------------ | ------------------------- | --------- |
    | <BEG> | {'BEG我'}        |                          |                           | NULL      |
    | O     | {‘我在’}         | {‘在北’，‘的北’，‘的朝’} |                           |           |
    | B-LOC |                  |                          | {‘北京’，‘北部’， ‘朝阳’} |           |
    | I-LOC | {‘京的’，‘部的’} |                          |                           | {‘阳END’} |

  - 总的 特征模板数

    - 3*8 = 24

    - 3*3 + 2\*3 = 15


### Next to do

+ 完善特征demo
+ Training
+ Inference