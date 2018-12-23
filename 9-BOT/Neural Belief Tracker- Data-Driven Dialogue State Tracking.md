# Neural Belief Tracker : Data-Driven Dialogue State Tracking

### ACL 2017

### Reference

+ Steve Young, Milica Gasic, Simon Keizer, Francois Mairesse, Jost Schatzmann,
  Blaise Thomson, Kai Yu. 2010. The Hidden Information Statemodel: A practical framework for
  POMDP-based spoken dialogue management. *Computer Speech and Language* 24: 150-174.

### Background

+ Separate SLU
  +  Given enough data, these models can learn which lexical features are good indicators for a given value and can capture ele- ments of paraphrasing (Mairesse et al., 2009). 
  + sequence labeling
  + Semantic parsing
+ Joint SLU/DST
  + pass

### Neural Belief Tracker

+ Belief State
  + 也叫Dialogue State ，可以翻译成置信状态

+ 在对话过程中，检测 slot-value 对， 完成用户的目标

![](/Users/sunhongchao/Desktop/屏幕快照 2018-12-20 上午11.57.06.png)

+ Input
  + user utterance
    + $$r$$
  + system dialogue acts  preceding user input
    + $$t$$
    + $$t_q, t_s, t_v$$
  + a single candidate slot-value pair that it needs to make decision
    + $$c$$

+ Represent Learing

  + NBT-DNN
  + NBT-CNN

+ Semantic Decoding

  + $$d$$
  + $$d_r$$, $$d_c$$, $$d$$
  + $$r$$ and $$c$$ directly interact through the semantic decoding module
  + 决定用户是否表达了 current candidate pair 
  + 并没有考虑dialogue context
  + Demo
    + 'I want Thai food' match 'food=Thai'
    + 'a pricey restaurent' match 'price=expensive'
  + 需要高质量的 pre-trained word vectors
    +  Delexicalisation-based model 可以处理前面的例子， 但是不能处理latter case
  + Slot : $$c_s$$ , Value:$$c_v$$
    + 多个词组成的 slot 和 value 采用 sum 的方式处理
  + map（$$c_s, c_v)$$ into sinlge vector c
    + $$ c = \delta(W_{c}^{s}(c_s + c_v) + b_{c}^{s})$$
  + match r and c
    + $$d = r \bigotimes c$$
    + $$\bigotimes$$ denotes element-wise vector multiplication

Context Modeling

- $$m$$
- all previous system and user utterances
- the most relevant one is **last system utterance**
- Last system utterance 
  - System Request
    - system aks the user about the vualue of special slot **$$T_q$$**
    - Bot  ：what price range would you like？
    - User ：any
  - System Confirm
    - BOT ：How about Thai Food
    - User ：Yes
    - $$T_s, T_v$$
- Make the Markovian decision to only consider the last set of system
  - change $$T_q, (T_s, T_v)$$  into word vector for system request and confirm acts
  - zero vectors if none
- 计算 system acts, candidate pair$$(c_s, c_v)$$, utterance 的相似性
  - $$m_r = (c_s \cdot t_q) r$$
  - $$m_c = (c_s \cdot t_s) (c_v \cdot t_v) r$$
  - $$\cdot​$$ denote dot predict

+ Binary Decision Maker

# Belief State Update Mechanism

+ Belief Tracking Model after ASR
+ applied to ASR N-best list

+ Dialog turn t
  + $$sys^{t-1}$$ : system output
  + $$h$$ : n ASR hypotheses
  + for any hypothesis $$h^t_i$$, slot s and value v
    + NBT model estimate $$P(s,v|h^t_{i}, sys^{t-1})$$
  + Combine N hypothese
    + $$P(s,v|h^t, sys^{t-1}) =\sum^{N}_{t-1} p^{t}_{i} P(s,v|h^t_{i}, sys^{t-1}) $$

