[TOC]

# Mulit-Turns Dialog System

# Target

### Tricks

+ 利用多轮的上下文信息，用户画像信息（更新），澄清话术（交互）
+ 提升分类器性能
+ 提升intent和slot 提取的性能

### Context

+ 记录前两句的数据
+ 输入
  + 前两句的文本
  + 当前句的文本
+ 记录前两句的domain, intent, slot
+ 利用前两句的domain, intent, slot 提取当前句的domain, intent, slot
  + 当前句的domain（分类）
    + 当前句的话术
    + 前两句的话术
    + 前两句的domain和intent(利用现有domain 分类器 来做标注)

### User Portrait

### Clarification

# Demo

+ Demo A

  ```
  User ：播放赵雷的歌
  User ：播放成都
  DM : 播放赵雷的成都
  ```

+ Demo B：多轮提槽

  + QQ音乐中搜索'阿刁'
  	+ 张韶涵 - 阿刁
  	+ 赵雷 - 阿刁

  ```
  User : 介绍一下赵雷
  User ：播放阿刁
  ```

+ Demo C：（有待设计NLG）

  ```
  User : 我想听成都
  Bot : 谁的？
  User ：赵雷的
  ```



![](http://img.pmcaff.com/FomDNcyHrznIM_ccW-Y5L0eO6adu-picture)

![](http://www.crownpku.com/images/201709/5.jpg)

# NLU

### 槽

- 词槽
	- 利用用户话中的关键词填写的槽
- 接口槽
	- 利用用户画像及其他场景信息填写的槽
- Demo
	- “我明天要坐火车去上海”
	- 词槽：
		- 出发时间：“明天”
		- 目的地：“上海”
	- 接口槽
		- 出发地：当前位置
- 同一个槽有多种填写方式
- 槽填写的优先级
	- 尝试填写词槽
	- 若失败，尝试填写第一接口槽『用户日程表中隐含的出发地』
	- 若失败，尝试填写第二接口槽『用户当前所在位置』
	- 若失败，判断是否该槽必填
	- 若必填，反问用户，重填词槽 *若非必填，则针对该槽组的填槽过程结束

### 槽的属性：**可默认填写/不可默认填写**

- 可默认填写（非必填）
- 不可默认填写（必填）

### **槽的属性：澄清话术**

### **槽的属性：澄清顺序**

### **槽的属性：平级槽或依赖槽**

- 平级槽：
	- 打车三槽
- 依赖槽
	- 手机号码槽

### 槽位

- 槽是由槽位构成的，一种槽位是一种填槽的方式

- 出发点的槽
  - 槽位1：上下文
  - 槽位2：直接询问
  - 槽位3：GPS定位

### 序列标注

+ deep belief network(DBNs), 取得了优于CRF baseline 的效果
	+ https://arxiv.org/pdf/1711.01731.pdf
		+ 参考文献15和17

# Dialog System

+ Task 1 : Dialog State Update
  + Dialog state encode every information relevant to the system
  + DM maintain a representation of current dialogue state
  + DM update the dialogue state as new information coming
+ Task 2 : Action Selection

  + Make decision on which actions to take on dialogue state
+ Outcome of Dialogue State
  + Update dialogue state
  + One or more select system action(s)

+ ### 分类

  - 开放域
    - 目的：识别用户意图，进入封闭域
  - 封闭域
    - 输入和输出是可以枚举的
    - 对话有明确的目的，且有流程

  ### Initiative

  - 系统主导
  - 用户主导
  - 混合

  ### Challenges

  人的复杂性（complex）、随机性（random）和非理性化（illogical）的特点导致对话管理在应用场景下面临着各种各样的**问题**，包括但不仅限于：

  - **模型描述能力与模型复杂度的权衡**
  - **用户对话偏离业务设计的路径**
    如系统问用户导航目的地的时候，用户反问了一句某地天气情况
  - **多轮对话的容错性**
    如 3 轮对话的场景，用户已经完成 2 轮，第 3 轮由于ASR或者NLU错误，导致前功尽弃，这样用户体验就非常差
  - **多场景的切换和恢复**
    绝大多数业务并不是单一场景，场景的切换与恢复即能作为亮点，也能作为容错手段之一
  - **降低交互变更难度，适应业务迅速变化**
  - **跨场景信息继承**

+ DM 设计

  - 一般在冷启动阶段，会先用规则方法打造一个 DM，快速上线并满足业务需求，收集数据之后再转换成模型。

  - DM 的**设计理念：**

  - **完整性**
    具备建模各种类型对话的能力（不仅仅是slot filling）
  - **独立性**
    当设计（变更）一个场景时，不需要考虑当前场景跳转到其他场景的情况
  - **模块化**
    一些常用的业务组件(如：确认，slot filling，翻页等)，能呈模块化复用(同时允许业务自定义内部的多种属性)

+ DM 里可以有很多小的 dialogs，这些 dialogs 的特点是：

  - 可以重用（reusable modules）
  - 完成一个简单操作（Perform a single operation）
  - 可以被其他 dialog 调用（callable from other dialogs）
  - 可以是全局的（”global”）

+ Global dialog 的特点是：

  - 在 recognizer 能够匹配时 trigger
  - 可以提供一些 conversation support，像是 help/cancel 这些全局功能
  - 应对各种话题切换（Tangents）

+ 通常只有一个 root dialog，在满足下面两个条件的情况下被唤醒

  - dialog stack 里没有其他的 dialog 剩余了
  - 当前时刻 recognizer 并不能 trigger 其他的 dialog

+ dialog stack 会存放目前已经被激活但还没完成的 dialog。dialog 一旦完成就会被 pop 出去

# Dialog state tracking

## Rule Methods

### Key Pharse Reactive Approaches

- 本质是关键词匹配
- 通常是捕获用户的最后一句话的**关键词/关键短语**来进行回应
- ELIZA，AIML（AI标记语言）
- 能支持一定的上下文实现简单的多轮对话
- https://github.com/Shuang0420/aiml
  - 支持python3
  - 支持中文
  - 支持*拓展

## Structure Methods

### Overview of Structure Methods

![](http://ata2-img.cn-hangzhou.img-pub.aliyun-inc.com/fd3925caf61526910ed7145058a05635.png)

### Trees-based: Approaches

- 通常把对话建模为通过树或者有限状态机（图结构）的路径
- 相比 **simple reactive approach**, 这种方法融合了更多的上下文，能用一组**有限的信息交换模板**来完成对话建模
- 适用于
  - **系统主导**
  - **需要从用户搜集特定的信息**
  - 用户对每个问题的答案在**有限的集合中**

### Graph-Base ：FSM（Finite State Machine)

- 把对话在有限的状态内跳转的过程，**每个状态都有对应的动作和回复**

- 如果能从开始节点顺利流转到终止节点，那就认为任务完成

  ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/fsa.png)

  ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/credit_card_eg.png)

- 特点：

  - 人为定义对话流程
  - **完全由系统主导，系统问，用户答**
  - 答非所问的情况直接忽略
  - 建模简单，能清晰明了的把交互匹配到模型
  - **难以扩展，很容易变得复杂**
  - 适用于简单任务，对简单信息获取很友好，难以处理复杂的问题
  - 缺少灵活性，表达能力有限，输入受限，对话结构/流转路径受限
  - 对特定领域要设计 task-specific FSM，简单的任务 FSM 可以比较轻松的搞定，但稍复杂的问题就困难了，毕竟要考虑对话中的各种可能组合，编写和维护都要细节导向，非常耗时。一旦要扩展 FSM，哪怕只是去 handle 一个新的 observation，都要考虑很多问题。实际中，通常会加入其它机制（如变量等）来扩展 FSM 的表达能力

  ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8hH8XW6rNxoNjNqZIwaicl3CnXChz1ko0yGT9nDUeh5octsVlDnpydoQ/640?wx_fmt=png)

## Principle-based Methods

### Frame-based Approaches

- 通过允许多条路径更灵活的获取信息，拓展了**FSM**
- 将对话构建成一个**填槽**的过程
- 槽就是多轮对话过程中将初步用户意图转化为明确用户指令所需要补全的信息
- 一个槽与任务处理中所需要获取的一种信息相对应
- 槽直接没有顺序，缺什么槽就向用户询问对应的信息

![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/slot_filling.png)

- 要素
  - Frame：槽位的集合，定义了需要用户提供什么信息
  - 对话状态：记录了那些槽位已经被填充了
  - 行为选择：下一步该做什么，填充什么槽位，还是进行何种操作
    - 行为选择可以按照槽位填充/槽位加权填充，或者利用本体选择
- 基于**框架/模板**的系统本质上是一个生成系统
  - 不同类型的输入激发不同的生成规则
  - 每个生成能够灵活填入对应的模板
  - 常常用于用户可能采取的行为相对有限，只希望用户在这些行为进行少许转化的场合
- 特点
  - 用户回答可以包含任何一个片段/全部的槽信息
  - 系统来决定下一个行为
  - 支持混合主导型系统
  - 相对灵活的输入，支持多种输入/多种顺序
  - 适用于相对复杂的信息获取
  - 难以应对更复杂的情境
  - 缺少层次
- 更多可参考[槽位填充与多轮对话](https://coffee.pmcaff.com/article/971158746030208/pmcaff?utm_source=forum&from=related&pmc_param%5Bentry_id%5D=950709304427648)

### Agenda + Frame(CMU Communicator)

- 对frame model 进行了改进
- 有了层次结构
- 能应对更复杂的信息获取
- 能支持话题切换、回退、退出
- 要素
  - product
    - 树的结构
    - 反应为完成这个任务需要所有信息的顺序
    - 相比Tree and FSM approach, 这里的产品数(product tree)的创新在于它是动态的，可以在session中对树进行一系列操作比如加一个子树或挪动子树
  - process
    - agenda
    - handler
- http://link.zhihu.com/?target=http%3A//www.cs.cmu.edu/%7Exw/asru99-agenda.pdf

### Information State Approaches

- 提出的背景

  - 很难去评估各种DM系统
  - 理论和实践模型存在很大的gap
    - 理论的模型
      - logic-based, BDI, plan-based, attention/intention
    - 实践的模型
      - finite-state 或者 frame-based

- Information State Models 作为对话建模的形式化理论，为工程化实现提供了理论指导

- 为改进当前的对话系统提供了大的方向

- Information-state theory 的关键是识别对话中流转信息的relevant aspects

  - 这些成分是如何被更新
  - 更新过程如何控制

- 理论框架

  ![](https://pic2.zhimg.com/80/v2-bd700b2e509e7d2d84a8ffad91a9ce55_hd.jpg)

- 要素

  - Statics
    - Informational components
    - Formal representaions
  - Dynamics
    - dialog move
    - update rules
    - update strategy

- 意义：

  - 可以遵循这一理论体系来构建/分析/评价/改进对话系统
- 基于information state的系统有
  - TrindKit Systems
    - GoDis
    - MIDAS
    - EDIS
    - SRI Autoroute
  - Successor Toolkits
    - Dipper
    - Midiki
  - Other IS approaches
    - Soar
    - AT&T MATCH system

### Plan-based Approaches

- BDI(Belief, Desire, Intention)模型
  - Cohen and Perrault 1979
  - Perrault and Allen 1980
  - Allen and Perrault 1980
- 基本假设
  - Cohen and Perrault 1979 提到的 **AI Plan model** : 一个试图发现信息的行为人，能够利用标准的plan找到让听话人找到说话人该信息的plan
  - Perrault and Allen 1980 和 Allen and Perrault 1980 将 BDI 应用于理解，特别是间接言语语效的理解，本质上是对 Searle 1975 的 speech acts 给出了可计算的形式体系
- 概念
  - goals
  - actions
  - plan construction
  - plan inference

## Statistical Methods

+ 基于统计的方法（A Survey on Dialogue Systems:
  Recent Advances and New Frontiers）
  - 会对每轮对话都计算每个slot的概率分布
  - robust sets of hand-crafted rules[93]
  - conditional random fields[26;25;63]
  - maximun entropy models[98]
  - web-style-ranking[101] 

### POMDP

+ 简单来说，它将对话表示成一个部分可见的马尔可夫决策过程
+ 所谓部分可见，是因为DM的输入是存在不确定性的，例如NLU的结果可能是错误的
+ 因此，对话状态不再是特定的马尔可夫链中特定的状态，而是针对所有状态的概率分布
+ 在每个状态下，系统执行某个动作都会有对应的回报（reward）
+ 基于此，在每个对话状态下，选择下一步动作的策略即为选择期望回报最大的那个动作
+ 这个方法有以下几个优点：
  + 只需定义马尔可夫决策过程中的状态和动作，状态间的转移关系可以通过学习得到；
  + 使用强化学习可以在线学习出最优的动作选择策略

+ 缺点
  + 即仍然需要人工定义状态，因此在不同的领域下该方法的通用性不强

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8YLZbrB8L8lciaEjHwjDUCtEBDODCKnXoicPYFyH4C4XZicialbbVvFpPnQ/640?wx_fmt=png)

- http://www.sohu.com/a/146206452_500659

![](http://img.mp.itc.cn/upload/20170605/ca386549107c4478bda222f030dd454f_th.jpg)

## RL-Based Methods

- 前面提到的很多方法
  - 需要人工来定规则的（hand-crafted approaches），然而人很难预测所有可能的场景，这种方法也并不能重用，换个任务就需要从头再来。
  - 而一般的基于统计的方法又需要大量的数据。
  - 再者，对话系统的评估也需要花费很大的代价。
  - 这种情况下，强化学习的优势就凸显出来了。RL-Based DM 能够对系统理解用户输入的不确定性进行建模，让算法来自己学习最好的行为序列。首先利用 simulated user 模拟真实用户产生各种各样的行为（捕捉了真实用户行为的丰富性），然后由系统和 simulated user 进行交互，根据 reward function 奖励好的行为，惩罚坏的行为，优化行为序列。由于 simulated user 只用在少量的人机互动语料中训练，并没有大量数据的需求，不过 user simulation 也是个很难的任务就是了
- ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/rl_arch.png)

## DL-Based Methods

+ 基本思路
  + 它的基本思路是直接使用神经网络去学习动作选择的策略，即将NLU的输出等其他特征都作为神经网络的输入，将动作选择作为神经网络的输出。这样做的好处是，对话状态直接被神经网络的隐向量所表征，不再需要人工去显式的定义对话状态。当然这种方法的问题时需要大量的数据去训练神经网络，其实际的效果也还有待大规模应用的验证。助理来也的对话系统中有尝试用该方法，但更多的还是传统机器学习方法和基于深度学习的方法结合

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8SnuXee0JpSC8FJNenrKD5P5VUfhYrmjOTEBZtokDsbnghPclibbJKWw/640?wx_fmt=png)

- 下文介绍了深度学习在信念跟踪上的应用, 在一个domain上可迁移到新的domain上
  - Deep neural network approach for the dialog state tracking challenge
- 下文提出了multi-domain RNN dialog state tracking models
  - Multidomain dialog state tracking using recurrent neural networks. 
- 下文提出了neural belief tracker(NBT)来识别slot-value对
  - Neural belief tracker: Data-driven dialogue state tracking.
- ALC 2018 **Fully NBT**
  - ![](https://pic1.zhimg.com/80/v2-f093af74079578025955f63b6bfe1c24_hd.jpg)

# Dialog policy optimization

+ 根据当前对话状态做出下一步反应
+ 线上购物的场景中，若上一步识别出的对话状态是“Recommendation”，那下一就会有推荐的action
+ 有监督学习
	+ rule-based agent 来做热启动，然后监督学习会在rule生成的action上进行
	+ Learning to respond with deep neural networks for retrieval-based humancomputer conversation system. 
+ 端到端强化学习
	+ Strategic dialogue management via deep reinforcement learning. arxiv.org, 2015

# NLG

- 闲聊：对于闲聊机器人来讲，往往在大量语料上用一个seq2seq的生成模型，直接生成反馈给用户的自然语言
- 垂直领域的以任务为目标的客服chatbot中往往不适用
- 客户需要的是准确的解决问题的答案



# User Modeling



