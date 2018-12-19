# View of Dialog System

### Links

+ https://speechlab.sjtu.edu.cn/pages/sz128/homepage/year/08/21/SLU-review-introduction/

### Algorithm

+ MDP 
+ POMDP
+ Reinforcement learning
+ end2end

### 对话管理的一些思考

+ https://yq.aliyun.com/articles/276269
+ ![](http://ata2-img.cn-hangzhou.img-pub.aliyun-inc.com/fd3925caf61526910ed7145058a05635.png)

+ ![](http://ata2-img.cn-hangzhou.img-pub.aliyun-inc.com/fd3925caf61526910ed7145058a05635.png)

### 知乎专栏:基于深度学习的对话系统

+ https://zhuanlan.zhihu.com/c_143965981
+ https://blog.csdn.net/irving_zhang/article/details/78865708

### Sequence-to-Sequence Learning for End-to-End Dialogue Systems

### Feudal Reinforcement Learning for Dialogue Management in Large Domains
- 本文来自剑桥大学和 PolyAI，论文提出了一种新的强化学习方法来解决对话策略的优化问题
- https://www.paperweekly.site/papers/1756

### End-to-End Optimization of Task-Oriented Dialogue Model with Deep Reinforcement Learning
- 一篇基于强化学习的端到端对话系统研究工作，来自 CMU 和 Google。
- 论文链接：http://www.paperweekly.site/papers/1257

### Learning to Ask Question in Open-domain Conversational Systems with Typed Decoders
+ ACL 2018 黄民烈
+ 深度对话模型问题:语义理解问题, 上下文理解问题, 个性身份一致性问题
+ 通过向用户提问, 能够将对话更好的进行下去
+ 提出一个好问题, 也体现了机器对人类语言的理解能力
+ 一个问题包括: interrogative(询问词), topic word(主题词), ordinary word(普通词)
+ 基于　encoder-decoder 的框架, 提出两种decoders(STD和HTD), 来估计生成出的句子中每个位置这三种词的分布

- SIGIR 2018 | 通过深度模型加深和拓宽聊天话题，让你与机器多聊两句
  - 目前大多数基于生成的对话系统都会有很多回答让人觉得呆板无趣，无法进行有意思的长时间聊天。近日，山东大学和清华大学的研究者联合提出了一种使用深度模型来对话题进行延展和深入的方法 DAWnet。
  - 该方法能有效地让多轮对话系统给出的答复更加生动有趣，从而有助于实现人与机器的长时间聊天对话。机器之心对该研究论文进行了摘要编译。此外，研究者还公布了他们在本论文中所构建的数据集以及相关代码和参数设置
  - 论文、数据和代码地址：https://sigirdawnet.wixsite.com/dawnet

### https://zhuanlan.zhihu.com/p/31828371
### Outline
+ Chit-Chat-oriented Dialogue Systems： 闲聊型对话机器人，产生有意义且丰富的响应。
+ Rule-based system：对话经过预定义的规则（关键词、if-else、机器学习方法等）处理，然后执行相应的操作，产生回复。（ELIZA系统，如果输入语句中没有发现预定义规则，则生成generic的响应）。缺点是规则的定义，系统越复杂规则也越多，而且其无法理解人类语言，也无法生成有意义的自然语言对话。处在比较浅层的阶段；
+ IR-based Systems：信息检索或者最近邻方法，要求生成的响应与对话存在语义相关性（VSM、TF-IDF、page-Rank、推荐等排序方法）。有点是比生成模型简单，直接从训练集中选择答案，且可以添加自定义规则干预排序函数较为灵活；缺点是无法应对自然语言的多变性、语境解构、连贯性等，对语义的细微差别也无法识别；
+ Generation-based Systems：将对话视为input-output mapping问题，提出了MT-based方法（SMT统计机器翻译、IBM-model、phrase-based MT等），这种方法复杂且无法很好的解决输入输出的对应关系（尤其是当句子较复杂的时候，只适合单词级别）。但是最近的NN、seq-to-seq等方法很好地解决了这些问题，可以生成更加丰富、有意义、特别的对话响应。但是还存在许多问题，比如沉闷的回应、agent没有一个固定的风格、多轮对话等等
+ Frame-based Dialogue Systems：定义一个对话的框架，及其中所涉及的重要元素。优点是目标明确框架对对话指导意义明显，适用于飞机票、餐馆等预定领域。缺点是框架设计需要人工成本，且无法迁移到别的领域，并未涉及到人类语言的理解层面。
+ Finite-State Machine Systems有限状态机：（用户使用预定义的模板提问，系统之响应能力范围之内的问题），这种方法的缺点是完全依赖于对框架slot的填充，而无法决定对话的进程和状态（用户接受建议、拒绝等）
+ State-based Systems：主要包含系统状态（上下文信息、用户意图、对话进程等）和系统行动两（基于state采取action）个部分。MDP、POMDP等模型。
+ Question-Answering (QA) Based Dialogue Systems：factoid QA-based，个人助手，需要回答各种各样的问题并且进行交互式对话。目前的研究点主要包括，bot如何通过对话进行自学习、对于out-of-vocab的词汇应该学会问，即学会与人交流、如何通过在线反馈学习（犯错时调整、正确时加强）
