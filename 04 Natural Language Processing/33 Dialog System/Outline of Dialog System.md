# View of Dialog System

### Reference
+ https://zhuanlan.zhihu.com/p/31828371
### Outline
+ Chit-Chat-oriented Dialogue Systems： 闲聊型对话机器人，产生有意义且丰富的响应。
+ Rule-based system：对话经过预定义的规则（关键词、if-else、机器学习方法等）处理，然后执行相应的操作，产生回复。（ELIZA系统，如果输入语句中没有发现预定义规则，则生成generic的响应）。缺点是规则的定义，系统越复杂规则也越多，而且其无法理解人类语言，也无法生成有意义的自然语言对话。处在比较浅层的阶段；
+ IR-based Systems：信息检索或者最近邻方法，要求生成的响应与对话存在语义相关性（VSM、TF-IDF、page-Rank、推荐等排序方法）。有点是比生成模型简单，直接从训练集中选择答案，且可以添加自定义规则干预排序函数较为灵活；缺点是无法应对自然语言的多变性、语境解构、连贯性等，对语义的细微差别也无法识别；
+ Generation-based Systems：将对话视为input-output mapping问题，提出了MT-based方法（SMT统计机器翻译、IBM-model、phrase-based MT等），这种方法复杂且无法很好的解决输入输出的对应关系（尤其是当句子较复杂的时候，只适合单词级别）。但是最近的NN、seq-to-seq等方法很好地解决了这些问题，可以生成更加丰富、有意义、特别的对话响应。但是还存在许多问题，比如沉闷的回应、agent没有一个固定的风格、多轮对话等等
+ Frame-based Dialogue Systems：定义一个对话的框架，及其中所涉及的重要元素。优点是目标明确框架对对话指导意义明显，适用于飞机票、餐馆等预定领域。缺点是框架设计需要人工成本，且无法迁移到别的领域，并未涉及到人类语言的理解层面。
+ Finite-State Machine Systems有限状态机：（用户使用预定义的模板提问，系统之响应能力范围之内的问题），这种方法的缺点是完全依赖于对框架slot的填充，而无法决定对话的进程和状态（用户接受建议、拒绝等）
+ State-based Systems：主要包含系统状态（上下文信息、用户意图、对话进程等）和系统行动两（基于state采取action）个部分。MDP、POMDP等模型。
+ Question-Answering (QA) Based Dialogue Systems：factoid QA-based，个人助手，需要回答各种各样的问题并且进行交互式对话。目前的研究点主要包括，bot如何通过对话进行自学习、对于out-of-vocab的词汇应该学会问，即学会与人交流、如何通过在线反馈学习（犯错时调整、正确时加强）