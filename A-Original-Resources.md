- 重现DeepMind星际争霸强化学习算法
  - https://zhuanlan.zhihu.com/p/29246185
- 迈向通用人工智能：星际争霸2人工智能研究环境SC2LE完全入门指南
  - https://zhuanlan.zhihu.com/p/28434323

---



- 概要：NIPS 2017 Deep Learning for Robotics Pieter Abbeel
  - https://zhuanlan.zhihu.com/p/32089849
  - Meta Learning Shared Hierarchies
  - Imitation Learning
    - One Shot Imitation Learning
      - 这篇论文提出一个比较通用的模仿学习的方法。这个方法在运行时，需要一个完成当前任务的完整演示，和当前状态。假设我要机器人搭方块，那么我给它一个完整的把方块搭好的视频演示，再告诉他当前方块都在哪里。这个模型会用CNN和RNN来处理任务的演示，这样，它就有一个压缩过的演示纲要。模型再用CNN处理当前状态，得到一个压缩过的当前状态信息。利用Attention Model来扫描演示纲要，我们就得到了“与当前状态最有关的演示的步骤”，再将这些信息全部传递给一个决策器。然后输出决策。具体的模型有很多细节，但大致流程如下
- Alpha
  - AlphaZero实战：从零学下五子棋（附代码）
    - https://zhuanlan.zhihu.com/p/32089487
    - https://github.com/junxiaosong/AlphaZero_Gomoku
  - AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
  - AlphaGo Zero: Mastering the game of Go without human knowledge

---

- 纯干货18 - 2016-2017深度学习-最新-必读-经典论文
  - https://zhuanlan.zhihu.com/p/32287815
  - 问答系统
    - IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
      在现代信息检索领域一直是两大学派之争的局面。一方面，经典思维流派是假设在文档和信息需求（由查询可知）之间存在着一个独立的随机生成过程。另一方面，现代思维流派则充分利用机器学习的优势，将文档和搜索词联合考虑为特征，并从大量训练数据中预测其相关性或排序顺序标签。
      本篇 SIGIR2017 的满分论文则首次提出将两方面流派的数据模型通过一种对抗训练的方式统一在一起，使得两方面的模型能够相互提高，最终使得检索到的文档更加精准。文章的实验分别在网络搜索、推荐系统以及问答系统三个应用场景中实现并验证了结果的有效性。
  - 代码生成
    -     DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning 25 apr 2017
          A Syntactic Neural Model for General-Purpose Code Generation 6 apr 2017
          RobustFill: Neural Program Learning under Noisy I/O 21 mar 2017
          DeepFix: Fixing Common C Language Errors by Deep Learning 12 feb 2017
          DeepCoder: Learning to Write Programs 7 nov 2016
          Neuro-Symbolic Program Synthesis 6 nov 2016
          Deep API Learning 27 may 2016
      

---

- Attention Mechanism
  - https://zhuanlan.zhihu.com/p/31547842
- (*)Meta Learning
  - https://zhuanlan.zhihu.com/p/32270990

---

- 【2018年深度学习十大警告性预测：DL硬件企业面临倒闭、元学习取代SGD、生成模型成主流、自实践自动知识构建、直觉机弥合语义鸿沟、可解释性仍不可及、缺少理论深度的DL论文继续井喷、打造学习环境以实现产业化、会话认知、AI伦理运用】《10 Alarming Predictions for Deep Learning in 2018》by Carlos E. Perez O网页链接
- 《Learning More Universal Representations for Transfer-Learning》Y Tamaazousti, H L Borgne, C Hudelot, M E A Seddik, M Tamaazousti [CEA & University of Paris-Saclay] (2017) O网页链接 
  - Universal Representations
  - Transfer-Learning
- 《Letter-Based Speech Recognition with Gated ConvNets》V Liptchinsky, G Synnaeve, R Collobert [Facebook AI Research] (2017) O网页链接 GitHub: https:\//github.com\/facebookresearch/wav2letter 
  - Letter-Based Speech Recognition 字符表示
  - Gated ConvNets

- 【Xi(Peter) Chen主讲的深度增强学习课程(Videos/Slides)】“Deep Reinforcement Learning” by Xi(Peter) Chen OpenAI/Berkeley AI Research Lab(http://t.cn/RHWQreD) via@数急 

---

【AI与深度学习2017年度综述】《AI and Deep Learning in 2017 – A Year in Review | WildML》by Denny Britz O网页链接 

【20行Python代码实现马尔科夫链生成鸡汤文字】《How I generated inspirational quotes with less than 20 lines of python code》by Ramtin Alami O网页链接 pdf:O网页链接 

“实用脚本：Ubuntu 16上全自动安装Nvidia驱动程序, Anaconda, CUDA, fastai等” O网页链接 

---



- Face Recognition
  #世界上最简单的人脸识别库
  本项目号称世界上最简单的人脸识别库，可使用 Python 和命令行进行调用。该库使用 dlib 顶尖的深度学习人脸识别技术构建，在户外脸部检测数据库基准（Labeled Faces in the Wild benchmark）上的准确率高达 99.38%。
  项目链接：https://github.com/ageitgey/face_recognition
  MUSE
  #多语言词向量 Python 库
  由 Facebook 开源的多语言词向量 Python 库，提供了基于 fastText 实现的多语言词向量和大规模高质量的双语词典，包括无监督和有监督两种。其中有监督方法使用双语词典或相同的字符串，无监督的方法不使用任何并行数据。
  无监督方法具体可参考 Word Translation without Parallel Data 这篇论文。
  论文链接：https://www.paperweekly.site/papers/1097
  项目链接：https://github.com/facebookresearch/MUSE
  FoolNLTK
  #中文处理工具包
  本项目特点：
  • 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
  • 基于 BiLSTM 模型训练而成
  • 包含分词，词性标注，实体识别，都有比较高的准确率
  • 用户自定义词典
  项目链接：https://github.com/rockyzhengwu/FoolNLTK
  Arnold
  #最擅长玩《毁灭战士》的游戏AI
  本项目来自卡耐基梅隆大学，是 2017 年 VizDoom《毁灭战士》AI 死亡竞赛冠军 Arnold 的 PyTorch 开源代码。
  论文链接：https://www.paperweekly.site/papers/1440
  项目链接：https://github.com/glample/Arnold
  Bottom-Up Attention VQA
  #2017 VQA Challenge 第一名
  本项目是 2017 VQA Challenge 第一名团队两篇论文的 PyTorch 复现。
  ■ 论文 | Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
  ■ 链接 | https://www.paperweekly.site/papers/754
  ■ 论文 | Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge
  ■ 链接 | https://www.paperweekly.site/papers/1441
  报告解读：2017 VQA Challenge 第一名技术报告
  项目链接：https://github.com/hengyuan-hu/bottom-up-attention-vqa
  YOLOv2 - PyTorch
  #PyTorch 版 YOLOv2
  著名物体检测库 YOLOv2 的 PyTorch 版本，本项目还可以将训练好的 model 转换为适配 Caffe 2。
  项目链接：https://github.com/ruiminshen/yolo2-pytorch
  Simple Railway Captcha Solver
  #基于 CNN 的台铁订票验证码辨识
  本项目利用简单的 Convolutional Neural Network 来实作辨识台铁订票网站的验证码，训练集的部分以模仿验证码样式的方式来产生、另外验证集的部分则自台铁订票网站撷取，再以手动方式标记约 1000 笔。
  目前验证集对于 6 码型态的验证码的单码辨识率达到 98.84%，整体辨识成功率达到 91.13%。
  项目链接：https://github.com/JasonLiTW/simple-railway-captcha-solver
  AlphaZero-Gomoku
  #用 AlphaZero 下五子棋
  这是一个将 AlphaZero 算法应用在五子棋的实现，由于五子棋相比围棋或国际象棋简单得多，所以只需几个小时就可以训练出一个不错的 AI 模型。
  ■ 论文 | AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
  ■ 链接 | https://www.paperweekly.site/papers/1297
  ■ 论文 | AlphaGo Zero: Mastering the game of Go without human knowledge
  ■ 链接 | https://www.paperweekly.site/papers/942
  项目链接：https://github.com/junxiaosong/AlphaZero_Gomoku
  gym-extensions
  #OpenAI Gym 扩展集
  这是一个 OpenAI Gym 库的扩展包，实现了包括：多任务学习、迁移学习、逆增强学习等功能。
  项目链接：https://github.com/Breakend/gym-extensions
  Myia
  #Python 深度学习框架
  Myia 是一个全新的 Python 深度学习框架，具有使用简单、自动微分和性能优化的特点。
  项目链接：https://github.com/mila-udem/myia
  

---



- QA && KB Repository
  - 怎么利用知识图谱构建智能问答系统？
    - https://www.zhihu.com/question/30789770/answer/116138035
  - https://zhuanlan.zhihu.com/p/25735572
  - 揭开知识库问答KB-QA的面纱1·简介篇
    - 什么是知识库（knowledge base, KB）
    - 什么是知识库问答（knowledge base question answering, KB-QA）
    - 知识库问答的主流方法
    - 知识库问答的数据集
- Deep Learning Practice and Trends
- 三分熟博士生の阅读理解与问答数据集 | 论文集精选 #03
  - https://zhuanlan.zhihu.com/p/30308726
- Distribution RL 
  - https://mtomassoli.github.io/2017/12/08/distributional_rl/
- 【深度学习迁移学习简介】《A Gentle Introduction to Transfer Learning for Deep Learning | Machine Learning Mastery》by Jason Brownlee
- 【(Python)多种模型(Naive Bayes, SVM, CNN, LSTM, etc)实现推文情感分析】’Sentiment analysis on tweets using Naive Bayes, SVM, CNN, LSTM, etc.' by Abdul Fatir GitHub: O网页链接 
- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
  @zxEECS 推荐
  #Natural Language Generation
  本文来自 Facebook AI Research。本文研究监督句子嵌入，作者研究并对比了几类常见的网络架构（LSTM，GRU，BiLSTM，BiLSTM with self attention 和 Hierachical CNN）, 5 类架构具很强的代表性。
  论文链接：https://www.paperweekly.site/papers/1332**
  代码链接：https://github.com/facebookresearch/InferSent**
- Structural Deep Network Embedding
  @YFLu 推荐
  #Representation Learning
  SDNE 是清华大学崔鹏老师组发表在 2016KDD 上的一个工作，目前谷歌学术引用量已经达到了 85，是一篇基于深度模型对网络进行嵌入的方法。
  SDNE 模型同时利用一阶相似性和二阶相似性学习网络的结构，一阶相似性用作有监督的信息，保留网络的局部结构；二阶相似性用作无监督部分，捕获网络的全局结构，是一种半监督深度模型。
  论文链接：https://www.paperweekly.site/papers/1142**
  代码链接：https://github.com/xiaohan2012/sdne-keras
  - 《Structural Deep Network Embedding》阅读笔记
    - https://zhuanlan.zhihu.com/p/24769965?refer=c_51425207
- Multilingual Hierarchical Attention Networks for Document Classification
  本文使用两个神经网络分别建模句子和文档，采用一种自下向上的基于向量的文本表示模型。	首先使用 CNN/LSTM 来建模句子表示，接下来使用双向 GRU 模型对句子表示进行编码得到文档表示。
  论文链接：https://www.paperweekly.site/papers/1152
  代码链接：https://github.com/idiap/mhan
  

---

  

- Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题](https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html)
  - https://github.com/deepmind/narrativeqa
  - 
  - DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。
- 没能去参加NIPS 2017？这里有一份最详细的现场笔记（附PDF）
- 视频：Pieter Abbeel NIPS 2017大会报告 《Deep Learning for Robots》（附PDF）
  - 12月6日下午，加州大学伯克利分校教授、机器人与强化学习领域的大牛Pieter Abbeel在NIPS 2017的报告视频。
- Uber 论文5连发宣告神经演化新时代，深度强化学习训练胜过 SGD 和策略梯度
  - 介绍了他们在基因算法（genetic algorithm）、突变方法（mutation）和进化策略（evolution strategies）等神经演化思路方面的研究成果，同时也理论结合实验证明了神经演化可以取代 SGD 等现有主流方法用来训练深度强化学习模型，同时取得更好的表现
  - http://www.gzhphb.com/article/104/1046105.html
  - Jeff Dean 
    - 神经演化是一个非常有潜力的研究方向
    - 另一个方向是稀疏激活的网络
  - 在深度学习领域，大家已经习惯了用随机梯度下降 SGD 来训练上百层的、包含几百万个连接的深度神经网络。虽然一开始没能严格地证明 SGD 可以让非凸函数收敛，但许多人都认为 SGD 能够高效地训练神经网络的重要原因是它计算梯度的效率很高
  - 借助新开发出的技术，Uber AI 的研究人员已经可以让深度神经网络高效地进化。同时他们也惊讶地发现，一个非常简单的基因算法（genetic algorithm）就可以训练带有超过四百万个参数的卷积网络，让它能够直接看着游戏画面玩 Atari 游戏；这个网络可以在许多游戏里取得比现代深度强化学习算法（比如 DQN 和 A3C）或者进化策略（evolution strategies）更好的表现，同时由于算法有更强的并行能力，还可以运行得比这些常见方法更快
  - Uber AI 的研究人员们进一步的研究表明，现代的一些基因算法改进方案，比如新颖性搜索算法（novelty search）不仅在基因算法的效果基础上得到提升，也可以在大规模深度神经网络上工作，甚至还可以改进探索效果、对抗带有欺骗性的问题（带有有挑战性的局部极小值的问题）；Q-learning（DQN）、策略梯度（A3C）、进化策略、基因算法之类的基于反馈最大化思路的算法在这种状况下的表现并不理想
  - 基因算法可以在 Frostbite 游戏中玩到 10500 分；而 DQN、A3C 和进化策略的得分都不到 1000 分。
  -     五篇新论文简介
        <1>
        《Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning》
        深度神经进化：在强化学习中，基因算法是训练深度神经网络的有竞争力的替代方案
        论文地址：https://arxiv.org/abs/1712.06567 
        重点内容概要：
        用一个简单、传统、基于群落的基因算法 GA（genetic algorithm）就可以让深度神经网络进化，并且在有难度的强化学习任务中发挥良好表现。在 Atari 游戏中，基因算法的表现和进化策略 ES（evolution strategies）以及基于 Q-learning（DQN）和策略梯度的深度强化学习算法表现一样好。
        深度基因算法「Deep GA」可以成功让具有超过四百万个自由参数的网络进化，这也是有史以来用传统进化算法进化出的最大的神经网络。
        论文中展现出一个有意思的现象：如果想要优化模型表现，在某些情况下沿着梯度走并不是最佳选择
        新颖性搜索算法（Novelty Search）是一种探索算法，它适合处理反馈函数带有欺骗性、或者反馈函数稀疏的情况。把它和深度神经网络结合起来，就可以解决一般的反馈最大化算法（比如基因算法 GA 和进化策略 ES）无法起效的带有欺骗性的高维度问题。
        论文中也体现出，深度基因算法「Deep GA」具有比进化策略 ES、A3C、DQN 更好的并行性能，那么也就有比它们更快的运行速度。这也就带来了顶级的编码压缩能力，可以用几千个字节表示带有数百万个参数的深度神经网络。
        论文中还尝试了在 Atari 上做随机搜索实验。令人惊讶的是，在某些游戏中随机搜索的表现远远好于 DQN、A3C 和进化策略 ES，不过随机搜索的表现总还是不如基因算法 GA。
        
        <2>
        《Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients》
        通过输出梯度在深度神经网络和循环神经网络中安全地进行突变
        论文地址： https://arxiv.org/abs/1712.06563 
        重点内容概要：
        借助梯度的安全突变 SM-G（Safe mutations through gradients）可以大幅度提升大规模、深度、循环网络中的突变的效果，方法是测量某些特定的连接权重发生改变时网络的敏感程度如何。
        计算输出关于权重的梯度，而不是像传统深度学习那样计算训练误差或者损失函数的梯度，这可以让随机的更新步骤也变得安全、带有探索性。
        以上两种安全突变的过程都不要增加新的尝试或者推演过程。
        实验结果：深度神经网络（超过 100 层）和大规模循环神经网络只通过借助梯度的安全突变 SM-G 的变体就可以高效地进化。
        
        <3>
        《On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent》
        对 OpenAI 的进化策略和随机梯度下降之间的关系的讨论
        论文地址：https://arxiv.org/abs/1712.06564
        重点内容概要：
        在 MNIST 数据集上的不同测试条件下，把进化策略 ES 近似计算出的梯度和随机梯度下降 SGD 精确计算出的梯度进行对比，以此为基础讨论了进化策略 ES 和 SGD 之间的关系。
        开发了快速的代理方法，可以预测不同群落大小下进化策略 ES 的预期表现
        介绍并展示了多种不同的方法用于加速以及提高进化策略 ES 的表现。
        受限扰动的进化策略 ES 在并行化的基础设施上可以大幅运行速度。
        把为 SGD 设计的 mini-batch 这种使用惯例替换为专门设计的进化策略 ES 方法：无 mini-batch 的进化策略 ES，它可以改进对梯度的估计。这种做法中会在算法的每次迭代中，把整个训练 batch 的一个随机子集分配给进化策略 ES 群落中的每一个成员。这种专门为进化策略 ES 设计的方法在同等计算量下可以提高进化策略 ES 的准确度，而且学习曲线即便和 SGD 相比都要顺滑得多。
        在测试中，无 mini-batch 的进化策略 ES 达到了 99% 准确率，这是进化方法在这项有监督学习任务中取得的最好表现。
        以上种种结果都可以表明在强化学习任务中进化策略 ES 比 SGD 更有优势。与有监督学习任务相比，强化学习任务中与环境交互、试错得到的关于模型表现目标的梯度信息的信息量要更少，而这样的环境就更适合进化策略 ES。
        
        <4>
        《ES Is More Than Just a Traditional Finite Difference Approximator》
        进化策略远不止是一个传统的带来有限个结果的近似方法
        论文地址：https://arxiv.org/abs/1712.06568
        重点内容概要：
        提出了进化策略 ES 和传统产生有限个结果的方法的一个重大区别，即进化策略 ES 优化的是数个解决方案的最优分布（而不是单独一个最优解决方案）。
        得到了一个有意思的结果：进化策略 ES 找到的解决方案对参数扰动有很好的健壮性。比如，作者们通过仿人类步行实验体现出，进化策略 ES 找到的解决方案要比基因算法 GA 和信赖域策略优化 TRPO 找到的类似解决方案对参数扰动的健壮性强得多。
        另一个有意思的结果：进化策略 ES 在传统方法容易困在局部极小值的问题中往往会有很好的表现，反过来说也是。作者们通过几个例子展示出了进化策略 ES 和传统的跟随梯度的方法之间的不同特性。
        
        <5>
        《Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents》
        通过一个寻找新颖性的智能体群落，改进用于深度强化学习的进化策略的探索能力
        论文地址：https://arxiv.org/abs/1712.06560
        重点内容概要：
        对进化策略 ES 做了改进，让它可以更好地进行深度探索
        通过形成群落的探索智能体提高小尺度神经网络进化的探索的算法，尤其是新颖性搜索算法（novelty search）和质量多样性算法（quality diversity），可以和进化策略 ES 组合到一起，提高它在稀疏的或者欺骗性的深度强化学习任务中的表现，同时还能够保持同等的可拓展性。
        确认了组合之后得到的新算法新颖性搜索进化策略 NS-ES 和质量多样性进化策略 QD-ES 的变体 NSR-ES 可以避开进化策略 ES 会遇到的局部最优，并在多个不同的任务中取得更好的表现，包括从模拟机器人在欺骗性的陷阱附近走路，到玩高维的、输入图像输出动作的 Atari 游戏等多种任务。
        这一基于群落的探索算法新家庭现在已经加入了深度强化学习工具包。
        
---
## Super Repository

Papers:
-
+ NLP
	+ https://web.stanford.edu/~jurafsky/slp3/
+ DRL4NLP
	+ https://github.com/ganeshjawahar/drl4nlp.scratchpad
	```
	+++Policy Gradients
	buck_arxiv17: Ask the Right Questions: Active Question Reformulation with Reinforcement Learning [arXiv]
	dhingra_acl17: Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access [arXiv] [code]
    paulus_arxiv17: A Deep Reinforced Model for Abstractive Summarization [arXiv]
    nogueira_arxiv17: Task-Oriented Query Reformulation with Reinforcement Learning [arXiv] [code]
    li_iclr17: Dialog Learning with Human-in-the-loop [arXiv] [code]
    li_iclr17_2: Learning through dialogue interactions by asking questions [arXiv] [code]
    yogatama_iclr17: Learning to Compose Words into Sentences with Reinforcement Learning [arXiv] 
    dinu_nips16w: Reinforcement Learning for Transition-Based Mention Detection [arXiv]
    clark_emnlp16: Deep Reinforcement Learning for Mention-Ranking Coreference models [arXiv] [code]
    +++Value Function
    narasimhan_emnlp16: Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning [arXiv] [code]
    +++Misc
    bordes_iclr17: Learning End-to-End Goal-Oriented Dialog [arXiv]
    weston_nips16: Dialog-based Language Learning [arXiv] [code]
    nogueira_nips16: End-to-End Goal-Driven Web Navigation [arXiv] [code]
	```

+ Playing Atari with Deep Reinforcement Learning
	+ https://arxiv.org/pdf/1312.5602.pdf
+ Deep Reinforcement Learning with a Natural Language Action Space
	+ https://arxiv.org/pdf/1511.04636.pdf
 + Bidirectional LSTM-CRF Models for Sequence Tagging
	 + https://arxiv.org/abs/1508.01991
+ Hinton 2006 Deep Belief Network
	+ list:
		- A fast learning algorithm for deep belief nets. Neural Computation
		- Greedy Layer-Wise Training of Deep Networks
		- Efficient Learning of Sparse Representations with an Energy-Based Model
	- key point:
		- unsup learn for pre train
		- train by layers
		- sup learn for tuning weight between layers 

Office sites:
-
+ Kaggle
	+ https://www.kaggle.com/
+ pygame
	+ http://www.pygame.org/news
+ Deep Reinforcement Learning for Keras.
	+ http://keras-rl.readthedocs.io/
	+ https://github.com/matthiasplappert/keras-rl
+ ai-code
	+ http://www.ai-code.org/
+ DeepMind
	+ https://deepmind.com/
+ 库
	开始尝试机器学习库可以从安装最基础也是最重要的开始，像numpy和scipy。
	+ 查看和执行数据操作：pandas（http://pandas.pydata.org/）
	+ 对于各种机器学习模型：scikit-learn（http://scikit-learn.org/stable/）
	+ 最好的gradient boosting库：xgboost（https://github.com/dmlc/xgboost）
	+ 对于神经网络：keras（http://keras.io/）
	+ 数据绘图：matplotlib（http://matplotlib.org/）
	+ 监视进度：tqdm（https://pypi.python.org/pypi/tqdm） 

Videos :
--
+ 21 Deep Learning Videos, Tutorials & Courses on Youtube from 2016
	+ https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/
+ RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning
	+ https://www.youtube.com/playlist?list=PLV_1KI9mrSpGFoaxoL9BCZeen_s987Yxb
+ 斯坦福2017季CS224n深度学习自然语言处理课程
	+ https://www.bilibili.com/video/av13383754/
+ CS224d: Deep Learning for Natural Language Processing ( Doing )
	+ https://www.bilibili.com/video/av9143821/?from=search&seid=9547251413889295037
+ CS 294: Deep Reinforcement Learning, Fall 2017
	+ http://rll.berkeley.edu/deeprlcourse/#lecture-videos
+ Morvan
	+ https://github.com/MorvanZhou
+ 李宏毅深度学习(2017)
	 - https://www.bilibili.com/video/av9770302/

------

Github Projects:
--
+ Machine Learning Mindmap / Cheatsheet ( Only Pictures)
	+ https://github.com/dformoso/machine-learning-mindmap
	+ A Mindmap summarising Machine Learning concepts, from Data Analysis to Deep Learning.
+ DeepMind : Teaching Machines to Read and Comprehend
	+ https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend
	+ This repository contains an implementation of the two models (the Deep LSTM and the Attentive Reader) described in Teaching Machines to Read and Comprehend by Karl Moritz Hermann and al., NIPS, 2015. This repository also contains an implementation of a Deep Bidirectional LSTM.
	
+ A-Guide-to-DeepMinds-StarCraft-AI-Environment
	+ https://github.com/llSourcell/A-Guide-to-DeepMinds-StarCraft-AI-Environment
	+ This is the code for "A Guide to DeepMind's StarCraft AI Environment" by Siraj Raval on Youtube
    + **Must install in venv to avoid wrong operation** 
   
---
+ **Keras Text Classification Library**
	+ https://github.com/raghakot/keras-text
	+ keras-text is a one-stop text classification library implementing various state of the art models with a clean and extendable interface to implement custom architectures.
+ **emnlp **
	+ https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf


----------
RL
---
+ 【Keras+OpenAI增强学习实践：Actor-Critic模型】《Reinforcement Learning w/ Keras + OpenAI: Actor-Critic Models》by Yash Patel O网页链接 pdf:O网页链接 ​​​​
+ 《Keras+OpenAI强化学习实践：深度Q网络》via:机器之心 O教程 | Keras+OpenAI强化学习实践：深度Q网络
+ 《A Tour of Gotchas When Implementing Deep Q Networks with Keras and OpenAi Gym》by Scott Rome O网页链接 
+ 【层次增强学习算法】《Learning a Hierarchy | OpenAI》 O网页链接 ref:《Meta Learning Shared Hierarchies》(2017) GitHub: https ://github .com/openai/mlsh
+ 《OpenAI实习生提出分层强化学习新算法》via:机器之心 
	+ 
+ Q-Learning月球着陆控制“QLearning in OpenAI Lunar Lander” 
	+ GitHub: https:\//github.com\/FitMachineLearning/FitML/ 
+ 强化学习入门 第一讲 MDP
	+ https://zhuanlan.zhihu.com/p/25498081
+ 强化学习入门及其实现代码
		+ http://www.jianshu.com/p/165607eaa4f9
+ Course
	+ 如2015年David Silver的经典课程Teaching ，
	+ 2017年加州大学伯克利分校Levine, Finn, Schulman的课程 CS 294 Deep Reinforcement Learning, Spring 2017 
	+ 卡内基梅隆大学的2017 春季课程Deep RL and Control 。
+ 【转载】近似动态规划与强化学习入门步骤
	+ http://www.cnblogs.com/stevenbush/articles/3353227.html
+ Awesome Reinforcement Learning
	+ https://github.com/aikorea/awesome-rl
	+ lots of resourses
+ http://www.jianshu.com/p/165607eaa4f9
	+ http://www.jianshu.com/p/165607eaa4f9
+ keras-rl

Other Resources:
--
+ 模型汇总16 各类Seq2Seq模型对比及《Attention Is All You Need》中技术详解
	+ https://zhuanlan.zhihu.com/p/27485097
+	模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用
	+ https://zhuanlan.zhihu.com/p/31547842?utm_source=wechat_session&utm_medium=social 
+ 人工智能 Java 坦克机器人系列强化学习-IBM Robo code
	+ https://www.ibm.com/developerworks/cn/java/j-lo-robocode2/index.html
+ 遗传算法
	 + https://www.zhihu.com/question/23293449
+ 蒙特卡罗算法
	+ https://www.zhihu.com/question/20254139
 + RL
	 + 深度强化学习（Deep Reinforcement Learning）入门：RL base & DQN-DDPG-A3C introduction
		 + https://zhuanlan.zhihu.com/p/25239682
	 + 深度增强学习前沿算法思想【DQN、A3C、UNREAL简介】
		 + http://blog.csdn.net/mmc2015/article/details/55271605
	 + Google Deepmind大神David Silver带你认识强化学习
		 + https://www.leiphone.com/news/201608/If3hZy8sLqbn6uvo.html
		 + http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf
 + NLP
	  + 综述 | 一文读懂自然语言处理NLP（附学习资料）
		 + http://www.pinlue.com/article/2017/11/1413/074849597604.html
 + DL

	   + 126 篇殿堂级深度学习论文分类整理 从入门到应用 | 干货9
		   + https://www.leiphone.com/news/201702/FWkJ2AdpyQRft3vW.html?utm_source=tuicool&utm_medium=referral
	   +  DeepLearningBook读书笔记
		   + https://github.com/exacity/simplified-deeplearning/blob/master/README.md 
 + ML
	 + 良心GitHub项目：各种机器学习任务的顶级结果（论文）汇总
		 + https://www.ctolib.com/topics-126416.html
 + Transfor Learning
	 + 14 篇论文为你呈现「迁移学习」研究全貌
		 + https://www.ctolib.com/topics-125968.html
 + SKLearn(工程用用的较多的模块介绍)
	 + http://blog.csdn.net/column/details/scikitlearninaction.html
 + Tensorflow 
	  + （较好）转TensorFlow实现案例汇集：代码+笔记
		  + https://zhuanlan.zhihu.com/p/29128378 
	  + 数十种TensorFlow实现案例汇集：代码+笔记
		  + http://dy.163.com/v2/article/detail/C3J6JU2U0511AQHO.html
	  + 【推荐】TensorFlow/PyTorch/Sklearn实现的五十种机器学习模型
		  +  https://mp.weixin.qq.com/s/HufdD3OSJIK2yAexM-Wb5w
 + Others
	 + cs231n课程笔记翻译
		 + http://www.cnblogs.com/xialuobo/p/5867314.html
	 + 全网AI和机器学习资源大合集（研究机构、视频、博客、书籍...）
		 + http://www.sohu.com/a/164766699_468650
	 
+ NIPS大会最精彩一日：AlphaZero遭受质疑；史上第一场正式辩论与LeCun激情抗辩；元学习&强化学习亮点复盘
	+ https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650734461&idx=1&sn=154e7ff280626bbd6feda4e5607eecc4&chksm=871b3b03b06cb215b3d1d85a08306fa232311d585788d1024ded9305a99dbe4104598df05e29&mpshare=1&scene=1&srcid=1209sY3iZJvGZUUoHQDqWNrA&pass_ticket=VPRBJUnIlp%2BYQtpx6zRWEjZE9o39jBz2mDq5fAz7NkU2RxaP%2BuJhsCR4DDwVHJbm#rd
+ 自然语言顶级会议ACL 2016谷歌论文汇集
	+ https://www.jiqizhixin.com/articles/2016-08-08-7
+ 解决机器学习问题有通法
	+ https://www.jiqizhixin.com/articles/2017-09-21-10
+ 比AlphaGo Zero更强的AlphaZero来了！8小时解决一切棋类！
	+ https://arxiv.org/pdf/1712.01815.pdf
	+ https://www.reddit.com/r/chess/comments/7hvbaz/mastering_chess_and_shogi_by_selfplay_with_a/
	+ https://zhuanlan.zhihu.com/p/31749249

---
	
Corpus:
-
+ 国家语委
+ 国内可用免费语料库
	+ http://www.cnblogs.com/mo-wang/p/4444858.html
+ 总词汇量在7400W+，可以用于训练很多模型例如Max Entropy、CRF、HMM......，优点是这份语料在分词基础上还做了词性标注，至于准确性还有待考究。
https://github.com/liwenzhu/corpusZh



    
