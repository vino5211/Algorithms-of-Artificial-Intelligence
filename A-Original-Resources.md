+ 贝叶斯机器学习前沿进展
	+ http://chuansong.me/n/2152434851911
- Imitation Learning
	- One Shot Imitation Learning
   	- 这篇论文提出一个比较通用的模仿学习的方法。这个方法在运行时，需要一个完成当前任务的完整演示，和当前状态。假设我要机器人搭方块，那么我给它一个完整的把方块搭好的视频演示，再告诉他当前方块都在哪里。这个模型会用CNN和RNN来处理任务的演示，这样，它就有一个压缩过的演示纲要。模型再用CNN处理当前状态，得到一个压缩过的当前状态信息。利用Attention Model来扫描演示纲要，我们就得到了“与当前状态最有关的演示的步骤”，再将这些信息全部传递给一个决策器。然后输出决策。
    - (不确定)https://github.com/tianheyu927/mil
- 概要：NIPS 2017 Deep Learning for Robotics Pieter Abbeel
  - https://zhuanlan.zhihu.com/p/32089849
- Meta Learning Shared Hierarchies
- 代码生成
	- DeepAM: Migrate APIs with Multi-modal Sequence to Sequence Learning 25 apr 2017
      A Syntactic Neural Model for General-Purpose Code Generation 6 apr 2017
      RobustFill: Neural Program Learning under Noisy I/O 21 mar 2017
      DeepFix: Fixing Common C Language Errors by Deep Learning 12 feb 2017
      DeepCoder: Learning to Write Programs 7 nov 2016
      Neuro-Symbolic Program Synthesis 6 nov 2016
      Deep API Learning 27 may 2016
---
- Distribution RL
  - https://mtomassoli.github.io/2017/12/08/distributional-r1/
- 深度学习迁移学习简介
	- A Gentle Introduction to Transfer Learning for Deep Learning | Machine Learning Mastery by Jason Brownlee
- 【(Python)多种模型(Naive Bayes, SVM, CNN, LSTM, etc)实现推文情感分析】’Sentiment analysis on tweets using Naive Bayes, SVM, CNN, LSTM, etc.'
- 【AI与深度学习2017年度综述】《AI and Deep Learning in 2017 – A Year in Review | WildML》by Denny Britz
- “实用脚本：Ubuntu 16上全自动安装Nvidia驱动程序, Anaconda, CUDA, fastai等” 
---

- 【2018年深度学习十大警告性预测：DL硬件企业面临倒闭、元学习取代SGD、生成模型成主流、自实践自动知识构建、直觉机弥合语义鸿沟、可解释性仍不可及、缺少理论深度的DL论文继续井喷、打造学习环境以实现产业化、会话认知、AI伦理运用】《10 Alarming Predictions for Deep Learning in 2018》by Carlos E. Perez O网页链接
- 《Learning More Universal Representations for Transfer-Learning》Y Tamaazousti, H L Borgne, C Hudelot, M E A Seddik, M Tamaazousti [CEA & University of Paris-Saclay] (2017) O网页链接 
  - Universal Representations
  - Transfer-Learning
- 《Letter-Based Speech Recognition with Gated ConvNets》V Liptchinsky, G Synnaeve, R Collobert [Facebook AI Research] (2017) O网页链接 GitHub: https:\//github.com\/facebookresearch/wav2letter 
  - Letter-Based Speech Recognition 字符表示
  - Gated ConvNets

---

- 没能去参加NIPS 2017？这里有一份最详细的现场笔记（附PDF）
- 视频：Pieter Abbeel NIPS 2017大会报告 《Deep Learning for Robots》（附PDF）
  - 12月6日下午，加州大学伯克利分校教授、机器人与强化学习领域的大牛Pieter Abbeel在NIPS 2017的报告视频。

---

Super Repository

Papers:


- NLP
  - https://web.stanford.edu/~jurafsky/slp3/
- DRL4NLP

- Playing Atari with Deep Reinforcement Learning
  - https://arxiv.org/pdf/1312.5602.pdf
- Deep Reinforcement Learning with a Natural Language Action Space
  - https://arxiv.org/pdf/1511.04636.pdf
- Bidirectional LSTM-CRF Models for Sequence Tagging
  - https://arxiv.org/abs/1508.01991
- Hinton 2006 Deep Belief Network
  - list:
    - A fast learning algorithm for deep belief nets. Neural Computation
    - Greedy Layer-Wise Training of Deep Networks
    - Efficient Learning of Sparse Representations with an Energy-Based Model
  - key point:
    - unsup learn for pre train
    - train by layers
    - sup learn for tuning weight between layers

Office sites:
- pygame
  - http://www.pygame.org/news
- Deep Reinforcement Learning for Keras.
  - http://keras-rl.readthedocs.io/
  - https://github.com/matthiasplappert/keras-rl
- ai-code
  - http://www.ai-code.org/
- DeepMind
  - https://deepmind.com/
- 库
  开始尝试机器学习库可以从安装最基础也是最重要的开始，像numpy和scipy。
  - 查看和执行数据操作：pandas（http://pandas.pydata.org/）
  - 对于各种机器学习模型：scikit-learn（http://scikit-learn.org/stable/）
  - 最好的gradient boosting库：xgboost（https://github.com/dmlc/xgboost）
  - 对于神经网络：keras（http://keras.io/）
  - 数据绘图：matplotlib（http://matplotlib.org/）
  - 监视进度：tqdm（https://pypi.python.org/pypi/tqdm） 

Videos :

- 21 Deep Learning Videos, Tutorials & Courses on Youtube from 2016
  - https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/
- RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning
  - https://www.youtube.com/playlist?list=PLV_1KI9mrSpGFoaxoL9BCZeen_s987Yxb
- 斯坦福2017季CS224n深度学习自然语言处理课程
  - https://www.bilibili.com/video/av13383754/
- CS224d: Deep Learning for Natural Language Processing ( Doing )
  - https://www.bilibili.com/video/av9143821/?from=search&seid=9547251413889295037
- CS 294: Deep Reinforcement Learning, Fall 2017
  - http://rll.berkeley.edu/deeprlcourse/#lecture-videos
- Morvan
  - https://github.com/MorvanZhou
- 李宏毅深度学习(2017)
  - https://www.bilibili.com/video/av9770302/

---

Github Projects:

- Machine Learning Mindmap / Cheatsheet ( Only Pictures)
  - https://github.com/dformoso/machine-learning-mindmap
  - A Mindmap summarising Machine Learning concepts, from Data Analysis to Deep Learning.
- DeepMind : Teaching Machines to Read and Comprehend
  - https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend
  - This repository contains an implementation of the two models (the Deep LSTM and the Attentive Reader) described in Teaching Machines to Read and Comprehend by Karl Moritz Hermann and al., NIPS, 2015. This repository also contains an implementation of a Deep Bidirectional LSTM.
- A-Guide-to-DeepMinds-StarCraft-AI-Environment
  - https://github.com/llSourcell/A-Guide-to-DeepMinds-StarCraft-AI-Environment
  - This is the code for "A Guide to DeepMind's StarCraft AI Environment" by Siraj Raval on Youtube
    - Must install in venv to avoid wrong operation 


---

Other Resources:

- 模型汇总16 各类Seq2Seq模型对比及《Attention Is All You Need》中技术详解
  - https://zhuanlan.zhihu.com/p/27485097
    +模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用
  - https://zhuanlan.zhihu.com/p/31547842?utm_source=wechat_session&utm_medium=social 
- 人工智能 Java 坦克机器人系列强化学习-IBM Robo code
  - https://www.ibm.com/developerworks/cn/java/j-lo-robocode2/index.html
- 遗传算法
  - https://www.zhihu.com/question/23293449
- 蒙特卡罗算法
  - https://www.zhihu.com/question/20254139
- RL
  - 深度强化学习（Deep Reinforcement Learning）入门：RL base & DQN-DDPG-A3C introduction
    - https://zhuanlan.zhihu.com/p/25239682
  - 深度增强学习前沿算法思想【DQN、A3C、UNREAL简介】
    - http://blog.csdn.net/mmc2015/article/details/55271605
  - Google Deepmind大神David Silver带你认识强化学习
    - https://www.leiphone.com/news/201608/If3hZy8sLqbn6uvo.html
    - http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf
- NLP
  - 综述 | 一文读懂自然语言处理NLP（附学习资料）
    - http://www.pinlue.com/article/2017/11/1413/074849597604.html
- DL
  - 126 篇殿堂级深度学习论文分类整理 从入门到应用 | 干货9
    - https://www.leiphone.com/news/201702/FWkJ2AdpyQRft3vW.html?utm_source=tuicool&utm_medium=referral
  - DeepLearningBook读书笔记
    - https://github.com/exacity/simplified-deeplearning/blob/master/README.md 
- ML
  - 良心GitHub项目：各种机器学习任务的顶级结果（论文）汇总
    - https://www.ctolib.com/topics-126416.html
- Transfor Learning
  - 14 篇论文为你呈现「迁移学习」研究全貌
    - https://www.ctolib.com/topics-125968.html
- SKLearn(工程用用的较多的模块介绍)
  - http://blog.csdn.net/column/details/scikitlearninaction.html
- Tensorflow 
  - （较好）转TensorFlow实现案例汇集：代码+笔记
    - https://zhuanlan.zhihu.com/p/29128378 
  - 数十种TensorFlow实现案例汇集：代码+笔记
    - http://dy.163.com/v2/article/detail/C3J6JU2U0511AQHO.html
  - 【推荐】TensorFlow/PyTorch/Sklearn实现的五十种机器学习模型
    - https://mp.weixin.qq.com/s/HufdD3OSJIK2yAexM-Wb5w
- Others
  - cs231n课程笔记翻译
    - http://www.cnblogs.com/xialuobo/p/5867314.html
  - 全网AI和机器学习资源大合集（研究机构、视频、博客、书籍...）
    - http://www.sohu.com/a/164766699_468650
- NIPS大会最精彩一日：AlphaZero遭受质疑；史上第一场正式辩论与LeCun激情抗辩；元学习&强化学习亮点复盘
  - https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650734461&idx=1&sn=154e7ff280626bbd6feda4e5607eecc4&chksm=871b3b03b06cb215b3d1d85a08306fa232311d585788d1024ded9305a99dbe4104598df05e29&mpshare=1&scene=1&srcid=1209sY3iZJvGZUUoHQDqWNrA&pass_ticket=VPRBJUnIlp%2BYQtpx6zRWEjZE9o39jBz2mDq5fAz7NkU2RxaP%2BuJhsCR4DDwVHJbm#rd
- 自然语言顶级会议ACL 2016谷歌论文汇集
  - https://www.jiqizhixin.com/articles/2016-08-08-7
- 解决机器学习问题有通法
  - https://www.jiqizhixin.com/articles/2017-09-21-10
- 比AlphaGo Zero更强的AlphaZero来了！8小时解决一切棋类！
  - https://arxiv.org/pdf/1712.01815.pdf
  - https://www.reddit.com/r/chess/comments/7hvbaz/mastering_chess_and_shogi_by_selfplay_with_a/
  - https://zhuanlan.zhihu.com/p/31749249
