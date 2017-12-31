Ask the Right Questions : Active Question Reformulation with Reinforcement Learning(用强化学习进行 主动问题重新生成 )

- Abstract
  - an approach that call Active Question Answering
  - We propose an agent that sits between the user and a black box question-answering system an which learns to reformulate questions to elicit the best possible answers.
  - (代理探测系统，可能有许多自然语言对最初的问题进行修改，并将返回的证据汇总到产生最好的答案。)
  - The reformulation system is trained end-to-end to maximize answer quality using policy gradient(梯度策略)
  - evaluate on SearchQA
       https://github.com/nyu-dl/SearchQA
       Here are the json files that are collecting using the Jeopardy! dataset:
       https://drive.google.com/open?id=0B51lBZ1gs1XTMVFsQlNEQUtpWVU
      
  - Results : Our agent improves F1 by 11% over a state-of-the-art base model that uses the original question/answer pairs
- Introduction
  - Web and social media have become primary sources of information. Users’ expectations and information seeking activities co-evolve(伴随) with the increasing sophistication（越来越复杂） of these resources.
  - Target Beyond navigation, document retrieval, and simple factual question answering,users seek direct answers to complex and compositional questions
  - Such search sessions may require multiple iterations(迭代), critical assessment（关键评估） and synthesis（综合）
       G. Marchionini. Exploratory search: From finding to understanding. Commun. ACM, 49(4):41–46, 2006. 
      
  - The productivity（生产力） of natural language yields a myriad of ways to formulate（表述） a question + In the face of complex information needs, humans overcome uncertainty by (1)reformulating questions(重新表述问题), (2)issuing multiple searches（多重搜索）, and (3)aggregating responses（聚合响应）
  - Inspired by humans’ ability to ask the right questions, we present an agent that learns to carry out this process for the user. 
  - The agent aims to maximize the chance of getting the correct answer by reformulating and reissuing a user’s question to the environment.The agent comes up with a single best answer by asking many pertinent(有关) questions and aggregating the returned evidence.
  - The internals of the environment are not available to the agent, so it must learn to probe a black-box optimally using only natural language.（环境的内部不能提供给代理, 所以它必须学会使用只用一个黑盒自然语言。）
  - Our method resembles active learning [Settles, 2010]. In active learning, the learning algorithm chooses which instances to send to an environment for labeling, aiming to collect the most valuable information in order to improve model quality.（选择一个实例放入环境进行标记，为了找到最有价值的信息来提升模型质量）
  - Similarly, our agent aims to learn how to query an environment optimally, aiming to maximize the chance of revealing the correct answer to the user(同样, 我们代理旨在学习如何以最佳方式查询环境, 目的是最大限度地向用户展示正确答案的机会) + Due to this resemblance(相似性) we call our approach Active（主动） Question Answering (AQA).
  - AQA differs from standard active learning in that it searches in the space of natural language questions and selects the question that yields the most relevant response（选择产生最相关响应的问题）
  - Further, AQA aims to solve each problem instance (original question) via active reformulation, rather than selecting hard ones for labelling to improve its decision boundary.（此外, AQA 的目标是解决每个问题实例(原始问题) 通过积极的重新制定, 而不是选择硬的标签改善其决策边界。）
  - The key component of our proposed solution（解决方案）, see Figure 1, is a sequence-to-sequence model that is trained using （1）reinforcement learning (RL) with a reward based on the (2)answer given by the QA environment
  - The second component to AQA combines the evidence from interacting with the environment using a convolutional neural network. (AQA 的第二个组件结合了使用卷积神经网络与环境交互的证据。)  + We evaluate on a dataset of complex questions taken from Jeopardy!, the SearchQA dataset [Dunn et al., 2017]` M. Dunn, L. Sagun, M. Higgins, U. Guney, V. Cirik, and K. Cho. SearchQA: A New Q&A Dataset A
  - Thus SearchQA tests the ability of AQA to reformulate questions such that the QA system has the best chance of returning the correct answer. AQA outperforms(胜过) a deep network built for QA, BiDAF [Seo et al., 2017a], which has produced state-of-the-art results on multiple tasks, by 11% absolute F1, a 32% relative F1 improvement. We conclude by proposing AQA as a general framework for stateful, iterative information seeking tasks.
      BiDAF
      
- Active Question Answering
  - The Agent-Environment Framework
  - The major departure(偏离) from the standard MT setting is that our model reformulates utterances(重新表述) in the same language.
  - Unlike in MT, there is little high quality training data available for monolingual paraphrasing(单语释义)
  - We address this first by pre-training the model on a related task and second, by utilizing the end-to-end signalsproduced though the interaction with the QA environment(与QA环境的交互中产生的端到端信号)
  - In the downward pass in Figure 1 the reformulator transforms the original question into one or many alternative questions used to probe the environment for candidate answers.
  - Active Question Answering Agent
  - The reformulator is trained end-to-end, using an answer quality metric(答案质量度量) as the objective(目标)
  - This sequence-level loss is non-differentiable(序列级损失是不可微的), so the model is trained using Reinforcement Learning
  - In the upward pass in Figure 1, the aggregator selects the best answer. For this we use an additional neural network. The aggregator’s task is to evaluate the candidate answers returned by the environment and select the one to return
  - Here, we assume that there is a single best answer, as is the case in our evaluation setting;returning multiple answers is a straightforward extension of the model. The aggregator is trained with supervised learning.
  - Question-Answering Environment
  - Finally, we require an environment to interact with. For this we use a competitive neural question answering model,BiDirectional Attention Flow (BiDAF)
       M. Seo, A. Kembhavi, A. Farhadi, and H. Hajishirzi. Bidirectional Attention Flow for Machine Comprehension. In Proceedings of ICLR, 2017a.
  - BiDAF is an extractive QA system. It takes as input a question and a document and returns as answer a continuous span(连续跨域) from the document
  - The model contains a bidirectional attention mechanism to score document snippets with respect to the question, implemented with multi-layer LSTMs and other components.
  - The environment is opaque, the agent has no access to its internals: parameters, activations,gradients, etc. AQA may only send questions to it, and receive answers.(这个场景使我们能够设计一个允许使用任何后端的通用框架。) 
  - However, it means that feedback on the quality of the question reformulations is noisy and indirect(然而，这意味着对问题重新制定的质量的反馈是嘈杂和间接的)
- 3 Training 
	- To train AQA we use a combination of reinforcement and supervised learning.
	- We also present a strategy to overcome data paucity(数据量少) in monolingual paraphrasing(单语分析)
	- 3.1 Question Answer Environment
		- We treat BiDAF [Seo et al., 2017a] as a static black box QA system 
		- We train the model on the training set for the QA task at hand(？？？)， see Section 4.5 for details.
		- Afterwards(然后),BiDAF becomes part of the environment and its parameters are not updated while training the AQA agent
		- In principle, we could train both the agent and the environment jointly to further improve performance. However, this is not our desired task: our aim is for the agent to learn to communicate using natural language with an environment over which is has no control(不受控制)
		- This setting generalizes（概括） to interaction（交互） with arbitrary(任意) information sources(这个设置概括了与任意信息源的交互)
	- 3.2 Policy Gradient() Training of the Reformulation Model
	    - For a given question q 0 , we want to return the best possible answer a, maximizing a reward R(a|q 0 )
	    - Typically, R is the token level F1 score on the answer.(???)
	    - The answer a = f (q) is an unknown function of a question q, computed by the environment
	    - Note, that the reward is computed with respect to the original question q 0 while the answer is produced using a question q.(???)
	    - Add
	    - Add
	    - Add

	- 3.3 Intialization of the Reformulation Model
    	- We pre-train the question reformulation model by building a paraphrasing Neural MT model, i.e. a model that can translate English to English.
    	- While parallel corpora(平行语料库) are available for many language pairs, English-English corpora are scarce(缺少的), so we cannot train monolingual model(单语的) directly
    	- Instead, we first produce a multilingual translation system that translates between several languages [Johnson et al., 2016].
    	- This allows us to use available bilingual corpora(双语语料).
    	- Multilingual training requires nothing more than adding two special tokens to the data indicating the source and target languages
    	- The encoder-decoder architecture of the translation model remains unchanged.
    	- As Johnson et al. [2016] show, this model can be used for zero-shot translation(无显式的语言对), i.e. to translate between language pairs for which it has seen no training examples
			- Zero shot translation
				- Zero-Shot翻译是指对不存在显式训练或者映射的语言对之间短语的翻译
				- http://www.infoq.com/cn/news/2017/02/zero-shot-translation
		- For example after training English-Spanish, English-French, French-English, and Spanish-English the model has learned a single encoder that encodes English, Spanish, and French and a decoder for the same three languages. Thus, we can use the same model for French-Spanish, Spanish-French and also English-English translation by adding the respective tokens, e.g. 
		- Johnson et al. [2016] note that zero-shot translation is generally worse than bridging, an approach that uses the model twice: first, to translate into a pivot language, and then into the target language.(经过一种中间语言中转)
		- However, the performance gap can be closed by running a few training steps for the desired language pair(期望的语料对). Thus, we first train on multilingual data（多语言）, then on a small corpus of monolingual data（单语言）.
	- 3.4 Answer Selection
		- With either beam search or sampling we can produce many rewrites of a single question from our reformulation system(产生许多重新构造的问题)
			- beam search
			- sampling
		- We issue each rewrite to the QA environment, yielding(产生) a set of (query, rewrite, answer) tuples from which we need to pick the best instance.
		- We train another neural network to pick the best answer from the candidates.
			- CNN
		- While this is a ranking problem, we frame it as binary classification, distinguishing between above and below average performance(区分平均水平的上方和下方)
		- In training, we compute the F1 score of the answer for every instance
			- F1 score
		- If the rewrite produces an answer with an F1 score greater than the average score of the other rewrites the instance is assigned a positive label
			- assigned a positive label
		- We ignore questions where all rewrites yield equally good/bad answers(产生相同好的/坏的结果的问题会被忽略)
		- For the classifier we evaluated FFNNs, LSTMs and CNNs and found that the performance of all systems was comparable. Since the inputs are triples of variable length sequences the latter two allow us to incorporate the tokens directly without the need for feature engineering.(由于输入是可变长度序列的三元组, 后两个允许我们直接合并令牌, 而无需进行特征工程)
		-  We choose a CNN for computational efficiency.

		-  Select Answer
		-  In particular, we use pre-trained embeddings for the tokens of query, rewrite, and answer.
		-  For each, we add a 1-D CNN followed by max-pooling.