-  Attention-based RNN in NLP
  - Neural Machine Translation by Jointly Learning to Align and Translate
    - 图中我并没有把解码器中的所有连线画玩，只画了前两个词，后面的词其实都一样。可以看到基于attention的NMT在传统的基础上，它把源语言端的每个词学到的表达（传统的只有最后一个词后学到的表达）和当前要预测翻译的词联系了起来，这样的联系就是通过他们设计的attention进行的
    
    - 在模型训练好后，根据attention矩阵，就可以得到源语言和目标语言的对齐矩阵了
    - 具体论文的attention设计如下
    
    - 使用一个感知机公式来将目标语言和源语言的每个词联系了起来，然后通过soft函数将其归一化得到一个概率分布，就是attention矩阵
    
    - 结果来看相比传统的NMT（RNNsearch是attention NMT，RNNenc是传统NMT）效果提升了不少，最大的特点还在于它可以可视化对齐，并且在长句的处理上更有优势
  - Effective Approaches to Attention-based Neural Machine Translation
    - attention在RNN中可以如何进行扩展
    - 提出了两种attention机制，一种是全局（global）机制，一种是局部（local）机制
    - global
    
    - local
      - 主要思路是为了减少attention计算时的耗费，作者在计算attention时并不是去考虑源语言端的所有词，而是根据一个预测函数，先预测当前解码时要对齐的源语言端的位置Pt，然后通过上下文窗口，仅考虑窗口内的词
      - 里面给出了两种预测方法，local-m和local-p，再计算最后的attention矩阵时，在原来的基础上去乘了一个pt位置相关的高斯分布。作者的实验结果是局部的比全局的attention效果好
- Attention-based CNN in NLP
  - ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
- Reference websites
  - http://www.cnblogs.com/robert-dlut/p/5952032.html
