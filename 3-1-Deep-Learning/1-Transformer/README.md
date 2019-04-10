# Summary of Transformer

# Info

+ [Attention is all your need](<https://arxiv.org/pdf/1706.03762.pdf>)

# Tips

+ 使用attention 机制，不使用cnn 和rnn， 可并行

+ 提出self-attention，自己和自己做attention，使得每个词都有全局的语义信息（长依赖）：

+ - 由于 Self-Attention 是每个词和所有词都要计算 Attention，所以不管他们中间有多长距离，最大的路径长度也都只是 1。可以捕获长距离依赖关系

+ 提出multi-head attention，可以看成attention的ensemble版本，不同head学习不同的子空间语义

# Framework

+ Self-attention
  + 乘法
  + 加法
+ LN
+ position encoder

+ Position-wise Feed Forward Network

+ Multi-head self attention layer
  + 全连接进行映射
  + split



# Reference

- [transfomer](https://github.com/Kyubyong/transformer)



