# Summary of Embedding

## Language Model Embedding
### Feature Based 
+ 利用语言模型的中间结果(LM Embedding), 作为额外的特征，引入到原来的模型中

+ 例如在TagLM中，使用两个单向RNN构成的语言模型，将语言模型的中间结果(如下)，带到序列标注公式中去

  $$ h_i^{LM} = [\overrightarrow {h_i^{LM}}; \overleftarrow{ h_i^{LM} }] $$

  

+ ELMo

### Fuin-tuning

+ 在已经训练好的语言模型基础上，加入少量的task-specific-parameters，例如分类问题在语言模型的基础上加一成softmax，然后在新的语料上进行fine-tuning

+ OpenAI GPT

  ![](https://ws2.sinaimg.cn/large/006tKfTcly1g1nc6ivg5xj30k008igmf.jpg)

### BERT (LM + Fine tuning with specific model)