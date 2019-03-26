# Summary of NLU

## Domain Classification
### Data
+ Data Labeling
  + 子搭建标注系统(Active Learning) 
  + 使用其他开源的标注工具BRAT
+ Data Statistics and Visualization
  + 纠错脚本：标注数据是否存在格式错误和前后标注不一致
  + 统计脚本：
    + 文本平均长度统计
    + 类别数量统计
    + 标注重复统计
    + 训练集测试集交叉统计
+ Data Preprocessing
  + Segment
    + CNN/BERT : None
    + Fasttext : Standfor Core NLP（Fasttext 分词之后的效果好于不分词的效果）
  + Punctuation
    + 保留
  + Number
    + 连续的数字变为<NUM>
    + 也可按文本的长度转化为对应的token
  + Stopwords
    + None
  + Dict
    + BERT : 单独的字典
    + Others ：Tencent Pretrained Embeeding 中自带的字典
  + Normalizaion
    + <NUM>
    + <PAD>
    + <UNK>

### Metric
+ Accuracy
+ P/R/F1

### Model

##### [Text CNN](https://arxiv.org/abs/1408.5882)

![](https://ws4.sinaimg.cn/large/006tNbRwly1fwv4l4e186j30qd0cjmxx.jpg)

+ Framework
  + Embedding Layer
    - Tencent Embedding
    - Fune-tuning
    - Dim 200
  + Convolution Layer
    + 卷积窗口的长度是词向量维度，宽度是定义的窗口大小
    + Filter size [4, 3, 2]
    + Filter number 256
  + Max Pooling Layer
    + 卷积之后的结果经过 max-pooling 进行特征选择和降维， 得到输入句子的表示
  + Results
    + 句子的表示 通过Dense后有两种方式得到最终的结果
      + Sigmoid(Multi Label)
      + Softmax(Single Label)
+ Experiment and Optimization
  + Hyper Parameters
    + Epoch
    + Early Stopping
    + Dropout
+ Varients
  + CNN 的变体

##### Text RNN

![](../../../../../../../Downloads/1540354954203.png)

+ Framwork
  + Embedding
    + 同CNN
  + BiLSTM
    + 将前向最后一个单元的Hidden state 和 反向最后一个单元的Hidden State 进行拼接
  + Results
    + 同CNN

##### [Fasttext](https://fasttext.cc/)

+ Tips

  + 不能跨平台，在不同的平台下要重新编译

+ Framework

  ![](https://img-blog.csdn.net/20180206120020822?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvam9obl9iaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

+ Input and Output
  + 输入是文本序列(词序列 或 字序列)
  + 输出的是这个文本序列属于不同类别的概率
+ Hierachical Softmax
  + 利用霍夫曼编码的方式编码Label（尤其适用用类别不平衡的情况）

+ N-Gram
  + 原始的是Word Bog， 没有词序信息， 因此加入了N-Gram
  + 为了提高效率，低频的N-Gram 特征要去掉

##### Capsule



##### BERT

+ BERT v.s. OpenAI GPT v.s. ELMo

+ ![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/DR1.png)

+ 

+ Summary of Classification

  | Model   | Tips                        |
  | ------- | --------------------------- |
  | TextCNN | 短文本                      |
  | RNN     | 长文本                      |
  | Fastext | 多类别，大数据量            |
  | Capsule | scalar to vector， 训练较慢 |
  | Bert    |                             |
  |         |                             |

  

## Slot Filling


### SOTAs
- Lattice LSTM + CRF
- deep belief network(DBNs), 取得了优于CRF baseline 的效果
  - https://arxiv.org/pdf/1711.01731.pdf
    - 参考文献15和17