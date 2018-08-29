# Memory Network(较多应用在问答中)
## Reference
- End-to-End Memory Networks

## 简单版
- query 和某个document $x_i$计算 Match, 得到$a_i$
- 由 $x_i$ 和 $a_i$ 的加权求和 得到 抽取出的信息, 并介入DNN网络, 得到最终Answer

![](/home/apollo/Pictures/Mem2.png)
## 较复杂的版本
- 计算Match 的时候使用 $x_i$
- 加权求和的时候使用$h_i$
- 使用两种不同的文本表示, 最终得到的结果较好
![](/home/apollo/Pictures/Mem1.png)

## Hopping
- 将Extract Information 作为 Query 的 vector 再次输入
- 可反复输入, 根据经验可以提高精度(具体原理不清楚)(反复思考,可以得到更精确的结果)


## Tree-LSTM + Attention