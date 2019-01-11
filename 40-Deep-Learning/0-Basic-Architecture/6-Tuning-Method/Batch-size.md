## Train Setting
- Batch-size
	+ batch size 32 相当于 前 32 次反向传播求得参数（w，b）的平均值 做 第33 次输入的初始化参数，即 item1，item2, ..., item 32 的反向传播的得到的参数做 batch 1 的结果，batch 1 的结果做batch 2 的初始化参数
	+ batch_size设的大一些，收敛得快，也就是需要训练的次数少，准确率上升得也很稳定，但是实际使用起来精度不高。batch_size设的小一些，收敛得慢，而且可能准确率来回震荡，所以还要把基础学习速率降低一些；但是实际使用起来精度较高。一般我只尝试batch_size=64或者batch_size=1两种情况。