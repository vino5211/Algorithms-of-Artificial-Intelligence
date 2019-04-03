# View of LSTM
### Reference
+ https://www.jianshu.com/p/32d3048da5ba
+ 有详细的数学推到：零基础入门深度学习(6) - 长短时记忆网络(LSTM)
+ https://www.zybuluo.com/hanbingtao/note/581764

### Outline of LSTM
+  LSTM，即Long Short Term Memory Networks 长短时间记忆网络,是RNN的一个变种，专门用于解决Simple-RNN的俩问题
	+ a：如果出入越长的话，展开的网络就越深，对于“深度”网络训练的困难最常见的是“梯度爆炸( Gradient Explode )” 和 “梯度消失( Gradient Vanish )” 的问题
	+ b：Simple-RNN善于基于先前的词预测下一个词，但在一些更加复杂的场景中，例如，“我出生在法国......我能将一口流利的法语。” “法国”和“法语”则需要更长时间的预测，而随着上下文之间的间隔不断增大时，Simple-RNN会丧失学习到连接如此远的信息的能力
+ LSTM通过对循环层的刻意设计来避免长期依赖和梯度消失，爆炸等问题
+ **长期信息的记忆在LSTM中是默认行为，而无需付出代价就能获得此能力**
+ ### 结构
	+ 从网络主题上来看，RNN和LSTM是相似的，都具有一种循环神经网络的链式形式。在标准的RNN中，这个循环节点只有一个非常简单的结构，如一个tanh层。LSTM的内部要复杂得多，在循环的阶段内部拥有更多的复杂的结构，即4个不同的层来控制来控制信息的交互
	
	+ ### LSTM 图
	
        ![](https://upload-images.jianshu.io/upload_images/2666154-0cd0f54f6003bbd7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
        ![](https://upload-images.jianshu.io/upload_images/2666154-2ffc78d0d62e0541.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/688)
        
        + ### cell 状态
        + 如下图，LSTM中在图上方贯穿运行的水平线指示了隐藏层中神经细胞cell的状态，类似于传送带，只与少量的线交互。数据直接在整个链上运行，信息在上面流动会很容易保持不变。状态C的变化受到控制门的影响
        ![](https://upload-images.jianshu.io/upload_images/2666154-b18a4be852eaf66a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/396)
        + ### 门结构
        + 下图就是一个门，包含一个Sigmoid网络层和一个Pointwise乘法操作。LSTM拥有三个门，来保护和控制细胞状态。0代表“不允许任何量通过”，1代表“允许任何量通过”
        ![](https://upload-images.jianshu.io/upload_images/2666154-5d94f5be49b4879b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/210)
        + ### 遗忘门
        + 首先，决定从细胞状态中丢弃什么信息。这个决策是通过一个称为“遗忘门”的层来完成的。该门会读取 $h_{t-1}$ 和 $x_t$，使用sigmoid函数输出一个在0-1之间的数值，输出给在状态$C_{t-1}$中每个细胞的数值
        + ### 数据更新
        + 然后确定什么样的新信息被存放在细胞状态中。这里包含两部分：
		+ 一部分是Sigmoid层，称为“输入门”，它决定我们将要更新什么值；
		+ 另一部分是tanh层，创建一个新的候选值向量~$C_t$，它会被加入到状态中。
	+ 这样，就能用这两个信息产生对状态的更新
	
        ![](https://upload-images.jianshu.io/upload_images/2666154-e7fd8c1c0e8ca191.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/686)
        
        + ### 更新细胞状态
        + 现在是更新旧细胞状态的时间了，$C_{t-1}$ 更新为 $C_t$ 。前面的步骤已经决定了将会做什么，现在就是实际去完成。把旧状态与 $f_t$ 相乘，丢弃掉我们确定需要丢掉的信息，接着加上$i_t$*~$C_t$。这就是新的候选值，根据更新每个状态的程度进行变化
        
        ![](https://upload-images.jianshu.io/upload_images/2666154-72a1be8cbc4793b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/638)
        
        + ### 输出门 
        + 首先，运行一个Sigmoid层来确定细胞状态的哪个部分将输出出去。接着，把细胞状态通过tanh进行处理( 得到一个在 -1~1 之间的值 ) 并将它和Sigmoid门相乘，最终仅仅会输出我们确定输出的那部分
        + ![](https://upload-images.jianshu.io/upload_images/2666154-aeaab02e20f8df2c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/675)

+ ### Loss
	+ 与RNN相同，都要最小化损失函数 l(t)。下面用 h(t) 表示当前时刻的隐藏层输出，y(t)表示当前时刻的输出标签，参考在后面的代码使用的是平方差损失函数，则损失函数被表示为：
	$$ l(t) = f(h(t), y(t)) = {||h(t)-y(t)||}^2$$
        + 全局的损失函数
        $$ L = \sum_{t=1}^r l(t)$$
        + https://www.jianshu.com/p/32d3048da5ba