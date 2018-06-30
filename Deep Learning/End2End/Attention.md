# Attention
## Reference
+ https://blog.csdn.net/mpk_no1/article/details/72862348
+ https://blog.csdn.net/aliceyangxi1987/article/details/76284315
	+ 如何自动生成文本摘要
		+ https://blog.csdn.net/aliceyangxi1987/article/details/72765285
	+ 使聊天机器人的对话更有营养
		+ https://blog.csdn.net/aliceyangxi1987/article/details/76128058
	+ AM 第一次应用到 NLP
		+ NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
			+ https://arxiv.org/pdf/1409.0473.pdf
## Soft AM
## Global AM / Local AM
## 静态AM
## 强制前向AM
## 加性注意力（additive attention）
## 乘法（点积）注意力（multiplicative attention）
## 自注意力（self-attention）
## 关键值注意力（key-value attention）


## 静态AM的代码
```
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
 
class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """
 
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        super(Attention_layer, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
 
        a = K.exp(uit)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
 
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print a
        # a = K.expand_dims(a)
        print x
        weighted_input = x * a
        print weighted_input
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

```