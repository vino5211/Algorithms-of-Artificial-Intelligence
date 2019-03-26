# keras还没有官方实现attention机制，有些attention的个人实现，在mnist数据集上做了下实验。模型是双向lstm+attention+dropout，话说双向lstm本身就很强大了。
# 参考链接:https://github.com/philipperemy/keras-attention-mechanism
# https://github.com/keras-team/keras/issues/1472

# mnist attention

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_DIM = 28
lstm_units = 64

# data pre-processing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# first way attention
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# build RNN model with attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop1 = Dropout(0.3)(inputs)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(10, activation='sigmoid')(drop2)
model = Model(inputs=inputs, outputs=output)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# print('Training------------')
# model.fit(X_train, y_train, epochs=10, batch_size=16)
#
# print('Testing--------------')
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('test loss:', loss)
# print('test accuracy:', accuracy)
