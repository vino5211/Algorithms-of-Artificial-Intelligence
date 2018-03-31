# Keras Clear up

## Emedding Demo
- https://yq.aliyun.com/articles/221681

## Blog
- https://blog.keras.io/
- https://github.com/fchollet/keras-blog


## Demo
- Keras as a simplified interface to TensorFlow: tutorial  
- https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html


## Document
- https://keras.io/getting-started/sequential-model-guide/

+ Models
	+ Sequential
	+ Model(Functional API)
+ Layers
	+ core Layers
	+ Convolutional Layers
		+ https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/
	+ Pooling Layers
    + Locally-connected Layers
    + Recurrent Layers
    + Embedding Layers
    + Merge Layers
    + Advanced Activations Layers 
    +  Normalization Layers
    +  Noise Layers
    +  Layer Wappers
    +  Writing your own Keras Layers
+ Preprocessing
	 + Sequence
	 + Text
     + Image
+ Losses   
+ Metrics
+ Optimizers
+ Activations
+ Callbacks
+ Datasets
+ Applications
+ Backend
+ Initializers
	+ Initialize method
		+ https://keras-cn.readthedocs.io/en/latest/other/initializations/
+ Regularizers
+ Constraints
+ Visualization
+ SKLearn API
+ Utils
	+ to_categorical
	```
	# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵，
	# 相当于将向量用one-hot重新编码
	Y_train = np_utils.to_categorical(y_train, nb_classes) 
	# y_train  5, 0, 4,.... 
	#Y_train  [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.], [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.],[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
	Y_test = np_utils.to_categorical(y_test, nb_classes) 
   ```


## Example
+ MLP
	![](http://image109.360doc.com/DownloadImg/2017/08/2407/109171385_2_20170824075454814)
    ```
    model = Sequential()
    model.add(Dense(5,input_shape(4,),activation='sigmoid'))
    model.add(Dense(1,activation='sigmod'))
    ```
+ 全连接
	![](http://image109.360doc.com/DownloadImg/2017/08/2407/109171385_4_20170824075455189)

    ```
   	Model = Sequential()
	# 输入层 + 隐含层1
    Model.add(Dense(10,activation='sigmod'),input_shape=(8,))
    # 隐含层2
    Model.add(Dense(8,activation='relu'))
	# 隐含层3
    Model.add(Dense(10,activation='relu'))
    # 输出层
    Model.add(Dense(5,activation='softmax'))
    ```