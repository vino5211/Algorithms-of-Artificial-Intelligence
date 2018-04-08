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
    	```
        # model.add(Embedding(len(vacab)+1, 200, weights=[embedding_matrix]))
        # input_dim : 字典长度
        # output_dim : 全链接嵌入的维度
        # input_length : 当输入序列的长度固定时，该值为其长度
        #                如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
        # 输入shape
        # 形如（samples，sequence_length）的2D张量
        # 输出shape
        # 形如(samples, sequence_length, output_dim)的3D张量
        #
    	```
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
	+ 初始化方法定义了对Keras层设置初始化权重的方法,不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是kernel_initializer 和 bias_initializer
	```
    model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    ```
	+ 一个初始化器可以由字符串指定（必须是下面的预定义初始化器之一），或一个callable的函数，例如
	```
    from keras import initializers
	model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))
	# also works; will use the default parameters.
	model.add(Dense(64, kernel_initializer='random_normal'))
	```
    + 预定义初始化方法
    	+ Zeros
    	+ Ones
    	+ Constant
    	+ RandomNormal
    		+ keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
    		+ mean：均值
			+ stddev：标准差
			+ seed：随机数种子
		+ RandomUniform
			+ minval : 均匀分布下边界
			+ maxval : 均匀分布上边界
			+ seed：随机数种子
		+ TruncatedNormal
			+ 截尾高斯分布初始化，该初始化方法与RandomNormal类似，但位于均值两个标准差以外的数据将会被丢弃并重新生成，形成截尾分布。该分布是神经网络权重和滤波器的推荐初始化方法。
		+ VarianceScaling
			+ 该初始化方法能够自适应目标张量的shape。
		+ Orthogonal
			+ 用随机正交矩阵初始化
		+ Identiy
			+ 使用单位矩阵初始化，仅适用于2D方阵
		+ lecun_uniform
			+ LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(3 / fan_in), fin_in是权重向量的输入单元数（扇入）

		+ lecun_normal
			+ LeCun正态分布初始化方法，参数由0均值，标准差为stddev = sqrt(1 / fan_in)的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）
		+ glorot_normal
			+ Glorot正态分布初始化方法，也称作Xavier正态分布初始化，参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）
		+ glorot_uniform
			+ Glorot均匀分布初始化方法，又成Xavier均匀初始化，参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (fan_in + fan_out))。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。
		+ he_normal
			+ He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
		+ he_uniform
			+ LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(6 / fan_in), fin_in是权重向量的输入单元数（扇入）
		+ 自定义初始化器
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