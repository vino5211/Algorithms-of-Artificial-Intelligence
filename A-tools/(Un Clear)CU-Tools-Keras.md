# Keras Clear up

Blog
-
https://blog.keras.io/
https://github.com/fchollet/keras-blog


Demo
-
Keras as a simplified interface to TensorFlow: tutorial    https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html


Document
-
https://keras.io/getting-started/sequential-model-guide/

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