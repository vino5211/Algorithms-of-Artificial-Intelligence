### ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs

- 第一种方法ABCNN0-1是在卷积前进行attention，通过attention矩阵计算出相应句对的attention feature map，然后连同原来的feature map一起输入到卷积层
- 第二种方法ABCNN-2是在池化时进行attention，通过attention对卷积后的表达重新加权，然后再进行池化
- 第三种就是把前两种方法一起用到CNN中