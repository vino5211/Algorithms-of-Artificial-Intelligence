# Evaluate
+ https://www.zhihu.com/question/30643044
+ accuracy
+ P R F1
+ ROC/AUC
	+ https://www.douban.com/note/284051363/

### Accuracy, Precision, Recall
+ TP, TN, FP, FN的定义
	+ TP: 预测为1(Positive)，实际也为1(Truth-预测对了)
	+ TN: 预测为0(Negative)，实际也为0(Truth-预测对了)
	+ FP: 预测为1(Positive)，实际为0(False-预测错了)
	+ FN: 预测为0(Negative)，实际为1(False-预测错了)
	+ 总的样本个数为：TP+TN+FP+FN


|  |Real=1 | Real=0|
|---|---|---|
|Predict=1 | TP | FP |
|Predict=0 | FN | TN|


+ 计算
	+ Accuracy = (预测正确的样本数)/(总样本数)=(TP+TN)/(TP+TN+FP+FN)
	+ Precision = (预测为1且正确预测的样本数)/(所有预测为1的样本数) = TP/(TP+FP)
	+ Recall = (预测为1且正确预测的样本数)/(所有真实情况为1的样本数) = TP/(TP+FN)