# Binary_classification_task_algorithm_framework
## 二分类任务算法框架
### python第三方库版本：
- lightgbm == 2.3.1  
- xgboost ==0.90  
- matplotlib ==3.1.3  
- seaborn ==0.12.2  
- numpy == 1.23.0  
- pandas ==1.5.0  
- catboost==1.0.3
- scikit-learn==0.23.1
- tensorflow==2.2.0
- nltk==3.4.5
## 数据
- 框架测试数据使用的匿名数据，如要使用您的数据，请将您的训练集和测试集分别在对应位置进行替换  
## 模型
|模型一|模型二|模型三|模型四|模型五|模型六|模型七|
| :---: | :--- | :--- | :--- | :---: | :--- | :---: |
|stacking：xgb/lgb(元模型逻辑回归)|stacking:xgb/lgb|stacking:lgb|voting:lgb|lgb:十折交叉验证|voting:xgb/lgb|xgb:十折交叉验证|

## 测试集提交方案
- 根据阈值将七个模型结果分别映射到0和1上，然后对行取众数得到最终的结果（硬投票）  
- 根据ks/auc/f1分别赋予每个模型不同的权重，最后将结果进行融合，并按照概率进行降序排列


## 文件说明
- sub :  模型建模结果输出
- result ：模型融合结果输出
- chart_ouput ：图表文件输出
- models ：模型文件输出

