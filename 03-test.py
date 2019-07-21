
import pandas as pd
filename = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter5/demo/data/sales_data.xls' # 当前路径省略即可，间隔用正斜杠；R中的当前路径以.代替
filename2 = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter6/demo/data/model.xls'
raw = pd.read_excel(filename,index_col = '序号')
x = raw.iloc[:,:3]
y = raw.iloc[:,3] # pandas Series

'''
import decisionTree as DT

# tree = DT.Tree(max_features = None,max_depth = 2)
tree = DT.Tree()
tree.fit(x,y)
tree.traversal()
print(tree.predict(x))
'''

'''
import randomForest as RF
forest = RF.randomForest()
forest.fit(x,y)
print(forest.predict(x,show='all'))
'''


import adaBoost as aB
adB = aB.adaBoost(n_estimators = 30)
adB.fit(x,y)
print(adB.base_weight_set)

# sample_weight_set = pd.DataFrame([1/x.shape[0]] * x.shape[0], index = x.index)
# index = adB.Roulette(sample_weight_set)
# resam_x = x.iloc[index,:]
# resam_y = y.iloc[index]
# resam_x.index = range(x.shape[0])
# resam_y.index = range(x.shape[0])
# print(resam_x)
# print(resam_y)
