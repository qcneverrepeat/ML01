import decisionTree

import pandas as pd
filename = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter5/demo/data/sales_data.xls' # 当前路径省略即可，间隔用正斜杠；R中的当前路径以.代替
filename2 = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter6/demo/data/model.xls'
raw = pd.read_excel(filename2)
x = raw.iloc[:,:3]
y = raw.iloc[:,3]

tree = decisionTree.Tree()
tree.fit(x,y)
tree.traversal()
