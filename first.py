import pandas as pd
import os
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score



pd.set_option('display.width', 1000)

data_folder = os.path.join("D:\python code")
data_filename = os.path.join(data_folder,"NBA","dicision trees sample.csv")

database=pd.read_csv(data_filename)

#清洗数据，调整格式
database.columns = ["Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts","Score Type2", "OT?", "Notes"]
#database.ix[:5] ix方法过期了

del database["Score Type2"]

#加一列HomeWin，来判断主场是否胜利
database["HomeWin"]=database["VisitorPts"]<database["HomePts"]

#用y_ture存起来
y_ture=database["HomeWin"].values

print("主场胜利概率: {0:.1f}%".format(100 * database["HomeWin"].sum() / database["HomeWin"].count()))

won_last = defaultdict(int)

#增加两列
database["HomeLastWin"] = False
database["VisitorLastWin"] = False
#判断上一场是否胜利 为上面两列赋值 .iterrows专门用来遍历表
for index, row in database.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    database.iloc[index] = row
    # Set current win
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]


#创建决策树
clf = DecisionTreeClassifier(random_state=14)

X_previouswins = database[["HomeLastWin", "VisitorLastWin"]].values
y_true = database["HomeWin"].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("由上场比赛数据训练出来准确率（决策树）: {0:.1f}%".format(np.mean(scores) * 100))



