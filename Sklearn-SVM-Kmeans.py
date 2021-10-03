# 監督式學習(告訴機器人答案 讓他去學習)
# linear regression model
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model  # 機器學習模組
# 預設兩個陣列
X = [[0.31], [0.45], [0.48], [0.55], [0.70],
     [0.88], [0.98], [1.05], [1.11], [1.22]]
y = [10000, 12000, 14000, 18000, 22000, 28000, 32000, 36000, 40000, 48000]
# 線性回歸模型丟進regr 開始訓練
regr = linear_model.LinearRegression()
regr.fit(X, y)   # fit標準答案集合(特徵,答案)
# 把X丟進去做預測得出y
y_pre = regr.predict(X)
# 預測x=1的點
y_pre = regr.predict([[1.0]])
y_pre
# 用plt作圖
plt.scatter(X, y, color="black")  # scatter畫點
plt.plot(X, y_pre, color="blue", linewidth=3)  # plot畫線
# 圖形上的 x y 做標題
plt.xlabel("carat")
plt.ylabel("price")
# 印出圖形
plt.show()
# SVM支援向量機 (已分群 用SVM找到最佳邊界)(解決高維度 處理非線性特徵的相互作用)
# 男生:2, 女生:1
X = np.array([[150, 40], [152, 40], [155, 45], [155, 46], [160, 50], [160, 48], [
             170, 65], [175, 70], [176, 80], [178, 75], [180, 82], [182, 80]])
y = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
# SVM的SVC分類器模型丟進clf 開始訓練
clf = SVC()
clf.fit(X, y)
# 訓練完成
print(clf.predict([[152, 42]]))
print(clf.predict([[176, 70]]))
# 非監督式學習(有2樣以上東西 經由特徵分類)
# K-Means (原先不知道怎麼分群 用K-Means找到分群)(K=2就是分2群)
# 只需要初始化X 給K-Means去分群
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [2, 4], [4, 2], [
             10, 10], [9, 8], [8, 9], [8, 8], [7, 7], [8, 6], [6, 8]])
# K-Means模型丟進clf 開始訓練
clf = KMeans(n_clusters=2)  # n_clusters分2群
clf.fit(X)
# 印出這些二維資料的標籤
clf.labels_
# 導入plt畫圖
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [2, 4], [4, 2], [
             10, 10], [9, 8], [8, 9], [8, 8], [7, 7], [8, 6], [6, 8]])
# scatter畫點
# scatter給入x座標和y座標 S是點的大小 c是color
plt.scatter(X[:, 0], X[:, 1], s=100, c=clf.labels_)
# 印出圖形
plt.show()
# numpy內的random做隨機的點
X = np.random.rand(100, 2)  # 100組2維的資料
# 分兩群 訓練
clf = KMeans(n_clusters=2)
clf.fit(X)
# scatter畫點
plt.scatter(X[:, 0], X[:, 1], s=100, c=clf.labels_)
# 印出圖形
plt.show()
