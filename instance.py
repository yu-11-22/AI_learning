import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans

a = np.random.randint(low=0, high=100, size=100)
x = a[:, np.newaxis]
print(x)

clf = KMeans(n_clusters=3)
clf.fit(x)

clf.labels_

X = x
y = clf.labels_

plt.scatter(X, y, s=10, c=clf.labels_)
plt.show()

clf = SVC()
clf.fit(X, y)

dic = {0: "甲班", 1: "乙班", 2: "丙班"}


def grouping(score):
    print(dic[int(clf.predict([[score]]))])


print(dic)
n = int(input("請輸入人數:"))
for n in range(n, 0, -1):
    score = float(input("請輸入分數:"))
    grouping(score)
