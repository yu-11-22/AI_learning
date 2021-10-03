# 8/10
# numpy模組 (處理矩陣)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
arrayA = np.array([1, 2, 3])
arrayB = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("arrayA:\n", arrayA)
print()
print("arrayB:\n", arrayB)
print(arrayB.shape)     # shape (列y,行x)
# 新增陣列元素  axis代表方向
newArrayA = np.append(arrayA, [4, 5, 6])
print("newArrayA:\n", newArrayA)
newArrayB = np.append(arrayB, [[7, 8, 9]], axis=0)
print("newArrayB:\n", newArrayB)
newArrayC = np.append(arrayB, [[7], [8]], axis=1)
print("newArrayC:\n", newArrayC)
# 一維陣列插入元素
newArrayD = np.insert(arrayA, 0, [0])
print("newArrayD:\n", newArrayD)
# 二維陣列插入元素
newArrayE = np.insert(arrayB, 0, [[-2, -1, 0]], axis=0)
print("newArrayE:\n", newArrayE)
newArrayF = np.insert(arrayB, 0, [[0]], axis=1)
print("newArrayF:\n", newArrayF)
# 一維、二維陣列刪除元素
newArrayG = np.delete(arrayA, 0)
print("newArrayG:\n", newArrayG)
newArrayH = np.delete(arrayB, 0, axis=0)
print("newArrayH:\n", newArrayH)
newArrayI = np.delete(arrayB, 0, axis=1)
print("newArrayI:\n", newArrayI)
# 一維、二維陣列查詢元素
print(arrayA[0])
print(arrayB[1, 0])
# 一維、二維陣列取代元素
arrayA[0] = 7
print(arrayA)
arrayB[1, 2] = 7
print(arrayB)
# 快速製作陣列
a = np.arange(15).reshape(3, 5)
print(a)
print(a.ndim)               # 陣列維度數量
print(a.size)               # 陣列中元素總數量
print(type(a))              # 陣列的資料型別
print(a.dtype.name)         # 陣列中元素的資料型別
print(a.itemsize)           # 陣列各元素占多少位元組
print(a.size*a.itemsize)    # 陣列佔記憶體的空間有多少位元組
# matplot模組   (畫圖處理)
x = np.linspace(-3, 3, 50)   # 宣告一個一維陣列，元素值在-3~3之間共有50個點
y1 = 2*x+1
y2 = x**2
# y1的圖
plt.figure()                 # figure空白圖片
plt.plot(x, y1)
# 畫出有y1,y2的圖
plt.figure(figsize=(8, 5))  # 設定圖片寬和高，單位為英吋
plt.plot(x, y2, color="yellow", linewidth=5.0, linestyle="-")  # 畫y2的線
plt.plot(x, y1, color="red", linewidth=5.0, linestyle="--")    # 畫y1的線
plt.title('我是圖表標題')     # 設定圖表標題
plt.show()                   # 將兩張圖產生出來
spending_df = pd.DataFrame({
    "生活費": [8000, 7000],
    "娛樂費": [3000, 5000],
    "交通費": [1000, 1200]
})
spending_df.head()
# 新增一欄
spending_df["住宿費"] = [8000, 8000]
spending_df.head()
# 新增一列
spending_df.loc[2] = [7500, 4000, 1200, 8000]    # 使用.loc
spending_df.head()
# 改變df欄名稱
rename_attribute = {"生活費": "110年生活費", "娛樂費": "110年娛樂費",
                    "交通費": "110年交通費", "住宿費": "110年住宿費"}
spending_df = spending_df.rename(rename_attribute, axis=1)
spending_df.head()
# 改變df列名稱
rename_index = {0: "一月", 1: "二月", 2: "三月"}
spending_df = spending_df.rename(rename_index, axis=0)
spending_df.head()
# 儲存成csv檔
spending_df.to_csv("spending_df.csv")
# 使用遮罩(mask)查詢
mask1 = spending_df["110年生活費"] == 7000
spending_df[mask1]
mask2 = spending_df["110年交通費"] > 1000
spending_df[mask2]
# 使用遮罩(mask)查詢,包含兩個條件,需兩個條件都成立
spending_df[(mask1 & mask2)]
