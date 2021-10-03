# 整合學習ensemble 將有限的模型結合在一起
# MLP
# MNIST阿拉伯數字辨識 60000個訓練資料10000個測試資料
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils    # 資料前處理
from keras.datasets import mnist    # MNIST位置 從別的雲端資料下載
# mnist.load_data資料集丟進四個變數中
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
# 訓練資料前 先確定使用者手冊資料是否正確
print("x_train_image:", x_train_image.shape)
print("y_train_label:", y_train_label.shape)
print("x_test_image:", x_test_image.shape)
print("y_test_image:", y_test_label.shape)
# 定義一個圖片函式


def plot_image(image):
    fig = plt.gcf()                       # 畫一個圖片
    fig.set_size_inches(2, 2)            # 設計圖片size
    plt.imshow(image, cmap="binary")     # cmap色彩空間表達 binary黑白
    plt.show()


# 印出圖片
plot_image(x_train_image[0])
# 印出標籤
y_train_label[0]
# 訓練模型前先架設模型
x_Train = x_train_image.reshape(60000, 784).astype("float32")
x_Test = x_test_image.reshape(10000, 784).astype("float32")
# 確定是否架設完成
print("x_train:", x_Train.shape)
print("x_test:", x_Test.shape)
# 上下限確定並設定變數
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255
# 用utils編碼0跟1 並指派到OneHot變數
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
# 測試
y_TrainOneHot[0]
y_train_label[:10]
y_TrainOneHot[:10]
# Sequential神經網路模型
model = Sequential()
model.add(Dense(units=256,                  # output256
                input_dim=784,                  # 第一層input784
                kernel_initializer="normal",    # 每個訊號都有權重
                activation="relu"))
model.add(Dense(units=10,
                kernel_initializer="normal",
                activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
train_history = model.fit(x=x_Train_normalize,
                          y=y_TrainOneHot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=0)

scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print("accuracy=", scores[1])

prediction = model.predict_classes(x_Test_normalize)
prediction

y_test_label
