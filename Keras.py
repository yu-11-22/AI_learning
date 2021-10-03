# CNN
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

print("train data:", x_img_train.shape, y_label_train.shape)
print("test data:", x_img_test.shape, y_label_test.shape)


def show(x_img, i):
    plt.figure(figsize=(2, 2))
    plt.imshow(x_img[i])
    plt.show()


show(x_img_test, 98)

x_img_train_normalize = x_img_train.astype("float32")/255
x_img_test_normalize = x_img_test.astype("float32")/255

y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

y_label_train_OneHot.shape
y_label_test_OneHot.shape

label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
              4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def show(x_img, y_label, i):
    print("label:", label_dict[y_label[i][0]], "predict:???")
    plt.figure(figsize=(2, 2))
    plt.imshow(x_img[i])
    plt.show()


show(x_img_test, y_label_test, 98)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=(32, 32, 3),
                 activation="relu",
                 padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 activation="relu",
                 padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1024, activation="relu"))

model.add(Dense(10, activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,
                          validation_split=0.2, epochs=10, batch_size=128, verbose=0)

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)

scores[1]

prediction = model.predict_classes(x_img_test_normalize)

prediction[:10]

y_label_test[:10]
