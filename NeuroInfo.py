import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

log = model.fit(x_train, y_train_cat, batch_size=5, epochs=15, validation_split=0.2)

model.evaluate(x_test, y_test_cat)


# Тестирование #


def cifar_object(ans):
    print('-' * 150)
    if ans == 0:
        return "Самолёт"
    elif ans == 1:
        return "Автомобиль"
    elif ans == 2:
        return "Птица"
    elif ans == 3:
        return "Кот"
    elif ans == 4:
        return "Олень"
    elif ans == 5:
        return "Собака"
    elif ans == 6:
        return "Лягушка"
    elif ans == 7:
        return "Лошадь"
    elif ans == 8:
        return "Корабль"
    else:
        return "Грузовик"


for i in range(1, 25):
    image = Image.open(str(i) + '.png').resize((32, 32))
    image1 = np.asarray(image).reshape(32, 32, 3)
    image1 = image1 / 255

    rec_img = np.expand_dims(image1, axis=0)
    rec_res = model.predict(rec_img)

    print(cifar_object(np.argmax(rec_res[0])))
    plt.imshow(image1)
    plt.show()
