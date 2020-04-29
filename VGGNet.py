import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout






def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

model = Sequential()

# Test model:
'''
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
'''

# Simplified VGGNet without any residuals:

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])




EPOCHS = 100
BS = 128


num_classes=10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


#datagen.fit(x_train)
#test_datagen.fit(x_test)

history = model.fit(datagen.flow(x_train, y_train, batch_size=BS),
    validation_data=test_datagen.flow(x_test, y_test, batch_size=BS),
	steps_per_epoch=len(x_train) // BS, epochs=EPOCHS)

'''
history = model.fit(datagen.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
	steps_per_epoch=len(x_train) // BS, epochs=EPOCHS)
'''

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()




