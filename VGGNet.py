# DD2424, Marcus Jirwe 960903, Eric Lind 961210, Matthew Norström 970313

# E-Level + some sophisticated data augumentation

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import regularizers

# A function used for plotting an image. 
def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()

if __name__=="__main__":
    # A ImageDataGenerator allowing dynamic data augmentation. 
    # Data augmentation includes rescaling the images, normalising the data, rotation of images, shifting images, and zooming.
    # Zoom range and fill mode are new
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range = 0.2,
        fill_mode = 'nearest',
        horizontal_flip=True)

    # Same thing but for the test data. Only thing done here is normalising and rescaling.
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True)


    # Define a simple sequential model, allowing us to stack layers on top of each other.
    model = Sequential()

    # Simplified VGGNet without any residuals:
    lamda = 5e-4 # The parameter for weight decay regularisation.

    # Because the model is very deep, we apply batch normalisation and dropout, which seemed to improve generalisation.

    # Reference for structure of VGGNet Model:
    # Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

    # We chose to mimic configuration D. However, only one fully connected layer was used as recommended by examiner.

    # Input shape as defined by CIFAR-10 images.
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer = regularizers.l2(lamda)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    # End of model.

    # Batch size defined in article, with some epoch number. Number of classes defined by CIFAR-10.
    EPOCHS = 500
    BS = 256
    num_classes=10

    # Download CIFAR-10 from keras. Only done once per machine.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Change the labels to one-hot encoding.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Save the history of the training in a dictionary. This is the accuracy and loss over the epochs (training & test). 
    history = model.fit(train_datagen.flow(x_train, y_train, batch_size=BS),
        validation_data=test_datagen.flow(x_test, y_test, batch_size=BS),
        steps_per_epoch=len(x_train) // BS, epochs=EPOCHS)


    # Plot the accuracy for training and test set over the epochs.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    # Plot the loss for training and test set over the epochs.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()




