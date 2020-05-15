# DD2424, Marcus Jirwe 960903, Eric Lind 961210, Matthew Norström 970313

# E-Level + some sophisticated data augumentation

# Changes done from base VGGNet:
# 1. Changed up the data augmentation so that the normalisation of the data is done outside the generator.
# 2. Changed the test data so that it is not created from the generator but is always the same.
# 3. Added a schedule for reducing the learning rate, allowing for faster training convergence.
# 4. Batch size reduced (image from using 32).
# 5. A second image for fair comparison was made using batch size 128 after 32 took too long. 

# Batch size 32:
    # Convergence training accuracy: 0.9204
    # Convergence validation accuracy: 0.8908

# Batch size 128:
    # Convergence training accuracy: 0.9535
    # Convergence training loss: 0.3515
    # Convergence validation accuracy: 0.9003
    # Convergence validation loss: 0.5753

# Parameters:
'''
Total params: 16,890,186
Trainable params: 16,873,546
Non-trainable params: 16,640
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

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

def lr_schedule(epoch): # Source, example of resnet / keras documentation
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 170:
        lr *= 1e-4
    elif epoch > 150:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

if __name__=="__main__":
    # A ImageDataGenerator allowing dynamic data augmentation. 
    # Data augmentation includes rescaling the images, normalising the data, rotation of images, shifting images, and zooming.
    # Zoom range and fill mode are new
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range = 0.2,
        fill_mode = 'nearest',
        horizontal_flip=True)

    # Define a simple sequential model, allowing us to stack layers on top of each other.
    model = Sequential()

    # Simplified VGGNet without any residuals:
    lamda = 5e-4 # The parameter for weight decay regularisation.

    # Because the model is very deep, we apply batch normalisation and dropout, which seemed to improve generalisation.

    # Reference for structure of VGGNet Model:
    # Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

    # We chose to mimic configuration D. However, only one fully connected layer was used as recommended by examiner.

    # Input shape as defined by CIFAR-10 images.
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer = regularizers.l2(lamda), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer= Adam(learning_rate=1e-3), metrics=['accuracy'])
    # End of model.

    model.summary()

    # Batch size defined in article, with some epoch number. Number of classes defined by CIFAR-10.
    EPOCHS = 200
    BS = 128
    num_classes=10

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=1,
                               patience=10,
                               verbose = 1,
                               min_lr=0.5e-8)

    #callbacks = [lr_scheduler]
    callbacks = [lr_reducer]

    # Download CIFAR-10 from keras. Only done once per machine.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Change the labels to one-hot encoding.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Save the history of the training in a dictionary. This is the accuracy and loss over the epochs (training & test). 
    history = model.fit(train_datagen.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS, epochs=EPOCHS, callbacks = callbacks)


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