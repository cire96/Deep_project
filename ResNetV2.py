import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Add, Input
from tensorflow.keras import regularizers

def res_net_block(input_Data, n_Filter, conv_Size=(3,3),conv_Stride=1):

    lamda = 5e-4 # The parameter for weight decay regularisation

    x = Conv2D(n_Filter, conv_Size, strides=(conv_Stride,conv_Stride), activation='relu', padding='same')(input_Data)
    x = BatchNormalization()(x)
    x = Conv2D(n_Filter, conv_Size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    if conv_Stride != 1:
        input_Data=Conv2D(filters=n_Filter,kernel_size=(1,1),strides=(conv_Stride,conv_Stride),padding='same')(input_Data)
        input_Data=BatchNormalization()(input_Data)

    x = Add()([x, input_Data])
    x = Activation('relu')(x)


    return x


if __name__ == "__main__":
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
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

    # Same thing but for the test data. Only thing done here is normalising and rescaling.
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True)

    inputs = Input(shape=(32, 32, 3))
    x=Conv2D(64, (7, 7), strides=(2,2), activation='relu', input_shape=(32, 32, 3), padding = 'same' )(inputs)
    x=MaxPooling2D(pool_size = (3,3), strides=(2,2))(x)
    
    #Conv2
    for i in range(3):
        x = res_net_block(x,64)

    Dropout(0.5)

    #Conv3
    x = res_net_block(x,128,conv_Stride=2)
    for i in range(3):
        x = res_net_block(x,128)

    Dropout(0.5)

    #Conv4
    x = res_net_block(x,256,conv_Stride=2)    
    for i in range(5):
        x = res_net_block(x,256)

    Dropout(0.5)

    #Conv5
    x = res_net_block(x,512,conv_Stride=2)
    for i in range(2):
        x = res_net_block(x,512)

    Dropout(0.5)

    x=GlobalAveragePooling2D()(x)
    x=Dense(10, activation='softmax')(x)

    model=keras.Model(inputs,x)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

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