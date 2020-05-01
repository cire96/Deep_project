

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Add
from tensorflow.keras import regularizers

def res_net_block(model,input_Data, n_Filter, conv_Size=(3,3),conv_Stride=1):
    print(model.summary())
    print(input_Data)
    x = model.add(Conv2D(n_Filter, conv_Size, strides=(conv_Stride,conv_Stride), activation='relu', padding='same')(input_Data))
    x = model.add(BatchNormalization())
    x = model.add(Conv2D(n_Filter, conv_Size, activation=None, padding='same'))
    x = model.add(BatchNormalization())

    if conv_Stride != 1:
        x=model.add(Conv2D(filters=n_Filter,kernel_size=(1,1),strides=(conv_Stride,conv_Stride)))
        x=model.add(BatchNormalization())

    x = model.add(Add()([x, input_Data]))
    x = model.add(Activation('relu')(x))
    
    return x

    '''x = Conv2D(n_Filter, conv_Size, activation='relu', padding='same')(input_Data)
    x = BatchNormalization()(x)
    x = Conv2D(n_Filter, conv_Size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_Data])
    x = Activation('relu')(x)'''


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
    
     # Define a simple sequential model, allowing us to stack layers on top of each other.
    model = Sequential()

    # Simplified VGGNet without any residuals:
    lamda = 5e-4 # The parameter for weight decay regularisation

    x=model.add(Conv2D(64, (7, 7), strides=(2,2), activation='relu', input_shape=(32, 32, 3), padding = 'same' ))
    x=model.add(MaxPooling2D(pool_size = (3,3), strides=(2,2)))
    
    #Conv2
    for i in range(3):
        x = res_net_block(model,x,64)

    #Conv3
    x = res_net_block(model,x,128,conv_Stride=2)
    for i in range(3):
        x = res_net_block(model,x,128)

    #Conv4
    x = res_net_block(model,x,256,conv_Stride=2)    
    for i in range(5):
        x = res_net_block(model,x,256)

    #Conv5
    x = res_net_block(model,x,512,conv_Stride=2)
    for i in range(2):
        x = res_net_block(model,x,512)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(10, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    EPOCHS = 2
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
    

