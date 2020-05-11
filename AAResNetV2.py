# B-Level implementation of attention augmentation on ResNet, requiring custom layers.

# 1. Changed up the data augmentation so that the normalisation of the data is done outside the generator.
# 2. Changed the test data so that it is not created from the generator but is always the same.
# 3. Added a schedule for reducing the learning rate, allowing for faster training convergence. (Test reduce on plateau)
# 4. Batch size reduced (image from using 128).
# 5. Regularisation added in the attention_augmentation file.

# Batch size 128:
    # Convergence training accuracy: 0.8943
    # Convergence training loss: 0.4550
    # Convergence validation accuracy: 0.8514
    # Convergence validation loss: 0.6168

# Parameters:
'''
Total params: 18,107,474
Trainable params: 18,073,554
Non-trainable params: 33,920
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Add, Input
from tensorflow.keras import regularizers
from attention_augmentation import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

def res_net_block(input_Data, n_Filter, conv_Size=(3,3),conv_Stride=1):
    '''This version of a ResNet block uses augmented attention blocks instead of just normal conv. layers.'''


    #ip = Input(shape=(32, 32, 3))
    x = aug_atten_block(input_Data, filters=n_Filter, kernel_size=conv_Size,
    strides=(conv_Stride,conv_Stride), relative_encodings=True, padding='same')

    x = Activation('relu')(x)
    
    #x = Conv2D(n_Filter, conv_Size, strides=(conv_Stride,conv_Stride), activation='relu', padding='same')(input_Data)
    
    x = BatchNormalization()(x)
    #x = Conv2D(n_Filter, conv_Size, activation=None, padding='same')(x)
    x = aug_atten_block(x, filters=n_Filter, kernel_size=conv_Size, padding='same', depth_k=0.25, depth_v=0.25, num_heads=4, relative_encodings=True)

    #x = Activation('relu')(x)
    x = BatchNormalization()(x)

    if conv_Stride != 1:
        #input_Data=Conv2D(filters=n_Filter,kernel_size=(1,1),strides=(conv_Stride,conv_Stride),padding='same')(input_Data)

        # Using an attention augmentation block on the input data here causes a reshape error somehow.

        input_Data=aug_atten_block(input_Data, filters=n_Filter,kernel_size=(1,1),strides=(conv_Stride,conv_Stride),padding='same')
        #augumented?
        input_Data=BatchNormalization()(input_Data)

    x = Add()([x, input_Data])
    x = Activation('relu')(x)


    return x

def main():
    # A ImageDataGenerator allowing dynamic data augmentation.
    # Data augmentation includes rescaling the images, normalising the data, rotation of images, shifting images, and zooming.
    # Zoom range and fill mode are new
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True
        )

    # Simplified VGGNet without any residuals:
    lamda = 5e-4 # The parameter for weight decay regularisation

    inputs = Input(shape=(32, 32, 3))
    #x=Conv2D(64, (7, 7), strides=(2,2), activation='relu', input_shape=(32, 32, 3), padding = 'same' )(inputs)
    x = aug_atten_block(inputs, filters=64, kernel_size=(7,7), strides=(2,2), padding='same',
    num_heads=4, relative_encodings=True)

    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (3,3), strides=(2,2))(x)
    
    #Conv2
    for i in range(3):
        x = res_net_block(x,64) #change name later

    #Conv3
    x = res_net_block(x,128,conv_Stride=2)
    for i in range(3):
        x = res_net_block(x,128)

    #Conv4
    x = res_net_block(x,256,conv_Stride=2)    
    for i in range(5):
        x = res_net_block(x,256)

    #Conv5
    x = res_net_block(x,512,conv_Stride=2)
    for i in range(2):
        x = res_net_block(x,512)

    x = AveragePooling2D(pool_size=8, padding = 'same')(x)
    #x=GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x=Dense(10, activation='softmax', kernel_initializer='he_normal')(x)

    model=keras.Model(inputs,x)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'],experimental_run_tf_function=False)

    model.summary()

    EPOCHS = 300
    BS = 128
    num_classes=10

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=1,
                               patience=15,
                               verbose = 1,
                               min_lr=0.5e-8)

    callbacks = [lr_reducer]

    # Download CIFAR-10 from keras. Only done once per machine.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

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
    
if __name__ == "__main__":
    main()