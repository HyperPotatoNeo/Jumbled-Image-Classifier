from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from random import randint


def mix(x_train):
    m=x_train
    a=[0,1,2,3]
    for i in range(3):
        l=randint(0,3)
        a[i]=l
        a[l]=i
    p=0
    for i in range(2):
        for j in range(2):
            if(a[p]<2):
                x_train[:,i*16:i*16+16,j*16:j*16+16,:]=m[:,a[p]*16:a[p]*16+16,0:16,:]
            elif(a[p]==2):
                x_train[:,i*16:i*16+16,j*16:j*16+16,:]=m[:,0:16,16:32,:]
            else:
                x_train[:,i*16:i*16+16,j*16:j*16+16,:]=m[:,16:32,16:32,:]
    return x_train,a

def rearrange(x_test,a):
    b=[0,0,0,0]
    x=0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                if(a[i,j,k]==1):
                    b[x]=k
                    break
    p=0
    m=x_test
    for i in range(2):
        for j in range(2):
            if(b[p]<2):
                x_test[i*16:i*16+16,j*16:j*16+16,:]=m[b[p]*16:b[p]*16+16,0:16,:]
            elif(b[p]==2):
                x_test[i*16:i*16+16,j*16:j*16+16,:]=m[0:16,16:32,:]
            else:
                x_test[i*16:i*16+16,j*16:j*16+16,:]=m[16:32,16:32,:]    
    return x_test

batch_size = 256
num_classes = 10
epochs = 2
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3),padding='same'))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3),padding='same'))
model1.add(Activation('relu'))
model1.add(Conv2D(16, (3, 3),padding='same'))
model1.add(Activation('relu'))
model1.add(Conv2D(4, (3, 3)))
model1.add(Activation('softmax'))



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0007, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
   
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    for i in range(10):
    # Fit the model on the batches generated by datagen.flow().
        model1.load_weights("saved_models/keras_cifar10_jumbled_model.h5")
        for i in range(1000):
            x_train[i*50:i*50+50],_=mix(x_train[i*50:i*50+50])
        for i in range(200):
            x_test[i*50:i*50+50],_=mix(x_test[i*50:i*50+50])
        a=model1.predict(x_train,batch_size=50)
        for i in range(50000):
            x_train[i]=rearrange(x_train[i],a[i])
        a=model1.predict(x_test,batch_size=50)
        for i in range(10000):
            x_test[i]=rearrange(x_test[i],a[i])
        model.load_weights("saved_models/keras_cifar10_trained_model.h5")
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

# Save model and weights
    
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

# Score trained model.
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

