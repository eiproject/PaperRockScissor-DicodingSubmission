import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 300//2, 200//2

paperDir = 'rockpaperscissors/paper/'
rockDir = 'rockpaperscissors/rock/'
scissorDir = 'rockpaperscissors/scissors/'

paperList = os.listdir(paperDir)
rockList = os.listdir(rockDir)
scissorList = os.listdir(scissorDir)

dirTrain = [paperList[:500], rockList[:500], scissorList[:500]]
dirValidation = [paperList[500:650], rockList[500:650], scissorList[500:650]]

def preprocessing(img_name = None, i = None):
    # print(img_name, i)
    if i == 0:
        img_dir = paperDir
    elif i == 1:
        img_dir = rockDir
    else:
        img_dir = scissorDir

    img = cv2.imread(img_dir+img_name) # Open image

    min_HSV = np.array([0, 60, 40], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binaryImg = cv2.inRange(hsvImg, min_HSV, max_HSV)
    masked = cv2.bitwise_and(img, img, mask=binaryImg)

    result = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, (img_width, img_height))
    result = np.expand_dims(result, axis=2)
    # print(binaryImg)
    # cv2.imshow('See This', result)
    # cv2.waitKey(0)

    return result

iTrain = 0
fileTrain = []
indexTrain = []

for dir in dirTrain:
    for file in dir:
        print(file, iTrain)
        newImage = preprocessing(file, iTrain)
        fileTrain.append(newImage)
        if iTrain == 0:
            indexTrain.append([1,0,0])
        elif iTrain == 1:
            indexTrain.append([0,1,0])
        elif iTrain == 2:
            indexTrain.append([0, 0, 1])

    iTrain+=1

iVal = 0
fileValidation = []
indexValidation = []

for dir in dirValidation:
    for file in dir:
        newImage = preprocessing(file, iVal)
        fileValidation.append(newImage)
        if iVal == 0:
            indexValidation.append([1,0,0])
        elif iVal == 1:
            indexValidation.append([0,1,0])
        elif iVal == 2:
            indexValidation.append([0, 0, 1])
    iVal += 1

print('shape = ', np.shape(fileTrain))
print('shape index = ', np.shape(indexTrain))


#
# if K.image_data_format() == 'channels_first':
#     input_shape = (1, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 1)

nb_train_samples = 500*3
nb_validation_samples = 150*3
epochs = 20
batch_size = 32

model = Sequential()
# model.add(Input(shape=(100,150,1)))
model.add(ZeroPadding2D(padding=(2,2), input_shape=(img_height, img_width, 1)))
model.add(Conv2D(32,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides= 2))
model.add(Conv2D(32, (5, 5),activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides= 1))
# model.add(Conv2D(64, (5, 5),activation='relu', strides=1))
# model.add(Conv2D(128, (5, 5),activation='relu', strides=1))
model.add(Flatten())

# model.add(Dense(512))
# model.add(Activation('relu'))
#
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='SGD',
            # optimizer='adam',
            metrics=['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        # horizontal_flip=True
)
train_datagen.fit(x=fileTrain)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True
)
validation_datagen.fit(x=fileValidation)

# train_generator = train_datagen.flow_from_directory(
#         fileTrain,
#         target_size=(300, 200),
#         color_mode='grayscale',
#         batch_size=32,
#         class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#         fileValidation,
#         target_size=(300, 200),
#         color_mode='grayscale',
#         batch_size=32,
#         class_mode='binary')

fileTrain = np.array(fileTrain)
indexTrain = np.array(indexTrain)

fileValidation = np.array(fileValidation)
indexValidation = np.array(indexValidation)

print('shape 2 = ', np.shape(fileTrain))
print('shape 2 index = ', np.shape(indexTrain))

model.fit_generator(
    train_datagen.flow(
        x=fileTrain,
        y=indexTrain,
        batch_size=32,
    ),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=train_datagen.flow(
        x=fileValidation,
        y=indexValidation,
        batch_size=32,
    ),
    validation_steps=nb_validation_samples // batch_size)

print(model.summary())
# model.save_weights('BismillahFirst-5epochs-W.h5')
model.save('model-dicoding-razif.h5')