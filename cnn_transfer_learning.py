
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from skimage import io
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input



conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

model =Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

conv_base.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())


batch_size=32

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# The list of classes will be automatically inferred from the subdirectory names/structure under train_dir
train_generator = train_datagen.flow_from_directory(
     r'C:\Users\Jatin\Desktop\dogcat\train',
    target_size=(224, 224), # resize all images to 224 x 224
    batch_size=batch_size,
    class_mode='binary') # because we use binary_crossentropy loss we need binary labels

validation_generator = test_datagen.flow_from_directory(
    r'C:\Users\Jatin\Desktop\dogcat\validation',
    target_size=(224, 224), # resize all images to 224 x 224
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size, 
    epochs=1,
    validation_data=validation_generator,
    validation_steps=800 // batch_size) 



# Model evaluation
scores_train = model.evaluate(train_generator,verbose=0)
scores_test = model.evaluate(validation_generator,verbose=0)
print("Train Accuracy: %.2f%%" % (scores_train[1]*100))
print("Test Accuracy: %.2f%%" % (scores_test[1]*100))



#For plotting Accuracy and Loss
def plot_accuracy_and_loss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
plot_accuracy_and_loss(history)