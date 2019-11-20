#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten,Activation,ZeroPadding2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam


from sklearn.preprocessing import LabelBinarizer

import numpy as np
import tensorflow as tf
import random
import datetime


# In[2]:


# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

# some constants
DATA_PATH = "/home/fanbaiw/2019/comp 309/project/Train_data_2019/Train_data"
IMAGE_SIZE_TUPLE = (300,300)
IMAGE_SIZE = 300
BATCH_SIZE= 9 # how many images process at one time, the less the better?  from 16 to 9


# In[3]:


def load_data():
    datagen = ImageDataGenerator(rotation_range=40, #data pre process
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    train = datagen.flow_from_directory(DATA_PATH, target_size= IMAGE_SIZE_TUPLE,
                                                      classes=['cherry', 'strawberry', 'tomato'], batch_size = BATCH_SIZE ,
                                                      class_mode='categorical', subset='training')

    validate = datagen.flow_from_directory(DATA_PATH, target_size= IMAGE_SIZE_TUPLE,
                                                      classes=['cherry', 'strawberry', 'tomato'], batch_size = BATCH_SIZE ,
                                                      class_mode='categorical', subset='validation')

    return train, validate 


# In[4]:


def construct_MLP():
   model = Sequential()
   model.add(Flatten(input_shape=(300, 300, 3)))
   model.add(Dense(64, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(3, activation='softmax'))

   model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
   return model


# In[5]:


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()
    
    #Conv2D(filters,kernel_size,inputshape)
    
    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE,3))) 
    model.add(Activation('relu'))   #Activation function
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) # max pooling layer
    
    #---------------------------------Convolutional Layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    #-------------------------------Flatten Layer
    
    model.add(Flatten())  
    #-------------------------------Dense Layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    #-------------------------------- complie the model
    model.compile(loss='categorical_crossentropy',
              optimizer = 'sgd',
              metrics=['accuracy'])
    return model


# In[6]:


def train_model(model, train, validate):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
#     from time import time
#     from keras.callbacks import TensorBoard
#     tb = TensorBoard(log_dir = './logs2{}'.format(time()), 
#                      write_graph = True, write_grads = True, write_images = True)
    
    model.fit_generator(train,steps_per_epoch=2000,validation_data=validate,epochs=70) #callbacks=[tb])
    return model


# In[7]:


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    print("Model Saved Successfully.")
    model.save("model/" + "CNNF" + '.h5')


# In[ ]:


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("Start: {}".format(start_time))
    x,y = load_data()
    #model = construct_model()
    model = construct_MLP()
    trained_model = train_model(model,x,y)
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("Time taken: {}".format(execution_time))
    save_model(model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




