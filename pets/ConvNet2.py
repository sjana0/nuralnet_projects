#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time


# In[10]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[11]:


# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[12]:


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


# In[13]:


X = X/255.0


# In[14]:


X.shape[1:]


# In[15]:


dense_layers = [0]
layer_sizes = [32, 64]
conv_layers = [3]


# In[ ]:


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            tensorboard = TensorBoard(log_dir='varying_logs/{}'.format(NAME))
            model = Sequential()
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(512))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
            if(layer_size == 64):
                model.save("0x64x3-CNN.model")
            if(layer_size == 32):
                model.save("0x32x3-CNN.model")


# In[ ]:




