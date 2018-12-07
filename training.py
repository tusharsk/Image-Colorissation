
# coding: utf-8

# In[ ]:


from keras import layers
from keras.layers import Conv2D,UpSampling2D,Input,Reshape,concatenate,RepeatVector
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
import numpy as np

from skimage.color import rgb2lab,lab2rgb,rgb2gray,gray2rgb
from sklearn.model_selection import train_test_split

import h5py

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
  
get_ipython().run_line_magic('matplotlib', 'inline')

from datapreprocess import load_dataset
from model import model_architecture


# In[ ]:


# Use this first time to create training data

X_LAB_train,X_features_train,Y_a_train,Y_b_train=load_dataset('training_images',2)
X_LAB_test,X_features_test,Y_a_test,Y_b_test=load_dataset('test_images',1)

# Save the training data in H5 file to use next time if required.
h5f = h5py.File('training_data.h5', 'w')
h5f.create_dataset('X_LAB_train', data=X_LAB_train)
h5f.create_dataset('X_features_train', data=X_features_train)
h5f.create_dataset('Y_a_train', data=Y_a_train)
h5f.create_dataset('Y_b_train', data=Y_b_train)
h5f.close()

# Save the testing data in H5 file to reuse next time
h5f = h5py.File('test_data.h5', 'w')
h5f.create_dataset('X_LAB_test', data=X_LAB_test)
h5f.create_dataset('X_features_test', data=X_features_test)
h5f.create_dataset('Y_a_test', data=Y_a_test)
h5f.create_dataset('Y_b_test', data=Y_b_test)
h5f.close()


# In[ ]:


# Use it to load dataset from h5 file
'''
h5f = h5py.File('training_data.h5','r')
X_LAB_train=h5f['X_LAB_train'][:]
X_features_train=h5f['X_features_train'][:]
Y_a_train=h5f['Y_a_train'][:]
Y_b_train=h5f['Y_b_train'][:]
h5f.close()

h5f=h5py.File('test_data.h5','r')
X_LAB_test=h5f['X_LAB_test'][:]
X_features_test=h5f['X_features_test'][:]
Y_a_test=h5f['Y_a_test'][:]
Y_b_test=h5f['Y_b_test'][:]
h5f.close()
'''


# In[ ]:


model=model_architecture()


# In[ ]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) # Decide some good optimiser by reading about possible options


# In[ ]:


model.fit([X_LAB_train,X_features_train],[Y_a_train,Y_b_train],epochs=10,batch_size=2)
model.save_weights('model.h5')


# In[ ]:


print(model.evaluate([X_LAB_test,X_features_test],[Y_a_test,Y_b_test],batch_size=2))

