
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

from model import model_architecture

from PIL import Image
import numpy as np

def preprocess_pred(gray_img):
    rgb_img=gray2rgb(gray_img)
    x_lab=rgb2lab(1.0*rgb_img/255)[:,:,0]
    x_lab=x_lab/100
    
    rgb_img=preprocess_input(rgb_img)
    rgb_img=np.expand_dims(rgb_img,axis=0)
    model=VGG16(weights=r"vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    x_features=model.predict(rgb_img)
    
    x_lab=np.expand_dims(x_lab,axis=0)
    x_lab=np.expand_dims(x_lab,axis=-1)
    
    return [x_lab,x_features]

def pred_coloredimage(model,gray_img):
    X=preprocess_pred(gray_img)
    Y_a,Y_b=model.predict(X)
    Y_a=np.argmax(Y_a,axis=-1)
    Y_a=Y_a*10.0
    Y_a=Y_a-125.0
    Y_a=np.expand_dims(Y_a,axis=-1)
    Y_b=np.argmax(Y_b,axis=-1)
    Y_b=Y_b*10.0
    Y_b=Y_b-125.0    
    Y_b=np.expand_dims(Y_b,axis=-1)
        
    colored_img=np.concatenate((X[0],Y_a,Y_b),axis=-1)
    colored_img=np.squeeze(colored_img,axis=0)
    return colored_img


def predict(img_path):
    # using color_mode as grayscale reads only the luminance component from the image
    K.clear_session()
    gray_img=image.load_img(img_path,color_mode='grayscale',target_size=(224,224)) 
    gray_img=image.img_to_array(gray_img)
    gray_img=np.squeeze(gray_img,axis=-1)
    model=model_architecture()
    model.load_weights('model.h5')
    colored_img=pred_coloredimage(model,gray_img)
    colored_img=lab2rgb(colored_img)
    #imshow(colored_img)
    plt.show()
    
    name=[]
    name=img_path.split('.')
    t_name=name[0].split('/')
    
    #Rescale to 0-20A55 and convert to uint8
    rescaled = (255.0 / colored_img.max() * (colored_img - colored_img.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)

    new_path=t_name[1]+'_colored.png'
    im.save(new_path)
    return new_path

