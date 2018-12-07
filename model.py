
# coding: utf-8

# In[ ]:


def model_architecture():
    
    X_1=Input((224,224,1))
    X_vgg16=Input((1000,))
    
    x=Conv2D(64,(3,3),activation='relu',padding='same',strides=(2,2))(X_1)
    x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x=Conv2D(128,(3,3),activation='relu',padding='same',strides=(2,2))(x)
    x=Conv2D(256,(3,3),activation='relu',padding='same')(x)
    x=Conv2D(256,(3,3),activation='relu',padding='same',strides=(2,2))(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same')(x)
    x=Conv2D(256,(3,3),activation='relu',padding='same')(x)
     
    
    # object classification model
    
    y_vgg16=RepeatVector(28*28)(X_vgg16)
    y_vgg16=Reshape((28,28,1000))(y_vgg16)
    
    
    #fusion
    
    x=concatenate([x,y_vgg16],axis=-1)  # capital c 'C' concatenate has got some issues so use lower case one
    
    x=Conv2D(256,(1,1),activation='relu',padding='same')(x)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x=UpSampling2D((2,2))(x)
    y_a=Conv2D(26,(1,1),activation='softmax',padding='same')(x)
    y_b=Conv2D(26,(1,1),activation='softmax',padding='same')(x)
    
    model=Model( inputs= [X_1, X_vgg16] , outputs=[y_a,y_b])
    
    return model    

