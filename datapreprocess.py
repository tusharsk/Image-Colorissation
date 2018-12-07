
# coding: utf-8

# In[ ]:


def load_dataset(s,num):
    '''
    This function reads image one by one and return numpy arrays containing values --
    
    X_LAB = It contains luminance value of all images and is of size : Number_of_samples*224*224*1
    X_features =It contains features extracted after running image through VGG16 model. It is of size
                Number_of_samples*1000
    Y= It contains values of a,b in LAB color space for each image. It is of size Number_of_samples*224*224*2
    
    Parameter s: It can be 'training_images' to load training data and 
                'test_images' for tesing data.
            num: It indicates total number of images to use.
    
    '''
    X_LAB=[]
    X_features=[]
    Y_a=[]
    Y_b=[]
    model=VGG16(weights=r"vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    for i in range(num):
        img_path=s+'/'+str(i)+'.jpg'
        img=image.load_img(img_path,target_size=(224,224))
        img=image.img_to_array(img)
        
        img_lab=rgb2lab(1.0*img/255)
        x_lab=img_lab[:,:,0]/100    #To center input around 0,1 remove it if not required update the prediction part accordingly
        y_a=img_lab[:,:,1]
        y_b=img_lab[:,:,2]
        
        x_lab=np.expand_dims(x_lab,axis=-1)
        
        X_LAB.append(x_lab)
        
        img=preprocess_input(img)
        img=np.expand_dims(img,axis=0)
        x_features=model.predict(img)
        x_features=np.squeeze(x_features,axis=0)
        X_features.append(x_features)
        
        Y_a.append(y_a)
        Y_b.append(y_b)
        
    X_LAB=np.array(X_LAB,dtype=float)
    X_features=np.array(X_features,dtype=float)
   
    
    Y_a=np.array(Y_a,dtype='int32')
    Y_a=Y_a//10
    Y_a=Y_a+13
    Y_a=to_categorical(Y_a,num_classes=26)
    
    Y_b=np.array(Y_b,dtype='int32')
    Y_b=Y_b//10
    Y_b=Y_b+13
    Y_b=to_categorical(Y_b,num_classes=26)
    
    print(X_LAB.shape)
    print(X_features.shape)
    print(Y_a.shape)
    print(Y_b.shape)
        
    return (X_LAB,X_features,Y_a,Y_b)
    

