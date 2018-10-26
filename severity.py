import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator

from keras.applications.vgg16 import VGG16,preprocess_input
nmod=VGG16(include_top=True,weights='imagenet')
for i in nmod.layers:
    i.trainable=False
    
def get_data_as_np(data_dir,df_location,img_height=331,img_width=331):
    # Inputs: Image directory, labels csv location
    # Outputs: Numpy arrays of 
    X=[]
    Y1=[]
    for file,tq in zip(os.listdir(data_dir),tqdm_notebook(range(len(os.listdir(data_dir))))):
        img=load_img(os.path.join(data_dir,file),target_size=(img_height,img_width))
        img=img_to_array(img)
        X.append(img)
    
    X=np.array(X)
    X=preprocess_input(X)
    
    df=pd.read_csv(df_location)
    Y1=(np.eye(np.ptp(df['Retinopathy grade'])+1)[df['Retinopathy grade']])
    Y2=(np.eye(np.ptp(df['Risk of macular edema '])+1)[df['Risk of macular edema ']])
    
    
    return X,Y1,Y2

def train():
    model=Model(inputs=nmod.input,outputs=nmod.get_layer('fc1').output)
    x=model.output
    #x=Dense(256, activation='relu',name='custom_1')(x)
    x=Dense(16, activation='relu',name='custom_2')(x)
    #x=Dropout(0.5,name='custom_dro_1')(x)
    #x=Dense(2, activation='relu',name='custom_2')(x)
    #x=Dropout(0.2,name='custom_dro_2')(x)
    x=Dense(5,activation='sigmoid',name='custom_3')(x)
    model=Model(inputs=nmod.input,outputs=x)

    model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    IMG_H=model.input_shape[1]
    IMG_W=model.input_shape[2]
    EPOCC=10
    BATCH_SIZE=64
    X_train,Y_train,Z_train=get_data_as_np(data_dir='../data/datasets/Disease_Grading/Original_Images/train/',
                                           df_location='/home/ubuntu/tf/other/dr/data/datasets/Disease_Grading/Groundtruths/a.Disease Grading_Training Labels.csv',
                                           img_height=IMG_H,img_width=IMG_W)
    X_val,Y_val,Z_val=get_data_as_np(data_dir='../data/datasets/Disease_Grading/Original_Images/val/',
                                           df_location='/home/ubuntu/tf/other/dr/data/datasets/Disease_Grading/Groundtruths/b.Disease Grading_Validation Labels.csv',
                                           img_height=IMG_H,img_width=IMG_W)
    nb_train_samples=X_train.shape[0]
    nb_val_samples=X_val.shape[0]

    datagen=ImageDataGenerator(rescale=1./255)
    print("Loaded Data")
    
    checkpointer=keras.callbacks.ModelCheckpoint(filepath='weights/twaa_{epoch:02d}_{val_loss:.2f}.hdf5', monitor='loss',
                                    verbose=1, save_best_only=False, 
                                    save_weights_only=False, mode='auto', 
                                    period=5)
    

    csvlog=keras.callbacks.CSVLogger(filename='weights/log_vgg16.csv', separator=',', append=True)

    model.fit_generator(generator=datagen.flow(X_train,Y_train,batch_size=BATCH_SIZE),
        steps_per_epoch=nb_train_samples // BATCH_SIZE*4,
        epochs=100,
        validation_data=datagen.flow(X_val,Y_val,batch_size=BATCH_SIZE),
            validation_steps=nb_val_samples // BATCH_SIZE*4,
    callbacks=[csvlog,checkpointer])
    
def test(test_dir='/home/ubuntu/tf/other/dr/data/datasets/Disease_Grading/Original_Images/val/',df_loc='/home/ubuntu/tf/other/dr/data/datasets/Disease_Grading/Groundtruths/Labels.csv'):    
    X_test,Y_test,Z_test=get_data_as_np(data_dir=test_dir,
                                           df_location=,
                                           img_height=IMG_H,img_width=IMG_W)
    BATCH_SIZE=16
    datagen=ImageDataGenerator(rescale=1./255)
    model.evaluate_generator(datagen.flow(X_test,Y_test,batch_size=BATCH_SIZE,
                                         steps=X_test.shape[0]//BATCH_SIZE)
                       