import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.preprocessing.image import load_img,img_to_array

def train():
    cell_input=Input(shape=(178,268,3))

    base_model=InceptionResNetV2(weights='imagenet',include_top=False,input_tensor=cell_input)
    for i in base_model.layers:
        i.trainable=False

    base_model.layers[-2].output_shape

    model=Model(inputs=base_model.input,outputs=base_model.layers[-2].output)
    x=model.output

    x=GlobalAveragePooling2D()(x)
    x=Dense(256, activation='relu',name='custom_1')(x)
    x=Dense(256, activation='relu',name='custom_2')(x)
    x=Dense(32, activation='relu',name='custom_3')(x)

    prob=Dense(1,activation='sigmoid',name='probability')(x)
    x_coord=Dense(1,activation='sigmoid',name='x_coord')(x)
    y_coord=Dense(1,activation='sigmoid',name='y_coord')(x)

    model=Model(inputs=base_model.input,outputs=[prob,x_coord,y_coord]) 
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.summary()

    # To use the DataGenerator module in utils folder
    import sys
    sys.path.append('./utils/')

    import random
    from data_generator import DataGenerator
    BATCH_SIZE=16
    
    #select few files as training, rest as validation
    image_dir='../../data/localization/Original_Images/Training/'
    files=os.listdir(image_dir)
    random.shuffle(files)
    TRAIN_FILES=files[:int(len(files)*0.7)]
    VALID_FILES=files[int(len(files)*0.7):]
    train_gen_inst=DataGenerator(image_dir='../../data/localization/Original_Images/Training/', 
                                 node_names={'in':model.input_names[0],'out':[i for i in model.output_names]},
                            files=TRAIN_FILES,batch_size=BATCH_SIZE,
                            df_path='../../data/localization/Groundtruths/Fovea_Center_Location/Fovea_Center_Training Set_Markups.csv')
    
    valid_gen_inst=DataGenerator(image_dir='../../data/localization/Original_Images/Training/', 
                                 node_names={'in':model.input_names[0],'out':[i for i in model.output_names]},
                            files=VALID_FILES,batch_size=BATCH_SIZE,
                            df_path='../../data/localization/Groundtruths/Fovea_Center_Location/Fovea_Center_Training Set_Markups.csv')


    model.fit_generator(train_gen_inst.get_generator(),epochs=20,shuffle=False,
                        steps_per_epoch=len(TRAIN_FILES)//BATCH_SIZE,
                        validation_data=valid_gen_inst.get_generator(),
                        validation_steps=len(VALID_FILES)//BATCH_SIZE
                       )
    model.save('my_model.h5')
    
def test():
    from keras.models import load_model
    model=load_model('my_model.h5')
    
    image_dir='../../data/localization/Original_Images/Testing/'
    
    train_gen_inst=DataGenerator(image_dir='../../data/localization/Original_Images/Testing/', 
                                 node_names={'in':model.input_names[0],'out':[i for i in model.output_names]},
                            files=TRAIN_FILES,batch_size=BATCH_SIZE,
                            df_path='../../data/localization/Groundtruths/Fovea_Center_Location/Fovea_Center_Testing Set_Markups.csv')
    
    valid_gen_inst=DataGenerator(image_dir='../../data/localization/Original_Images/Testing/', 
                                 node_names={'in':model.input_names[0],'out':[i for i in model.output_names]},
                            files=VALID_FILES,batch_size=BATCH_SIZE,
                            df_path='../../data/localization/Groundtruths/Fovea_Center_Location/Fovea_Center_Testing Set_Markups.csv')


    model.evaluate_generator(train_gen_inst.get_generator(),epochs=20,shuffle=False,
                        steps_per_epoch=len(TRAIN_FILES)//BATCH_SIZE,
                        validation_data=valid_gen_inst.get_generator(),
                        validation_steps=len(VALID_FILES)//BATCH_SIZE
                       )

def main():
    do_what=sys.argv[1]
    if args.do_what=='train':
        train()
    elif args.do_what=='test':
        test()
    else:
         print("Give 'train' or 'test' as commands")
    
if __name__ == '__main__':
    main()