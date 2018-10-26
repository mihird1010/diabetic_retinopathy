import keras
import os
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from keras.preprocessing.image import load_img,img_to_array

class DataGenerator():
    
    def __init__(self, 
                 image_dir=None,
                 files=[],
                 df_path=None,
                 shape=(268,178),
                 batch_size=32,
                 number_of_epochs=10,
                 steps_per_epoch=64,
                 node_names={'in':'image','out':['prob','x_coord','y_coord']}):
        '''
        Class like image data generator. Reads Images in the directory, data frame with the and can return generators which can
        be passed as argument in the fit_generator function of Keras.
        
        Sample use:        
        train_gen=DataGenerator(image_dir='...', files=TRAINING_FILES,df_path='...')
        val_gen=DataGenerator(image_dir='...', files=VALIDATION`_FILES,df_path='...')
        
        model.fit_generator(generator=train_gen.get_train_generator(),
                            validation_data=valgen.get_val_generator(),
                            validation_steps=BATCH_SIZE,
                            epochs=NUM_EPOCHS,
                            steps_per_epoch=NUM_SAMPLES//NUM_EPOCHS)
        '''
        self.batch_size = batch_size
        self.node_names=node_names
        self.CELL_HEIGHT=shape[1]
        self.CELL_WIDTH=shape[0]
        
        if 2 in [len(i.split('.')) for i in files]:
            self.files=[i.split('.')[0] for i in files]
        else:
            self.files=files
        
        ## Y cordinate and X co ordinate are same as height and width respectively
        ## Y=Height; X=Width
        # Finds how many pixels one image will shift in this epoch. Re-init every epoch start
        # Same shift for nth image in one batch, in one epoch. 
        # Only 500 images so, would make sense to train lot of epochs. 
        # Random shifting will make sure excessive overfitting isn't occuring
        
        self.w_shift=np.random.randint(shape[0]-1, size=(self.batch_size))
        self.h_shift=np.random.randint(shape[1]-1, size=(self.batch_size))
        
        self.image_dir=image_dir
        self.df_path=df_path
        # Yes, return in init
        # Order of outputs is probability, X coordinate, Y coordinate. All less than 1.
    
    # Positive sampling
    def get_positve_samples(self,batch):
        crops=[]
        for ct,data in enumerate(batch):
            res={}
            img=load_img(os.path.join(self.image_dir,'{}.jpg'.format(data['file_name'])))
            img=img_to_array(img)

            # every image gets cropped by the same number of pixels for an epoch
            h=data['y_coord']-self.h_shift[ct]
            w=data['x_coord']-self.w_shift[ct]

            crop=img[h:(self.CELL_HEIGHT+h),w:(self.CELL_WIDTH+w),:]
            res['arr']=(crop)
            res['y_shift']=float(self.h_shift[ct]/self.CELL_HEIGHT)
            res['x_shift']=float(self.w_shift[ct]/self.CELL_WIDTH)
            crops.append(res)
        return crops



    # Negative sampling
    def get_negative_samples(self,batch):
        # Returns array crops with just all data given in batch
        # Attempt 1:
        # select random cells within the central 50%, without the co-ordinates \ob
        crops=[]
        IMAGE_WIDTH=4288
        IMAGE_HEIGHT=2848
        for ct,data in enumerate(batch):

            img=load_img(os.path.join(self.image_dir,'{}.jpg'.format(data['file_name'])))
            img=img_to_array(img)

            # every image gets cropped by the same number of pixels for an epoch
            y=data['y_coord']
            x=data['x_coord']

            x_samp=IMAGE_WIDTH//4+np.random.randint(IMAGE_WIDTH//2)
            y_samp=IMAGE_HEIGHT//4+np.random.randint(IMAGE_HEIGHT//2)

            while (x_samp< x <(x_samp+self.CELL_WIDTH)) or (y_samp< y <(y_samp+self.CELL_HEIGHT)):
                x_samp=IMAGE_WIDTH//4+np.random.randint(IMAGE_WIDTH//2)
                y_samp=IMAGE_HEIGHT//4+np.random.randint(IMAGE_HEIGHT//2)

            crop=img[y_samp:(self.CELL_HEIGHT+y_samp),x_samp:(self.CELL_WIDTH+x_samp),:]
            crops.append(crop)

        return crops

    def get_generator(self):

        #Read df
        df=pd.read_csv(self.df_path)
        df.dropna(axis='columns',how='all',inplace=True)
        df.dropna(inplace=True)
        df=df.rename(columns = {'X- Coordinate':'X','Y - Coordinate':'Y'})

        # Checks if batch size odd
        ODD_BATCH_SIZE=0
        if self.batch_size%2!=0:
            ODD_BATCH_SIZE=1
            batch_size=int((self.batch_size+1)/2)
        else:
            batch_size=int(self.batch_size/2)

        X=self.files
        while True:
            batch=yield
            if batch is not None:
                start=batch*batch_size
                if start>len(X)-batch_size:
                    start=-batch_size
                    x=X[start:]

                else:
                    x=X[start:start+batch_size]
            else:
                x=X[:batch_size]

            # Working on the batch to get the required data

            crop_inputs=[]
            for i in x:
                data={}
                data['file_name']=i
                df[df['Image No']==i]['X']
                data['x_coord']=int(df[df['Image No']==i]['X'])
                data['y_coord']=int(df[df['Image No']==i]['Y'])
                crop_inputs.append(data)

            #return stuff    


            #getting positive crops
            cropped_stuff=self.get_positve_samples(crop_inputs)
            arr_pos=np.array([i['arr'] for i in cropped_stuff])
            x_cord_pos=np.array([i['x_shift'] for i in cropped_stuff])
            y_cord_pos=np.array([i['y_shift'] for i in cropped_stuff])

            #getting negative crops
            arr_neg=np.array(self.get_negative_samples(crop_inputs))

            # Combining
            result_in={}
            result_in[self.node_names['in']]=np.concatenate((arr_pos,arr_neg),axis=0)


            # Pro🅱️a🅱️ility for a 🅱️atch
            ones=np.ones(batch_size)
            if ODD_BATCH_SIZE:
                zeros=np.zeros(batch_size-1)
            else:
                zeros=np.zeros(batch_size)

            result_out={}
            result_out[self.node_names['out'][0]]=np.concatenate((ones,zeros),axis=0)                    
            result_out[self.node_names['out'][1]]=np.concatenate((x_cord_pos,zeros),axis=0)
            result_out[self.node_names['out'][2]]=np.concatenate((y_cord_pos,zeros),axis=0)

            for node_out in self.node_names['out']:
                arr=[i for i in cropped_stuff]

            # Sending
            yield (result_in,result_out)