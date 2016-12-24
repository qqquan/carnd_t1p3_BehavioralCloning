import argparse
import numpy as np
from util_qDatasetManager import qDatasetManager
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, ActivityRegularization
from keras.layers import Convolution2D, MaxPooling2D
import keras
from time import strftime
from datetime import timedelta, datetime
import pytz
import os

class qModelTrainer:

    def __init__(self, input_file_loc = None, enable_incremental_learning = False, debug_size = None):

        if enable_incremental_learning:
            # new learning materials

            ls_records = [  
                            # 'recordings/rec7_1stCurve/driving_log.csv',
                            # 'recordings/rec8_1stCurve_LeftRecov/driving_log.csv',
                            # 'recordings/rec6_2ndCurve/driving_log.csv',
                            # 'recordings/rec4_recovery/driving_log.csv',
                            'recordings/rec5_udacity/data/driving_log.csv',
                            

                         ]  
        else:
             ls_records = [  
                            # 'recordings/rec3_finer_steering/driving_log.csv',
                            # 'recordings/rec4_recovery/driving_log.csv',
                            # 'recordings/rec2_curve/driving_log.csv',
                            'recordings/rec5_udacity/data/driving_log.csv',
                         ]  
                          
        # self.DatasetMgr = qDatasetManager(ls_records, debug_size=3)
        self.DatasetMgr = qDatasetManager(ls_records, debug_size = debug_size)

        self.InputShape = self.DatasetMgr.getInputShape()

        if enable_incremental_learning:
            self.reloadModel('model.json')
        else:
            self.model = Sequential()
            self.buildModel()


    def buildModel(self):

        self.model.add(Convolution2D(nb_filter= 24, 
                                     nb_row=5,
                                     nb_col=5, 
                                     name='cnn0',
                                     input_shape = self.InputShape
                                    )
                       )

        self.model.add(ELU())
   
        self.model.add(Dropout(0.3))



        self.model.add(Convolution2D(    nb_filter= 36, 
                                         nb_row=5,
                                         nb_col=5, 
                                         border_mode='valid',
                                         name='cnn1',
                                        )
                          )
        self.model.add(ELU())



        self.model.add(Convolution2D(    nb_filter= 48, 
                                         nb_row=5,
                                         nb_col=5, 
                                         name='cnn2',
                                         border_mode='valid',

                                        )
                          )
        self.model.add(ELU())

        self.model.add(Convolution2D(64, 3,3,name='cnn3'))
        self.model.add(ELU())

    
        self.model.add(Convolution2D(64, 3,3,name='cnn4'))
        self.model.add(MaxPooling2D( (2,2) ))
        self.model.add(ELU())
        # self.model.add(MaxPooling2D( (2,2) ))

        self.model.add(ActivityRegularization())

        #FC0
        self.model.add(Flatten(name='fc0_flatten'))
        self.model.add(Dense(1164,name='fc0'))
        self.model.add(Dropout(0.3))
        self.model.add(ELU())
        self.model.add(Dense(100,name='fc1'))
        self.model.add(ELU())
        self.model.add(Dense(50,name='fc2'))
        self.model.add(ELU())
        self.model.add(Dense(10,name='fc3'))
        self.model.add(ELU())

        #FC7
        self.model.add(Dense(1, name='fc7'))


        # self.Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # self.model.compile(loss='mean_squared_error', optimizer=self.Optimizer, metrics=['accuracy'])
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #use the default learning rate to follow drive.py


        print('cnn0: ', self.model.get_layer(name="cnn0").output_shape)
        print('cnn1: ', self.model.get_layer(name="cnn1").output_shape)
        print('cnn2: ', self.model.get_layer(name="cnn2").output_shape)
        print('cnn3: ', self.model.get_layer(name="cnn3").output_shape)
        print('cnn4: ', self.model.get_layer(name="cnn4").output_shape)

    def trainModel(self, epoch):
        generator_train = self.DatasetMgr.runBatchGenerator
        generator_vali = self.DatasetMgr.runValiBatchGenerator

        num_samples = self.DatasetMgr.getInputNum()
        num_vali_samples = len(self.DatasetMgr.getValidationY())
        # if generator == None:
        #     history = self.model.fit(self.X_Train , self.y_Train , nb_epoch=epoch, batch_size=64,  validation_split=0.2 ,shuffle=True, verbose = 2)
        # else:

        if num_samples< 64: # debugging
            history = self.model.fit_generator(generator_train(batch_size=1), num_samples, epoch, validation_data=generator_vali(), nb_val_samples=num_vali_samples )
        else:
            history = self.model.fit_generator(generator_train(batch_size=64), num_samples, epoch, validation_data=generator_vali(), nb_val_samples=num_vali_samples )

    def trainModel_SavePerEpoch(self, epoch):
        generator_train = self.DatasetMgr.runBatchGenerator
        generator_vali = self.DatasetMgr.runValiBatchGenerator
        num_samples = self.DatasetMgr.getInputNum()
        num_vali_samples = len(self.DatasetMgr.getValidationY())

        for i in range(epoch):
            if num_samples< 64: # debugging

                history = self.model.fit_generator(generator_train(batch_size=1), num_samples, 1, validation_data=generator_vali(), nb_val_samples=num_vali_samples )
            else:
                history = self.model.fit_generator(generator_train(batch_size=64), num_samples, 1, validation_data=generator_vali(), nb_val_samples=num_vali_samples )

            print("Epoch: ", i+1)    
            self.saveModel()



    def saveModel(self, str_model_name='model'):
        relative_path = 'checkpoints'

        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        dt_now = datetime.now(pytz.timezone('US/Eastern'))
        # dt_now = datetime.now(pytz.timezone('US/Pacific'))

        str_time = dt_now.strftime("%Y%m%d_%H%M%S")


        file_name = str_time + '_' + str_model_name 

        file_loc = os.path.join(relative_path, file_name) 

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file_loc+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_loc+".h5")
        print("Saved model to disk")

    def reloadModel(self, model_file_name):
        from keras.models import model_from_json

        with open(model_file_name, 'r') as jfile:
            self.model = model_from_json(jfile.read())
        self.model.compile("adam", "mse")
        weights_file = model_file_name.replace('json', 'h5')
        self.model.load_weights(weights_file)

    def debugModel(self):

        self.DatasetMgr.loadDataToMemory()

        np_images = self.DatasetMgr.getImg()

        steering_angle = float(self.model.predict(np_images, batch_size=1))
        print('Angle Prediction: ' , steering_angle)


def getArgs():
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--cfg', type=str, default="None", help='configuration commands')

    args = parser.parse_args()

    return args

def main():
    import time

    time_start = time.time()

    args = getArgs()

    racer_trainer = qModelTrainer(enable_incremental_learning=False, debug_size = None)
    # racer_trainer = qModelTrainer(enable_incremental_learning=False, debug_size = 2)

    

    if 'per_epoch' in args.cfg:
        racer_trainer.trainModel_SavePerEpoch(args.epoch)
    else:        
        racer_trainer.trainModel(args.epoch)
        racer_trainer.saveModel()

    # racer_trainer.debugModel()


    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    
if __name__ == "__main__": main()
