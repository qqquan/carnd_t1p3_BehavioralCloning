import argparse
import numpy as np
from util_qDatasetManager import qDatasetManager
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout, ELU, ActivityRegularization, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras
from time import strftime
from datetime import timedelta, datetime
import pytz
import os
from keras.optimizers import Adam


class PerEpochSave(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return 
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        relative_path = 'checkpoints'
        str_model_name = 'model'

        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        dt_now = datetime.now(pytz.timezone('US/Eastern'))
        # dt_now = datetime.now(pytz.timezone('US/Pacific'))

        str_time = dt_now.strftime("%Y%m%d_%H%M%S")


        file_name = str_model_name+ '_'+ str_time + '_'+'Epoch'+ str(epoch+1)

        file_loc = os.path.join(relative_path, file_name) 

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file_loc+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_loc+".h5")
        print("\n Saved model to disk: ", file_name ,'\n')

        return

class qModelTrainer:
    '''
    lr_offset: angle offset for left-right images
    '''

    def __init__(self, input_file_loc = None, enable_incremental_learning = False, debug_size = None, enable_aug_flip = True, batch_size = 255, enable_tiny_model = False, lr_offset = 0.1):

        if enable_incremental_learning:
            # new learning materials

            ls_records = [  
                            'recordings/rec13_sideDirt1/driving_log.csv',
                            # 'recordings/rec8_1stCurve_LeftRecov/driving_log.csv',
                            # 'recordings/rec6_2ndCurve/driving_log.csv',
                            # 'recordings/rec4_recovery/driving_log.csv',
                            # 'recordings/rec5_udacity/data/driving_log.csv',
                            

                         ]  
        else:
             ls_records = [  
                            'recordings/rec24_dirtSide2/driving_log.csv',
                            # 'recordings/rec13_sideDirt1/driving_log.csv',
                            # 'recordings/rec22_rightTurn4/driving_log.csv',
                            # 'recordings/rec23_after1stTurn2/driving_log.csv',
                            # 'recordings/rec16_troubleSpots/driving_log.csv',
                            # 'recordings/rec18_rightTurn/driving_log.csv',
                            # 'recordings/rec19_rightTurn2/driving_log.csv',
                            # 'recordings/rec21_rightTurn3/driving_log.csv',
                            # 'recordings/rec17_troubl_dirt/driving_log.csv',
                            # 'recordings/rec20_after1stTurn/driving_log.csv',
                            # 'recordings/rec15_MentorSD/driving_log.csv',
                            # 'recordings/rec11_backwardTrack/driving_log.csv',
                            # 'recordings/rec14_backTrack3/driving_log.csv',
                            # 'recordings/rec10_right_turn/driving_log.csv',
                            # 'recordings/rec3_finer_steering/driving_log.csv',
                            # 'recordings/rec2_curve/driving_log.csv',
                            'recordings/rec5_udacity/data/driving_log.csv',
                         ]  
        print('Load dataset: ', ls_records)
        self.DatasetMgr = qDatasetManager(ls_records, debug_size = debug_size, enable_aug_flip = enable_aug_flip, offset_leftright_img = lr_offset, enable_tiny_model = enable_tiny_model)

        self.InputShape = self.DatasetMgr.getInputShape()
        self.batch_size = batch_size
        self.path_model_checkpoints = 'checkpoints'

        if enable_incremental_learning:
            self.reloadModel('model.json')
        elif enable_tiny_model:
            self.model = Sequential()
            self.buildModel_tiny()
        else:
            self.model = Sequential()
            self.buildModel_basic()

        # #visualize the model
        # import pydot
        # from keras.utils.visualize_util import plot
        # plot(self.model, to_file='model.png')

        self.clearSavedModels()

    def buildModel_basic(self):

        print('Build the basic model..')

        self.model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=self.InputShape, output_shape=self.InputShape))
        
        self.model.add(Convolution2D(16, 11, 11, subsample=(9, 9),  border_mode="same"))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(16, 6, 6, subsample=(4, 4), border_mode="same"))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(.5))
        self.model.add(Flatten())
        
        self.model.add(Dense(1))

        self.Optimizer = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=self.Optimizer , loss="mse")    
            
        self.model.summary() 

    def buildModel_tiny(self):
        '''
            from Mengxi Wu
        '''
        print('Build the tiny model..')

        self.model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=self.InputShape))
        self.model.add(Convolution2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
        self.model.add(MaxPooling2D((4,4),(4,4),'valid'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error',optimizer='adam')

        self.model.summary()

    def buildModel_Desoto(self):
        '''
            from Stewart Desoto
        '''
        print('Build the Desoto model..')

        self.model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=self.InputShape, output_shape=self.InputShape))

        #scale image from 160x320 to 32x64
        self.model.add(Convolution2D(nb_row=1, nb_col=1, border_mode='valid', 
                                nb_filter=3, init='normal',
                               subsample=(5,5),
                                name='subsample'))
        # 5x5 with 2x2 striding          
        self.model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='valid', 
                                nb_filter=24,
                                   activation='relu',
                                  init='normal'))
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization())

        self.model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='valid', 
                                nb_filter=36,
                                   activation='relu',
                                  init='normal'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(BatchNormalization())

        # Nvidia model includes 3rd conv layer, which we don't use
        # model.add(Convolution2D(nb_row=5, nb_col=5, border_mode='same', 
        #                         nb_filter=48,
        #                            activation='relu', subsample=(4,4), 
        #                           init='normal'))
        # model.add(BatchNormalization())

        # 3x3 with no striding
        self.model.add(Convolution2D(nb_row=3, nb_col=3, border_mode='valid', 
                                nb_filter=64,
                                   activation='relu',  
                                  init='normal'))
        self.model.add(BatchNormalization())

        self.model.add(Convolution2D(nb_row=3, nb_col=3, border_mode='valid', 
                                nb_filter=64,
                                   activation='relu',  
                                  init='normal'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten(name='flatten'))

        self.model.add(Dense(output_dim=1164, init='normal', activation='relu'))
        self.model.add(Dropout(p=0.5))
        self.model.add(BatchNormalization())

        self.model.add(Dense(output_dim=100, init='normal', activation='relu'))
        self.model.add(Dropout(p=0.5))
        self.model.add(BatchNormalization())

        self.model.add(Dense(output_dim=50, init='normal', activation='relu'))
        self.model.add(Dropout(p=0.5))
        self.model.add(BatchNormalization())

        self.model.add(Dense(output_dim=10, init='normal', activation='relu'))
        self.model.add(Dropout(p=0.5))
        self.model.add(BatchNormalization())

        self.model.add(Dense(output_dim=1, name='output', init='normal'))
        
        self.model.compile(loss='mse', metrics=['mse'], optimizer=Adam())
        self.model.summary() 

    def buildModel_nvidia(self):
        '''
            from http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        '''
        print('Build the nvidia model..')


        self.model.add(BatchNormalization(input_shape=self.InputShape))

        self.model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  name='cnn0',border_mode='valid',))
        self.model.add(ELU())

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(36, 5,5, name='cnn1', border_mode='valid'))
        self.model.add(ELU())


        self.model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  name='cnn2', border_mode='valid' ) )
        self.model.add(ELU())

        self.model.add(Dropout(0.3))

        self.model.add(Convolution2D(64, 3,3,name='cnn3', border_mode='valid'))
        self.model.add(ELU())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        self.model.add(Convolution2D(64, 3,3,name='cnn4', border_mode='valid'))
        self.model.add(ELU())

        #FC0
        # self.model.add(BatchNormalization())
        self.model.add(Flatten(name='fc0_flatten'))
        
        self.model.add(Dense(100,name='fc1'))
        self.model.add(ELU())
        self.model.add(Dropout(.5))

        self.model.add(Dense(50,name='fc2'))
        self.model.add(ELU())

        self.model.add(Dropout(0.3))

        self.model.add(Dense(10,name='fc3'))
        self.model.add(ELU())

        #FC7
        self.model.add(Dense(1, name='fc7'))

        self.Optimizer = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(loss='mean_squared_error', optimizer=self.Optimizer, metrics=['acc'])


        # self.model.compile(loss='mse', optimizer='adam') 
        self.model.summary() 

    def buildModel_commaai(self):
        '''
            from https://github.com/commaai/research/blob/master/train_steering_model.py
        '''
        print('Build the comma.ai model..')

        self.model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=self.InputShape, output_shape=self.InputShape))
        
        self.model.add(Convolution2D(16, 8, 8, subsample=(5, 5),  border_mode="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 6, 6, subsample=(4, 4), border_mode="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        
        
        self.model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        
        self.model.add(Dropout(.3))
        
        self.model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Activation('relu'))

        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(.5))
        
        self.model.add(Dense(1))

        self.Optimizer = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=self.Optimizer , loss="mse")    
            
        self.model.summary() 

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
            history = self.model.fit_generator(generator_train(batch_size=self.batch_size), num_samples, epoch, 
                                                validation_data=generator_vali(), nb_val_samples=num_vali_samples,  
                                                callbacks = [PerEpochSave()])

    def trainModel_SavePerEpoch(self, epoch):
        generator_train = self.DatasetMgr.runBatchGenerator
        generator_vali = self.DatasetMgr.runValiBatchGenerator
        num_samples = self.DatasetMgr.getInputNum()

        num_vali_samples = len(self.DatasetMgr.getValidationY())

        for i in range(epoch):
            if num_samples< 64: # debugging

                history = self.model.fit_generator(generator_train(batch_size=1), num_samples, 1, validation_data=generator_vali(), nb_val_samples=num_vali_samples )
            else:
                history = self.model.fit_generator(generator_train(batch_size=self.batch_size), num_samples, 1, validation_data=generator_vali(), nb_val_samples=num_vali_samples )

            print("Epoch: ", i+1)    
            self.saveModel()



    def saveModel(self, str_model_name='model'):
        relative_path = self.path_model_checkpoints 

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
        print("Saved model to disk", file_name)

    def reloadModel(self, model_file_name):
        from keras.models import model_from_json

        with open(model_file_name, 'r') as jfile:
            self.model = model_from_json(jfile.read())
        self.model.compile("adam", "mse")
        weights_file = model_file_name.replace('json', 'h5')
        self.model.load_weights(weights_file)

    def debugModel(self):

        self.DatasetMgr.loadDataToMemory()

        np_img_c, np_img_l, np_img_r = self.DatasetMgr.getImg() #todo: fix tuple return

        exp_angles = self.DatasetMgr.getY()

        img_idx = 0  #TODO: only support zero, because getY() returns a stacked angle list, while getImg returns separate side images.

        img = np_img_c[None,img_idx,:] # maintain the required dimension as model input
        steering_angle = float(self.model.predict(img, batch_size=1))
        print('Angle Prediction(center): ' , steering_angle)

        img = np_img_l[None,img_idx,:]
        steering_angle = float(self.model.predict(img, batch_size=1))
        print('Angle Prediction(left): ' , steering_angle)

        img = np_img_r[None,img_idx,:]
        steering_angle = float(self.model.predict(img, batch_size=1))
        print('Angle Prediction(right): ' , steering_angle)        


    def clearSavedModels(self):
        import glob, os
        relative_path = self.path_model_checkpoints 

        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        file_pattern = os.path.join(relative_path, "*.*")
        filelist = glob.glob(file_pattern)
        for f in filelist:
            os.remove(f)


def getArgs():
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=255, help='Batch Size.')
    parser.add_argument('--lr_offset', type=float, default=0.1, help='offset value to compensate left-right image flipping.')
    parser.add_argument('--cfg', type=str, default="None", help='configuration commands')
    parser.add_argument("--increm", default=False, action="store_true" , help="enable incremental learning on top of a trained model")
    parser.add_argument("--tiny", default=False, action="store_true" , help="enable tiny model")
    parser.add_argument("--no_flip", default=False, action="store_true" , help="enable incremental learning on top of a trained model")
    parser.add_argument('--seed', type=int, default=111111, help='seed to avoid randomness in Numpy random generator.')
    
    args = parser.parse_args()

    return args

def main():
    import time

    time_start = time.time()

    args = getArgs()

    np.random.seed(args.seed) 
       
    if args.no_flip:
        enable_flip = False
    else:
        print('Enable image flip in preprocessing.. ')
        enable_flip = True

    if args.epoch == None:
        print('Enter debugging mode - only a small subset of data is used' )
        racer_trainer = qModelTrainer(enable_incremental_learning=False, debug_size = 3, lr_offset = args.lr_offset )
        # racer_trainer = qModelTrainer(enable_incremental_learning=False, debug_size = 2)

        epochs = 15 # sanity check if we can overfit a small example set --> check if loss is driven to zero
        print('Expected Steering Angle: \n', racer_trainer.DatasetMgr.getY())

        for epo in range(epochs):
            generator_train = racer_trainer.DatasetMgr.runBatchGenerator
            num_samples = racer_trainer.DatasetMgr.getInputNum()
            print('epo: ', epo)
            racer_trainer.model.fit_generator( generator_train(batch_size=2), num_samples, 1 )

            print('Epoch: ', epo+1)
            racer_trainer.debugModel()

    elif args.increm:
        print('Enable Incremental Learning Method..')
        racer_trainer = qModelTrainer(enable_incremental_learning=True, debug_size = None, batch_size = args.batch_size, lr_offset = args.lr_offset )    
        racer_trainer.trainModel_SavePerEpoch(args.epoch)

    elif args.tiny:
        print('Enable tiny model..')
        print('Batch Size: ',args.batch_size)
        racer_trainer = qModelTrainer(enable_tiny_model=True, debug_size = None, batch_size = args.batch_size, enable_aug_flip= enable_flip,
                     lr_offset = 0.3 )    
                     # lr_offset = args.lr_offset )    
        racer_trainer.trainModel(args.epoch) 

    else:
        #normal training


        racer_trainer = qModelTrainer(enable_incremental_learning=False, enable_aug_flip= enable_flip, batch_size = args.batch_size, lr_offset = args.lr_offset )    

        if 'per_epoch' in args.cfg:
            racer_trainer.trainModel_SavePerEpoch(args.epoch)
        else:        
            racer_trainer.trainModel(args.epoch)
            racer_trainer.saveModel()

    # racer_trainer.debugModel()


    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
    
if __name__ == "__main__": main()


# self.np_img_loc_X_Train:  ['recordings/rec9_udacity_1image/IMG//center_2016_12_01_13_40_46_798.jpg'
#  'recordings/rec9_udacity_1image/IMG//left_2016_12_01_13_40_46_798.jpg'
#  'recordings/rec9_udacity_1image/IMG//right_2016_12_01_13_40_46_798.jpg']
# self.np_angle_y_Train:  [-0.3825653 -0.3425653 -0.4225653]