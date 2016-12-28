import pandas as pd
import numpy as np
import cv2

IMG_SCALE = 0.2 

def normalizeImg(np_images):

    return (np_images-128)/255

def cropRoadImage(np_img):

    h = np_img.shape[0]
    h_new_begin = int(h*0.3)

    return np_img[h_new_begin:h,:, :]


def prepImg(a_image, scale = IMG_SCALE):

    a_image = cropRoadImage(a_image)
    img_resi = cv2.resize(a_image,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    img_resi_norm = normalizeImg(img_resi)
    return img_resi_norm

#drop angles that's less than a threshold
def prepAngle(np_angles):

    filter_threshold = 0.01 #0.25 degree
    ls_angle_filtered = [angle if angle<filter_threshold else 0 for angle in np_angles] 
    np_angle_filtered = np.array(ls_angle_filtered)  

    return np_angle_filtered

def loadImgToNumpy(ls_img, img_scale):
    ls_image_data = []
    for file_loc in ls_img:
        img = cv2.imread(file_loc)
        img_prep = prepImg(img, img_scale)
        if img_prep is not None:
                ls_image_data.append(img_prep)

    return np.array(ls_image_data)


def getCrossValiSets(X, y, training_len=0.75):

    import sklearn.model_selection

    X_Train, X_Vali, y_Train, y_Vali =  sklearn.model_selection.train_test_split(  X, 
                                                                                    y, 
                                                                                    train_size =0.75, 

                                                                                    )


    return (X_Train, X_Vali, y_Train , y_Vali)


# adapt image locations from 3rd party recordings 
# relative_path: recording folder path relative to the model.py
# Option 1: IMG/
# Option 2: 3rd_party_path/IMG/
# return the numpy array of the true path relative to the local machine
def localizeImgPath(df_sheet, relative_csv_loc):
    import os

    csv_path_rel, _ = os.path.split(relative_csv_loc)

    img_path_rel = os.path.join(csv_path_rel, 'IMG/')


    img_loc_original =  df_sheet.get_value(0,0)

    img_path_original, _ = os.path.split(img_loc_original)


    img_path_new = img_path_rel


    df_sheet_new = df_sheet.replace( {img_path_original:img_path_new}, regex=True)

    return df_sheet_new



class qDatasetManager:


    IDX_COL_CENTER_IMG = 0
    IDX_COL_LEFT_IMG = 1
    IDX_COL_RIGHT_IMG = 2
    IDX_COL_ANGLE = 3

    def __init__(self, ls_file_loc, img_scale=IMG_SCALE , debug_size = None):
        
        self.img_scale = img_scale

        
        col_indx = [qDatasetManager.IDX_COL_CENTER_IMG, qDatasetManager.IDX_COL_LEFT_IMG, qDatasetManager.IDX_COL_RIGHT_IMG, qDatasetManager.IDX_COL_ANGLE]

        ls_dataframes = []
        for a_cvs in ls_file_loc:
            df_sheet = pd.read_csv(a_cvs, skipinitialspace=True, usecols=col_indx, header=None )

            df_sheet_newpath = localizeImgPath(df_sheet, a_cvs)

            ls_dataframes.append(df_sheet_newpath)

        df_complete_records = pd.concat(ls_dataframes)

        self.np_sim_sheet = np.array([])
        self.np_sim_sheet = df_complete_records.values


        if debug_size: #debug:
            self.np_sim_sheet = self.np_sim_sheet[0:debug_size]

        self.np_img_loc_augm , self.np_angle_augm  = self.augmentNumpyDataset()

        if debug_size == None:
            (self.np_img_loc_X_Train, self.np_img_loc_X_Vali, self.np_angle_y_Train , self.np_angle_y_Vali) = getCrossValiSets(self.np_img_loc_augm , self.np_angle_augm)
        else:
            #debug
            self.np_img_loc_X_Train =   self.np_img_loc_augm
            self.np_angle_y_Train = self.np_angle_augm 
            self.np_img_loc_X_Vali = np.array([])
            self.np_angle_y_Vali = np.array([])


        self.np_images_center = np.array([]) 
        self.np_images_left = np.array([]) 
        self.np_images_right = np.array([]) 
        self.X_Train = np.array([])
        self.y_Train = np.array([])

        self.X_Vali = np.array([])
        self.y_Vali = np.array([])



    def augmentNumpyDataset(self ):
        np_center_img = self.getCenterImgLocList()
        np_left_img = self.getLeftImgLocList()
        np_right_img = self.getRightImgLocList()

        np_img_augm = np.concatenate([np_center_img, np_left_img, np_right_img])


        np_angle = self.getSteeringAngleList()
        np_angle_center = np_angle
        np_angle_offset = 0.04 # 1 degree:  0.04 - 1 degree; 
        np_angle_left = np_angle_center +  np_angle_offset #the left camera sees a image that requires right turn
        np_angle_right = np_angle_center - np_angle_offset

        np_angle_augm = np.concatenate([np_angle_center, np_angle_left, np_angle_right])

        return np_img_augm, np_angle_augm

    def runBatchGenerator(self, batch_size=64):

        num_total = self.getInputNum()

        np_xx_loc = self.getImgLocArray()


        np_yy = self.getY() # full reference output

        batch_start_idx = range(0, num_total, batch_size)

        while True:
            for start_idx in batch_start_idx:

                ls_x_loc = []
                end_idx = 0
                if (start_idx+batch_size) < num_total:
                    end_idx = start_idx+batch_size

                elif (num_total > batch_size):
                    end_idx = num_total # take whatever left, although having fewer than batch_size
                elif (num_total <= batch_size):
                    end_idx = batch_size

                ls_x_loc = np_xx_loc[start_idx:end_idx]

                np_x = loadImgToNumpy(ls_x_loc, self.img_scale)

                np_y = np_yy[start_idx:end_idx]

                yield (np_x, np_y)

    def runValiBatchGenerator(self, batch_size=int(64*0.25)): #TODO: currently 25% of dataset is allocated for validation. make it configurable.


        np_xx_loc = self.np_img_loc_X_Vali


        np_yy = self.getValidationY() # full reference output
        num_total = len(np_yy) #TODO: use a private constant to store num of validation points

        batch_start_idx = range(0, num_total, batch_size)

        while True:
            for start_idx in batch_start_idx:

                ls_x_loc = []
                end_idx = 0
                if (start_idx+batch_size) < num_total:
                    end_idx = start_idx+batch_size

                elif (num_total > batch_size):
                    end_idx = num_total # take whatever left, although having fewer than batch_size
                elif (num_total < batch_size):
                    end_idx = batch_size


                ls_x_loc = np_xx_loc[start_idx:end_idx]

                np_x = loadImgToNumpy(ls_x_loc, self.img_scale)

                np_y = np_yy[start_idx:end_idx]

                yield (np_x, np_y)




    def loadDataToMemory(self):

        ls_center_img = self.getCenterImgLocList()
        ls_left_img = self.getLeftImgLocList()
        ls_right_img = self.getRightImgLocList()

        self.np_images_center = loadImgToNumpy(ls_center_img, self.img_scale) 
        self.np_images_left = loadImgToNumpy(ls_left_img, self.img_scale) 
        self.np_images_right = loadImgToNumpy(ls_right_img, self.img_scale) 

        np_X_raw_norm = self.np_images_center #prep'ed during loading.  prepImg(self.np_images)
        np_y_raw_norm = self.getY()

        # TODO: error on stratification. to find how to transform y_raw for regression stratification
        # self.X_Train, self.X_Vali, self.y_Train, self.y_Vali = getCrossValiSets(np_X_raw_norm, np_y_raw_norm)
        self.X_Train = np_X_raw_norm
        self.y_Train =np_y_raw_norm

        self.X_Vali = np.array([])
        self.y_Vali = np.array([])


    def getImgScale(self):
        return self.img_scale

    def getSteeringAngleList(self):
        np_angle_orig = self.np_sim_sheet[:, qDatasetManager.IDX_COL_ANGLE]
        return prepAngle(np_angle_orig)
    
    def getCenterImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_CENTER_IMG]
    
    def getLeftImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_LEFT_IMG]
    
    def getRightImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_RIGHT_IMG]

                        
    def getY(self):
        angle_list = self.np_angle_y_Train
        ls_angle_dim_correction = np.expand_dims(angle_list, axis=1)
        return ls_angle_dim_correction

    def getImgLocArray(self):
        return self.np_img_loc_X_Train

    def getImg(self):

        #TODO: add preImages(np_images) to loop through each img - np_images[i,:,:,:]
        return self.np_images_center, self.np_images_left, self.np_images_right

    def getTrainingX(self):
        return self.X_Train

    def getTrainingY(self):
        return self.y_Train

    def getValidatoinX(self):
        return self.X_Vali


    #return validation angles of shape (none,num_of_elements)
    def getValidationY(self):

        np2d_ValiY = np.expand_dims(self.np_angle_y_Vali, axis=1)
        return np2d_ValiY

    def getInputShape(self): 
        a_img = loadImgToNumpy([self.np_sim_sheet[0,0]], self.img_scale)  #example for calculating the image shape
        return a_img.shape[1:]     

    def getInputNum(self):
        return len(self.getY())


def main():

    TST_BatchSize = 2
    TST_SampleSize = 10 
    ls_records = [  
                    'recordings/rec9_udacity_1image/driving_log.csv',
                    'recordings/rec5_udacity/data/driving_log.csv',
                   # 'recordings/rec0/driving_log.csv',
                 ]
    dataset_mgr = qDatasetManager(ls_records, debug_size=TST_SampleSize)

    dataset_mgr.loadDataToMemory()


    np_img_c, np_img_l, np_img_r = dataset_mgr.getImg()

    img_idx = 0

    cv2.imshow('center image',np_img_c[img_idx])
    cv2.waitKey(0)
    cv2.imshow('left image',np_img_l[img_idx])
    cv2.waitKey(0)
    cv2.imshow('right image',np_img_r[img_idx])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = np_img_c[img_idx]

    print('image shape: ', img.shape)

    X_Train = dataset_mgr.getTrainingX()
    y_Train = dataset_mgr.getTrainingY()
    X_Vali  = dataset_mgr.getValidatoinX()
    y_Vali  = dataset_mgr.getValidationY()

    print('Training Shape: {}   {}'.format(X_Train.shape, y_Train.shape ))
    print('Validation Shape: {}   {}'.format(X_Vali.shape, y_Vali.shape ))
    print('Input Shape: ', dataset_mgr.getInputShape())

    print('max(Y): ' ,max(y_Train))
    print('min(Y): ' ,min(y_Train))
    print('Num of Inputs: ', dataset_mgr.getInputNum())

    assert( TST_SampleSize == (X_Train.shape[0] + X_Vali.shape[0]))
    assert(len(dataset_mgr.getInputShape()) == 3)

   
    test_count = 0
    for x,y in dataset_mgr.runBatchGenerator(TST_BatchSize):
        print("batch generator: ", x.shape,y.shape)
        assert(x.shape[0] == y.shape[0])
        assert (x.shape[0] > 0)
        
        test_count += 1
        if (test_count > (dataset_mgr.getInputNum()/TST_BatchSize)):
            break #get out of infinate generator after all samples are debugged

    TST_img_idx = 1 
    np_img_loc_array = dataset_mgr.getImgLocArray()
    print(" img: ", np_img_loc_array[TST_img_idx ])

    np_angle = dataset_mgr.getY()
    print(" angle: ", np_angle[TST_img_idx ])
 

    print("Image Scale: ", dataset_mgr.getImgScale())


    validation_data = dataset_mgr.runValiBatchGenerator()
    # validation_data = dataset_mgr.runBatchGenerator

        # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__'))


    print('val_gen', val_gen)


    next(validation_data)


if __name__ == "__main__": main()
