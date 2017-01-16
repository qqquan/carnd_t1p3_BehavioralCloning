import pandas as pd
import numpy as np
import cv2

IMG_SCALE = 1 

def normalizeImg(np_images):
    return (np_images-128.0)/255.0

# cut off upper part of image
def cropRoadImage(np_img, cutoff_ratio = 0.35):

    h = np_img.shape[0]
    h_new_begin = int(h*cutoff_ratio)

    return np_img[h_new_begin:h,:, :]


def image_trim(img_bgr):
    '''
    by Mengxi Wu

    resize and extract V  channel from color space to HSV

    Oputput 3d shape: (row, column, 1)
    '''
    trimed = img_bgr#[20:140]
#     resized = cv2.resize(img_bgr,(32,16))
    resized = cv2.resize((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))[:,:,1],(32,16))


    row, col = resized.shape
    img_prep = np.reshape(resized, (row, col, 1))
    return img_prep


# a_image is a BGR image
def prepImg(a_image, scale = IMG_SCALE):

    # a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2YUV)

    a_image = cropRoadImage(a_image)

    a_image = cv2.resize(a_image,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    # a_image = normalizeImg(a_image)

    return a_image

#drop angles that's less than a threshold
def prepAngle(np_angles):

    filter_threshold = 0.01 #0.25 degree
    ls_angle_filtered = [angle if np.absolute(angle)<filter_threshold else 0 for angle in np_angles] 
    np_angle_filtered = np.array(ls_angle_filtered)  

    return np_angle_filtered

def loadImgToNumpy(ls_img, img_scale, enable_aug_flip = False, enable_tiny_model=False):

    ls_image = []
    ls_image_flip = []
    for file_loc in ls_img:
        img = cv2.imread(file_loc)
        if True == enable_tiny_model:
            img_prep = image_trim(img)

        else:
            img_prep = prepImg(img, img_scale)

        if img_prep is not None:
            ls_image.append(img_prep)
            if enable_aug_flip == True:
                img_flip = np.fliplr(img_prep)
                ls_image_flip.append(img_flip)


    if enable_aug_flip == True:
        return np.array(ls_image), np.array(ls_image_flip)
    else:
        return np.array(ls_image)


def getCrossValiSets(X, y, training_len=0.75):

    import sklearn.model_selection

    X_Train, X_Vali, y_Train, y_Vali =  sklearn.model_selection.train_test_split(   X, 
                                                                                    y, 
                                                                                    train_size =training_len, 

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




def removeSmallValues(np_matrix, colum, thresh = 0.01, prob = 0.3 ):
    '''
    np_matrix: 2d array
    colum: if the element in column is less than threshold, then the row is removed from the matrix
    thresh: value less than threshold is considered small value
    prob: probablity of the actual removal 

    '''
    idx=0
    for row in np_matrix:
        if np.random.rand(1) < prob:
            if row[colum] < thresh:
                np_matrix = np.delete(np_matrix,idx, axis=0)
        else:
            idx+=1 #increment when there is no removal. the index is for the matrix after removal

        

    return np_matrix

class qDatasetManager:


    IDX_COL_CENTER_IMG = 0
    IDX_COL_LEFT_IMG = 1
    IDX_COL_RIGHT_IMG = 2
    IDX_COL_ANGLE = 3

    # offset_leftright_img: 0.04 - 1 degree; 0.1 - 2.5 degree
    def __init__(self, ls_file_loc, img_scale=IMG_SCALE, enable_aug_flip=True, offset_leftright_img = 0.1, debug_size = None, enable_tiny_model = False):
        print('Initialize qDatasetManager..')
        self.img_scale = img_scale
        self.enable_aug_flip = enable_aug_flip
        self.enable_tiny_model = enable_tiny_model
        self.offset_leftright_img = offset_leftright_img
       
        print('Read csv log file.. ') 
        col_indx = [qDatasetManager.IDX_COL_CENTER_IMG, qDatasetManager.IDX_COL_LEFT_IMG, qDatasetManager.IDX_COL_RIGHT_IMG, qDatasetManager.IDX_COL_ANGLE]

        ls_dataframes = []
        for a_cvs in ls_file_loc:
            df_sheet = pd.read_csv(a_cvs, skipinitialspace=True, usecols=col_indx, header=None )

            df_sheet_newpath = localizeImgPath(df_sheet, a_cvs)

            ls_dataframes.append(df_sheet_newpath)

        df_complete_records = pd.concat(ls_dataframes)

        self.np_sim_sheet = np.array([])
        self.np_sim_sheet = df_complete_records.values


        #remove row that has a small angle, because the image pattern of a straight lane are similar and are more repeatative than curve data 
        # self.np_sim_sheet = removeSmallValues(self.np_sim_sheet, colum = 3, thresh = 0.01, prob = 0.4 )


        if debug_size: #debug:
            self.np_sim_sheet = self.np_sim_sheet[0:debug_size]

        print('Augment data.. ')
        self.np_img_loc_augm , self.np_angle_augm  = self.augmentNumpyDataset()


        if debug_size == None:
            if True == self.enable_tiny_model:
                print('training split: ', 0.9)
                (self.np_img_loc_X_Train, self.np_img_loc_X_Vali, self.np_angle_y_Train , self.np_angle_y_Vali) = getCrossValiSets(self.np_img_loc_augm , self.np_angle_augm, training_len = 0.9)
            else:
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

        print('offset_leftright_img: ', self.offset_leftright_img)
        np_angle_offset = self.offset_leftright_img  #  0.04 - 1 degree; 
        np_angle_left = np_angle_center +  np_angle_offset #the left camera sees a image that requires right turn
        np_angle_right = np_angle_center - np_angle_offset

        np_angle_augm = np.concatenate([np_angle_center, np_angle_left, np_angle_right])

        return np_img_augm, np_angle_augm

    def runBatchGenerator(self, batch_size=64):


        np_xx_loc = self.getImgLocArray()


        np_yy = self.getY() # full reference output
        num_total = len(np_yy)

        if self.enable_aug_flip:
            batch_size =int(batch_size/2)  #half the size, because  flipping later doubles the actual size 
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

                if True == self.enable_aug_flip :
                    np_x, np_x_flip = loadImgToNumpy(ls_x_loc, self.img_scale,   enable_aug_flip= self.enable_aug_flip, enable_tiny_model= self.enable_tiny_model)
                else:
                    np_x = loadImgToNumpy(ls_x_loc, self.img_scale, enable_aug_flip = self.enable_aug_flip, enable_tiny_model=self.enable_tiny_model)



                np_y = np_yy[start_idx:end_idx]
                np_y_flip = np_y * (-1.0)

                if self.enable_aug_flip == True:
                    np_x = np.vstack((np_x, np_x_flip))
                    np_y = np.vstack((np_y, np_y_flip))


                yield (np_x, np_y)

    def runValiBatchGenerator(self, batch_size=int(64*0.1)): #TODO: currently 25% of dataset is allocated for validation. make it configurable.


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

                np_x = loadImgToNumpy(ls_x_loc, self.img_scale, enable_tiny_model=self.enable_tiny_model)

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
        
        # return prepAngle(np_angle_orig)
        return np_angle_orig
    
    def getCenterImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_CENTER_IMG]
    
    def getLeftImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_LEFT_IMG]
    
    def getRightImgLocList(self):
        return self.np_sim_sheet[:, qDatasetManager.IDX_COL_RIGHT_IMG]

    #return numpy array of [.., angle]                    
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
        a_img = loadImgToNumpy([self.np_sim_sheet[0,0]], self.img_scale, enable_tiny_model=self.enable_tiny_model )  #example for calculating the image shape
        return a_img.shape[1:]     

    def getInputNum(self):
        num_raw_input = len(self.getY())


        if self.enable_aug_flip == True:
            num = 2* num_raw_input  #flipping image doubles the number of images.
        else:
            num = num_raw_input

        return num

def main():

    TST_BatchSize = 2
    TST_SampleSize = 10 
    ls_records = [  
                    'recordings/rec9_udacity_1image/driving_log.csv',
                    'recordings/rec5_udacity/data/driving_log.csv',
                   # 'recordings/rec0/driving_log.csv',
                 ]
    dataset_mgr = qDatasetManager(ls_records, debug_size=TST_SampleSize)


    a = np.array([[1,0.2],[3,.4],[5,.006]])
    print('before removal: \n', a)

    b = removeSmallValues(a, colum = 1)
    print('np.random.rand(1) :', np.random.rand(1) )
    print('after removal: \n ', b)

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
        print("batch generator output - X shape and y shape: ", x.shape,y.shape)
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

    batch_size = 10
    training_data_gen = dataset_mgr.runBatchGenerator(batch_size)
    np_x, np_y = next(training_data_gen)
    idx_img = 2
    cv2.imshow(' image before flip ',np_x[idx_img])
    cv2.waitKey(0)
    cv2.imshow(' image after flip ',np_x[idx_img + batch_size])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    print('val_gen', val_gen)
    print('np_img_c[img_idx] max:' , np.max(np_img_c[img_idx]))

    next(validation_data)


if __name__ == "__main__": main()
