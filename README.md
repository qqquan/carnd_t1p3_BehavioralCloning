# CarND Project 3 : Behavioral Cloning


## 1. Model Architecture

Two Convolutional Network(CNN) Layers are used, following by one Fully Connected(FC) layer.

----------------------------------------------------------------------------------------------------
Layer (type)                     |Output Shape          |Param #     |Connected to                     
---------------------------------|----------------------|------------|---------------------------------
lambda_1 (Lambda)                |(None, 104, 320, 3)   |0           |lambda_input_1[0][0]             
convolution2d_1 (Convolution2D)  |(None, 12, 36, 16)    |5824        |lambda_1[0][0]                   
activation_1 (Activation)        |(None, 12, 36, 16)    |0           |convolution2d_1[0][0]            
convolution2d_2 (Convolution2D)  |(None, 3, 9, 16)      |9232        |activation_1[0][0]               
activation_2 (Activation)        |(None, 3, 9, 16)      |0           |convolution2d_2[0][0]            
dropout_1 (Dropout)              |(None, 3, 9, 16)      |0           |activation_2[0][0]               
flatten_1 (Flatten)              |(None, 432)           |0           |dropout_1[0][0]                  
dense_1 (Dense)                  |(None, 1)             |433         |flatten_1[0][0]                  


Total params: 15,489; 
Trainable params: 15,489; 
Non-trainable params: 0.


## 2. Training Strategy
1. The base dataset used is from Udacity, so the test result can be easily communicated among the community.
2. Try established steering training models, such as Nvida model and comma.ai model. The preliminary result is that car only drive straight for a short time, then start weaving. The hypothesis is that the dataset is not big enough for those two complex models. So we can either add more training data, or reduce the model complexity.
3. Based on evaluation of the simulation conditions, there are relatively few features, such as clear lane mark or road edge. The graphic is relatively simple pattern, so a simple Machine Learning network should be able to handle.
4. Starting from the basic one CNN network, the performance is not good. Additional layer is added until that the performance shows the model can handle both straight lane and curve. In the end, two layers of CNN is deemed to be sufficient to recognize road side edges and lane marks. 
5. Tune hyper-parameters of the chosen model.


## 3. Design and Tuning

The following discusses the trade-offs and personal thoughts behind network designs and hyper-parameter tuning.


### 3.1 Data Collection and Preprocessing

#### Training Content
- Normal driving that keeps the car in the center of the lane. It also contains training data to teach left-right turns.
- Counter-clockwise driving that train on the right turn capability

#### Corner Cases
- Recovery:  
 
  Stop recording when the car drifts to the side, and start recording when the car is steering back to the lane center.
 
- Trouble Spots:  
 
  The car is mostly trained on the concrete road condition. Other road condition such as dirt or stone bridge has different tecture on the surface, and has a smaller portion in the training dataset. Additional dataset on those trouble spot would help the model learn those corner cases of driving conditions.

#### Data Augmentation
- Left-right Camera Compensation

Two side cameras are mounted at two sides of the windshield. What left camera sees is simulating that the car leans on left. This requires additional steering to left. Same argument applies for the right camera. The extra steering required from left or right is an offset value, which is a hyper-parameter and requires tuning.

- Horizontal Flip

The normal track consists most of left-turn curves. In order to augment training data for right-turn curve situations, images are flipped horizontally from left to right.
- Color Space

RGB, HSV, YUV can offer different properties for the image recognition capability of the model. Further study is needed to investigate the performance difference. The final solution of this report uses RGB. 

#### Data Normalization 
In order to have zero-mean and small-variance input data, the image is normalized to (-0.5, 0.5) with the following formula: x/127.5 - 1.0

### 3.2 Model Architecture
- Number of CNN layers

More layers of CNN handles more complex data, while requires more training examples. 

- Number of Fully Connected(FC) Layers

Depending on the number of extracted features and learning target, more FC layer handles more difficult target. Because the simulation environment is relatively simple, single FC layer that matches to the single regression value.

- Dropout

A simple network architecture can easily get overfitted. Dropout() discards data during training and compensate neurons at prediction. It offers good capability to prevent overfitting, and is suitable for this simple network model.

- Activation type: RELU vs ELU

ELU promises faster learning, but the regression seems worse during testing. Relu is used for steady performance.

- Number of Batch Normalization Layers

Batch Normalization promises faster learning, but the regression seems extremely worse during testing. The guess is that dataset and model are not large enough for Batch Normalization to become effective. It is not used.

### 3.3 Hyper-parameters

- Batch size

Smaller batch size provides more back-prop iterations and can provide faster learning. The trade-off is that a larger batch size may have a better view for back-prop to descend and reduces possibility of local optimal point. A proper value depends on the dataset and the learning problem. In the final model and dataset, 128 seems provide a steady performance of the training result.

- Epoch number

Too large the number, the training overfits the model; too small, the model is under-trained. Saving model after each epoch alleviates the problem, but is very time consuming. The final model is from epoch 36.

- CNN kernal size

It depends on the complexity of useful features in the image or data input. Because the simulation environment is not complex, and the feature extraction only needs to differentiate lines from other shapes, a large kernel that grosses over data is sufficient. 

- CNN stride 

The testing shows the performance is sensitive to the stride size. One step less or more shows a big difference in performance. Reason is unclear. The value is hand-tuned carefully and painstakingly.

- Random seed

This is another hyper-parameter to tune or do trail-error. It affects weights&bias initialization, and has a great impact on the final performance in the testing.


## 4. Code Architecture

The SW design splits training and data-processing into two separate modules: model.py, util_qDatasetManager.py. The simulation interfacing is in drive.py. 

Scalability and portability is a key focus on the SW architecture design. The code runs on either AWS server or personal computer transparently. Even though the recorded dataset uses absolute local full path, the DatasetManager converts all to relative paths for the local machine. Therefore, additional recovery data can easily be stacked to a path list, and the code can run on both AWS and computer with a same dataset package. Hyper-parameter tunning is also supported through command-line arguments.

- model.py

    - Define network architecture
    - Provide command line arguments for tuning batch_size, epoch, left-right image offset, etc

- util_qDatasetManager.py

    - Load dataset
    - Pre-process data
    - Provide Image generator for Keras to load image to memory on the fly.

- drive.py

    - Simulate the trained model on simulator.

- model.json

    - Trained model.

- model.h5

    - Trained weights.

