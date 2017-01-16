# Self-driving Car Nanodegree Term1 Project 3 : Behavioral Cloning


## Model Architecture

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


## Training Strategy
1. The base dataset used is from Udacity, so the test result can be easily communicated among the community.
2. Try established steering training models, such as Nvida model and comma.ai model. The preliminary result is that car only drive straight for a short time, then start weaving. The hypothesis is that the dataset is not big enough for those two complex models. So we can either add more training data, or reduce the model complexity.
3. Based on evaluation of the simulation conditions, there are relatively few features, such as clear lane mark or road edge. The graphic is relatively simple pattern, so a simple Machine Learning network should be able to handle.
4. Starting from the basic one CNN network, the performance is not good. Additional layer is added until that the performance shows the model can handle both straight lane and curve. In the end, two layers of CNN is deemed to be sufficient to recognize road side edges and lane marks. 
5. Tune hyper-parameters of the chosen model.


## Design

### Training Data

#### Training Content
- Normal driving that keeps the car in the center of the lane. It also contains training data to teach left-right turns.
- Counter-clockwise driving that train on the right turn capability

#### Corner Cases
- Recovery:  
 
  Stop recording when the car drifts to the side, and start recording when the car is steering back to the lane center.
 
- Trouble Spots:  
 
  The car is mostly trained on the concrete road condition. Other road condition such as dirt or stone bridge has different tecture on the surface, and has a smaller portion in the training dataset. Additional dataset on those trouble spot would help the model learn those corner cases of driving conditions.

#### Data Augmentation
##### Left-right Camera Compensation
##### Horizontal Flip

#### Color Space

#### Normalization 

### Model Architecture
#### Number of CNN layers
#### Number of Fully Connected(FC) Layers
#### Activation type: RELU vs ELU
#### Number of Batch Normalization Layers

### Hyper-parameters

#### Batch size
#### Epoch number
#### Number of Layers
#### CNN kernal size
#### CNN stride 
#### Random seed


## Trained Models
  The similator excutes the same model differently every time it loads the same saved model and weights. The following are the ones found finishing the track during training. 
  ### Basic 1 
  ### Basic 2
  ### Tiny 1