# Term1 Project: Behavioral Cloning

## Design

### Training Data

#### Training Content
  Normal driving that keeps the car in the center of the lane
  Counter-clockwise driving that train on the right turn capability
#### Corner Cases
##### Recovery
  Stop recording when the car drifts to the side, and start recording when the car is steering back to the lane center.
##### Trouble Spots
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