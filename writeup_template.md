# **Behavioral Cloning**  
---
**Behavioral Cloning Project**   

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_architecture.png "Model Architecture"
[image2]: ./training_stages.png "Training Stages"
[image3]: ./examples/center.png "Recovery Image"
[image4]: ./examples/center_translated.png "Recovery Image"
[image5]: ./examples/left.png "Recovery Image"
[image6]: ./examples/right.png "Normal Image"
[image7]: ./examples/center_flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The same simulator, with changed speed can be run by executing:
```sh
python drive-fast.py model.h5
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach and final Model Architecture

My first step was to use a convolution neural network model similar to the NVIDIA model published in the paper : End to End Learning for Self-Driving Cars (https://arxiv.org/pdf/1604.07316v1.pdf). The purpose of this network is optimized for the exactly same problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting, which was expected since the original model is used with a huge ammount of data. Therefore the size of the model was greatly reduced and some layers of dropout were used.

The final model architecture can be seen below:
![alt text][image1]

Firstly the input image is normallized and cropped to remove unneeded information.
Then The basic architecture of NVIDIA model is used. However in the first 3 convolutional filters, the depth is greatly reduced. After these 3 filters, a dropout layer is used to reduce overfitting. Normally, 2 more convolutional layers follow up, but one of these was completely removed. Afterward one more dropout layer is used. Lastly, the fully connected (dense) layers follow the pipeline. However, the first big layer of 1164 neurons is completely removed, to reduce overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains 2 Dropout layers after the convolutional layers to reduce overfitting. Any attempt to use dropout layers between the fully connected layers resulted in worse results.

The final model was also trained in stages and using the validation loss as a guide, the proper number of epochs were chosen to avoid overfitting. 
![alt text][image2]
A very good balance betweeen accurancy and overfitting was achieved at model stage 4, after 6 epochs. Indeed the models were tested and this model had the best performance. Therefore 6 epochs was chosen on the final model.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### 4. Appropriate training data & data augmentation

The training data used were the data provided by Udacity. The ammount of data was not enough to properly train the model. This dataset included left and right camera captured images, which were used to increase the ammount of data. These were needed to have the steering angle adjusted and the value of 0.25 was found to work well.

The original dataset is not balanced, since the circuit has many left turns and the car is driven mostly relatively straight. To balance the set, approximately 50% of the generated set has the image and the angle flipped (to avoid the left-turning bias). Also A filter was used to reduce the number of data where the car was driving relatively straight.

To augment the dataset with new data and avoid manually recording new data, I used random image translation combined with a python generator. 
To calculate the value of the angle when translating the image, the guideline of Assist.Prof Vivek Yadav was used (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). Therefore 0.004 steering angle units per pixel were added or subtracted per pixel shift when shifting the image right or left accordingly.

Following are examples of the augmentation process:

Original center image, with steering angle 0.0904655:   
![alt text][image3]   
Translated image, new steering angle -0.0986641488061542:   
![alt text][image4]   
Left camera image, steering angle 0.3404655:   
![alt text][image5]   
Right camera image, steering angle -0.1595345:   
![alt text][image6]   
Flippled center image, steering angle -0.0904655:   
![alt text][image7]   

The use of the generator was vital, since it allows the unlimited generation of new data (with the use of the image translation) without the actual need of pre-processing the data.

The final video files are run1 and run1-fast. The second video file has the speed tweaked so that it runs faster.


