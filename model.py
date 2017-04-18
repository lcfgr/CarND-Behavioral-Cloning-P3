
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import random

# Reads csv
def read_data_csv(data_directory, csv_filename, data):
    with open(data_directory+csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Create a new table, by extracting the left and right camera images and compensating steering angle
def separate_data_csv(data, correction = 0.20):
    separated_data = []
    for row in data:
        center, left, right, steering, throttle, brake, speed = row
        separated_data.append((data_directory+row[0].strip(),float(steering)))
        if correction !=0:
            separated_data.append((data_directory+row[1].strip(), (float(steering)+correction)))
            separated_data.append((data_directory+row[2].strip(), (float(steering)-correction)))           
    return separated_data

# Create a flipped image (with probability 0.5) and calculate the appropriate angle
def random_flip(img, angle):
    if random.random()<0.5:
        img = cv2.flip(img,1)
        angle = -1*angle
    return img, angle

# Translate image randomly. The ratio of pixel shift was acquired from
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def translate_img(img,angle,translate_range = 100):
    rows, cols, _ = img.shape
    translate_x = translate_range*np.random.uniform()-translate_range/2
    angle = angle + translate_x/translate_range*0.4
    translate_y = 40*np.random.uniform()-20
    Translate_Matrix = np.float32([[1,0,translate_x],[0,1,translate_y]])
    img  = cv2.warpAffine(img,Translate_Matrix,(cols,rows))
    return img,angle

# Any image pre-processing should go here
def process_img(img):
    return img

def generator(data_log, batch_size = 32):
    count = 0
    images = []
    angles = []
    while True: # Loop forever so the generator never terminates
        data_log = shuffle(data_log)
        for img_filename, steering in data_log:
            # Reject 80% of smaller angles, since we have many straight going images
            if steering < 0.5 and random.random()<0.8:
                continue
            # If have not reached the batch size
            if count < batch_size :
                img = cv2.imread(img_filename)
                
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # Add random translation to the image
                img, steering = translate_img(img, steering)
                # Flip the image, randomly (0.5 probability)
                img,steering = random_flip(img,steering)
                # Do any preprocessing needed
                img = process_img(img)
                images.append(img)
                angles.append(steering)
                count += 1
            else:
                images, angles = shuffle(images,angles)
                yield np.asarray(images), np.asarray(angles)
                #reset all temporary batch parameters to zero
                count =0
                images = []
                angles = []

def valid_generator(validation_log,batch_size = 32):
    images = []
    angles = []
    while True:
        validation_log = shuffle(validation_log)

        for img_filename, steering in validation_log:
            img = process_img(cv2.imread(img_filename))
            images.append(img)
            angles.append(steering)
            
            if len(images) >= batch_size:
                images, angles = shuffle(images, angles)
                yield np.asarray(images), np.asarray(angles)

				from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout

def nvidia_model():
    input_shape = (160,320,3)
    model = Sequential()
    # normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = input_shape, output_shape = input_shape))
    # crop
    model.add(Cropping2D(cropping=((32, 25), (0, 0)),input_shape=(160, 320, 3)))
    #50,20 / 60,20
    
    model.add(Conv2D(3,3,3, subsample=(2,2), border_mode='same',activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Conv2D(6,5,5, subsample=(2,2), border_mode='same',activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Conv2D(9,5,5, subsample=(2,2), border_mode='same',activation='elu'))
    model.add(Dropout(0.5))
    
   # model.add(Conv2D(64,3,3, border_mode='same',activation='elu'))
    model.add(Conv2D(12,3,3, border_mode='same',activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    #model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

# Set data folder
data_directory = './data/'
# read udacity csv
data_log = []
data_log = read_data_csv(data_directory,'driving_log.csv', data_log)
train_log, validation_log = train_test_split(data_log, test_size=0.2)
# Process validation log 
validation_log = separate_data_csv(validation_log, correction = 0)


# Process training log, steering correction of left/right images set to 0.25
train_log = separate_data_csv(train_log, correction = 0.25)

train_generator = generator(train_log, batch_size=64)
validation_generator = generator(validation_log, batch_size=64)
model = nvidia_model()

# Train the model in multiple stages to chose easily a model with smaller overfitting
model.fit_generator(train_generator, samples_per_epoch =12800, validation_data = validation_generator, nb_val_samples = 1280, nb_epoch =6)
model.save('model.h5')