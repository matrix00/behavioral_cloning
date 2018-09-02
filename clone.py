import csv
import cv2
import numpy as np

#reading driving log that has 3 images, steering wheel angle, etc
lines = []
with open("Run1/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		

#last image
images = []
#steering wheel angle
correction  = [0, 0.15, -0.15] # this is a parameter to tune

#steering_left = steering_center + correction
#steering_right = steering_center - correction
p=0
pstart=3018 
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		file_name = source_path.split('/')[-1]
		img_path = 'Run1/IMG/'+file_name
		image = cv2.imread(img_path)
		images.append(image)
		measurement = float(line[3])  + correction[i]
		measurements.append(measurement)
		p = p+1
		if (p > pstart and p < pstart+20):
 			print (' p' , p, ' image name ', img_path, ' steering angle ', measurement)


#augment images and with flipped images, 
#augment measurments with negative for flip images
aug_images = []
aug_measurements = []

for image, measurement in zip(images, measurements):
	aug_images.append(image)
	aug_measurements.append(measurement)
	aug_images.append(cv2.flip(image,1))
	aug_measurements.append(measurement*-1.0)

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)


print(X_train.shape)

#create a simple model for testing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D


from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

"""
    NVIDIA model 
    Normalization
    Convolution: 5x5, filter: 24, strides: 2x2
    Convolution: 5x5, filter: 36, strides: 2x2 
    Convolution: 5x5, filter: 48, strides: 2x2
    Convolution: 3x3, filter: 64, strides: 1x1
    Convolution: 3x3, filter: 64, strides: 1x1
    Drop out (0.5)
    Fully connected: neurons: 100
    Fully connected: neurons: 50
    Fully connected: neurons: 10
    Fully connected: neurons: 1 (output)
"""


#from keras import backend as K
#K.set_image_dim_ordering('th')

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Cropping2D(cropping=((30,10), (0,0))))
model.add(Convolution2D(24,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')
