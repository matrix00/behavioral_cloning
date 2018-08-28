import csv
import cv2
import numpy as np

#reading driving log that has 3 images, steering wheel angle, etc
lines = []
with open("driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		

#last image
images = []
#steering wheel angle
measurements = []
for line in lines:
	source_path = line[0]
	file_name = source_path.split('/')[-1]
	img_path = 'IMG/'+file_name
	image = cv2.imread(img_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)

#create a simple model for testing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#model.add(Activation('softmax'))



model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')
