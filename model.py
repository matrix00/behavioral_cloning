import csv
import cv2
import numpy as np
import random
from sklearn.utils import shuffle


#images_dir = "T1TrainImages"
images_dir= "T1Images"
#images_dir_reverse = "T1Reverse"
#images_dir = "T2TrainImages"

reverse_img = False 

def batch_generator(img_set, angle_set, batch_size):
	images = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
	angles = np.empty(batch_size)
	while True:
		i = 0
		for index in np.random.permutation(len(img_set)):
			images[i] = img_set[index]
			angles[i] = angle_set[index]
			i += 1
			if (i == batch_size):
				break
        
		yield images, angles

#pre process image
#NVIDIA paper converts RGB to YUV
def pre_process_image(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


#reading driving log that has 3 images, steering wheel angle, etc
lines = []
with open(images_dir+"/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		

if reverse_img:
	with open(images_dir_reverse+"/driving_log.csv") as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
		

#last image
images = []
#steering wheel angle
correction  = [0, 0.2, -0.2] # this is a parameter to tune
#correction  = [0, 0.15, -0.15] # this is a parameter to tune

#validation set from center images
val_images = []
val_str_angle = []

#steering_left = steering_center + correction
#steering_right = steering_center - correction
p=0
pstart=3018 
measurements = []

print('total images ', len(lines))
img_no=0
for line in lines:
	for i in range(3):
		source_path = line[i]
		file_name = source_path.split('/')[-1]
		img_path = images_dir+'/IMG/'+file_name
		image = cv2.imread(img_path)

		#print('processing img no', img_no, ' image name ', file_name)
		img_no +=1

		#pre process image
		image = pre_process_image(image)

		images.append(image)
		measurement = np.float32(line[3]) *  (1.0+ correction[i])
#		measurement = np.float32(line[3])+ correction[i]
		measurements.append(measurement)

		#validation set
		#use center image
		if (i==0):
			val_images.append(image)
			val_str_angle.append(measurement)

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
	if np.random.rand() < 0.5:
		aug_images.append(cv2.flip(image,1))
		aug_measurements.append(measurement*-1.0)


from sklearn.model_selection import train_test_split

#X_train, X_valid, y_train, y_valid = train_test_split(aug_images, aug_measurements, test_size=0.20)


X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

X_valid = np.array(val_images)
y_valid = np.array(val_str_angle)

print('training set ', X_train.shape)
print('validation set ', X_valid.shape)

##generate batch for training image
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = X_train[0].shape

print (' h ', IMG_HEIGHT, ' w ', IMG_WIDTH, ' ch ', IMG_CHANNEL)

#shuffle data
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)


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

#checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, mode='auto')


model = Sequential()

#normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#crop the image to take off top sky and bottom car dash
model.add(Cropping2D(cropping=((50,10), (0,0))))

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

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
    
BATCH_SIZE=32
EPOCH = 2
SAMPLE_PER_EPOCH=len(X_train)
VALID_SAMPLES = len(X_valid)
STEPS_PER_EPOCH = len(X_train)

train_generator = batch_generator(X_train, y_train, BATCH_SIZE)
validation_generator = batch_generator(X_valid, y_valid, BATCH_SIZE)

#model.fit_generator(batch_generator(X_train, y_train, BATCH_SIZE), SAMPLE_PER_EPOCH, EPOCH, max_q_size=1, validation_data=batch_generator(X_valid, y_valid, BATCH_SIZE), nb_val_samples=len(X_valid), verbose=1)

#model.fit_generator(train_generator, samples_per_epoch = SAMPLE_PER_EPOCH, epochs=EPOCH, validation_data=validation_generator, nb_val_samples=VALID_SAMPLES )


#model.fit_generator(batch_generator(X_train, y_train, BATCH_SIZE), steps_per_epoch=None, epochs=EPOCH, verbose=1, callbacks=None, validation_data=batch_generator(X_valid, y_valid, BATCH_SIZE),  max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

#model.fit_generator(train_generator, steps_per_epoch= STEPS_PER_EPOCH, validation_data=validation_generator, validation_steps=VALID_SAMPLES, epochs=EPOCH, verbose = 1)

model.save('model.h5')
