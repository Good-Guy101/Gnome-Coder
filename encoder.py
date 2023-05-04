import sys, random, math, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2

#Constants
PARAMS = 100

BATCH_SIZE = 1
EPOCHS = 2000
LR = 0.001
NUM_RAND_GNOMES = 10

MODEL_EXISTS = False

def save_image(x, fname):
	img = np.transpose(x * 255, [1, 2, 0])
	img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
	cv2.imwrite(fname, img)

#Load data set
print("Loading data...")
#y_train = np.load('y_data.npy').astype(np.float32) / 255.0
x_train = np.expand_dims(np.arange(np.random.randint(255, size=(100,255,255,3)).shape[0]), axis=1)
print(x_train)
assert(False)



#Create Model

print("Loading Keras...")
#os.environ['THEANORC'] = "./gpu.theanorc"
#os.environ['KERAS_BACKEND'] = "theano"
#import theano
#print("Theano Version: " + theano.__version__)

from keras.initializers import RandomUniform
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Embedding
from keras.layers import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model
from keras import backend as K

if MODEL_EXISTS:
	print("Loading Model...")
	model = load_model("Encoder.h5")
else:

	print("Building Model...")
	model = Sequential()

	#Embedding/"Encoder" layers
	model.add(Embedding(100, PARAMS, input_length=1))
	model.add(Flatten(name="embed"))


	model.add(Reshape((1, 1, PARAMS), name="decoder"))
	#(1, 1, 100)

	#Decoder
	model.add(Conv2DTranspose(256, 2, activation="relu"))								#(2,2)
	model.add(Conv2DTranspose(256, 3, activation="relu"))								#(4,4)
	model.add(Conv2DTranspose(256, 5, activation="relu"))								#(8,8)
	model.add(Conv2DTranspose(256, 5, activation="relu"))								#(12,12)
	model.add(Conv2DTranspose(256, 5, activation="relu"))								#(16,16)
	model.add(Conv2DTranspose(256, 5, activation="relu", strides=2, padding="same"))	#(32,32,256)
	model.add(Conv2DTranspose(128, 5, activation="relu", strides=2, padding="same"))	#(64,64,128)
	model.add(Conv2DTranspose(128, 5, activation="relu", strides=2, padding="same"))	#(128,128,128)
	model.add(Conv2DTranspose(3, 5, activation="sigmoid", strides=2, padding="same"))		#(256,256,3)

	model.compile(optimizer=Adam(learning_rate=LR), loss="mse")

#Compile "Encoder" and Decoder 
print("Compiling SubModels...")
encoder = Model(inputs=model.input, outputs=model.get_layer("embed").output)
decoder = Model(inputs=model.get_layer("decoder").input, outputs=model.layers[-1].output)

seed = np.random.normal(0.0, 1.0, (NUM_RAND_GNOMES, PARAMS))

def generate_rand_gnomes(seed, iters):
	x_enc = encoder.predict(x_train, batch_size=BATCH_SIZE)

	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	e, v = np.linalg.eig(x_cov)

	np.save('means.npy', x_mean)ain_loss = []
	np.save('stds.npy', x_stds)
	np.save('evals.npy', e)
	np.save('evecs.npy', v)
	
	e_list = e.tolist()
	e_list.sort(reverse=True)
	plt.clf()
	plt.bar(np.arange(e.shape[0]), e_list, align='center')
	plt.draw()
	plt.savefig('evals.png')
	
	x_vecs = x_mean + np.dot(v, (seed * e).T).T
	y_gnomes = decoder.predict(x_vecs)[0]
	print(y_gnomes)
	for i in range(y_gnomes.shape[0]):
		save_image(y_gnomes[i], 'rand' + str(i) + '.png')
		if i < 5 and (iters % 10) == 0:
			if not os.path.exists('morph' + str(i)):
				os.makedirs('morph' + str(i))
			save_image(y_gnomes[i], 'morph' + str(i) + '/img' + str(iters) + '.png')

generate_rand_gnomes(seed, 0)