#Code Referenced from https://keras.io/examples/generative/vae/ 
import sys, random, math, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import tensorflow as tf
import keras

from keras.layers import Layer,Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, load_model

PARAMS = 100
IMG_DIM = (256, 256, 3)

def gen_encoder(latent_dim):
	inputs = Input(shape=(IMG_DIM))
	x = Conv2D(3, 5, activation="relu", strides=2, padding="same")(inputs)
	x = Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)	
	x = Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2D(256, 3, activation="relu")(x)
	x = Conv2D(256, 3, activation="relu")(x)
	x = Conv2D(256, 3, activation="relu")(x)
	x = Flatten()(x)
	output = Dense(latent_dim*2)(x)
	model = Model(inputs, output, name="encoder")
	return model

def gen_decoder(latent_dim):
	latent_inputs = Input(shape=(latent_dim,))
	x = Dense(2 * 2 * 256, activation="relu")(latent_inputs)
	x = Reshape((2, 2, 256))(x)
	x = Conv2DTranspose(256, 3, activation="relu")(x)
	x = Conv2DTranspose(256, 3, activation="relu")(x)
	x = Conv2DTranspose(256, 3, activation="relu")(x)
	x = Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
	x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
	outputs = Conv2DTranspose(3, 5, strides=2, padding="same")(x)
	model = Model(latent_inputs, outputs, name="decoder")
	return model

class CVAE(Model):
	def __init__(self, latent_dim):
		super().__init__()
		self.latent_dim = latent_dim
		self.encoder = gen_encoder(latent_dim)
		self.decoder = gen_decoder(latent_dim)
		
	@tf.function
	def sample(self, eps=None):
		if(eps is None):
			eps = tf.random.normal(shape=(1, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, input):
		mean, logvar = tf.split(self.encoder(input), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, input, apply_sigmoid=False):
		logits = self.decoder(input)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)
			return probs
		return logits

	def save(self):
		self.encoder.save_weights("encoder")
		self.decoder.save_weights("decoder")

	def load(self):
		self.encoder.load_weights("encoder")
		self.decoder.load_weights("decoder")


def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def loss_ELBO(model, input, beta=1):
	mean, logvar = model.encode(input)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)
	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
	logpz = log_normal_pdf(z, 0., 0.)
	logqz_x = log_normal_pdf(z, mean, logvar)
	return -tf.reduce_mean(logpx_z + logpz - beta*logqz_x)

@tf.function
def train_step(model, input, optimizer):
	with tf.GradientTape() as tape:
		loss = loss_ELBO(model, input)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_epoch_gnomes(model, epoch, sample):
	mean, logvar = model.encode(sample)
	z = model.reparameterize(mean, logvar)
	predict = model.sample(z)
	fig = plt.figure(figsize=(4, 4))

	for i in range(predict.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predict[i])
		plt.axis("off")
	plt.savefig("./img_epoch/"+"img_epoch_{:04d}.png".format(epoch))
	#plt.show()
	plt.close()
