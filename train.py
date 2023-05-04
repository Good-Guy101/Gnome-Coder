import os, time, sys
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import imageio
import tensorflow as tf
from IPython import display

DATA = "./gnomesAugmented/"

BATCH_SIZE = 8
EPOCHS = 1000
LR = 0.0001
NUM_RAND_GNOMES = 16
LATENT_DIM = 25
NEW_MODEL = True

NUM_TRAIN = 885
NUM_TEST = 177

print("Loading Keras...")
import keras
from gnencoder import *

print("Loading Data...")
print("Data Dir = " + DATA)
with open("train_data.npy", "rb") as f:
    a = np.load(f)
a = (a / 255.).astype("float32")
train_data = tf.data.Dataset.from_tensor_slices(a).shuffle(NUM_TRAIN).batch(BATCH_SIZE)
with open("test_data.npy", "rb") as f:
    a = np.load(f)
a = (a / 255.).astype("float32")
test_data = tf.data.Dataset.from_tensor_slices(a).shuffle(NUM_TEST).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.legacy.Adam(LR)
#random vector for sampling
random_vector = tf.random.normal(shape=[NUM_RAND_GNOMES, LATENT_DIM])

cvae = CVAE(LATENT_DIM)

if(os.path.exists("./encoder.index") and not NEW_MODEL):
    cvae.load()

for test_batch in test_data.take(1):
  test_sample = test_batch[0:NUM_RAND_GNOMES, :, :, :]

generate_epoch_gnomes(cvae, 0, test_sample)

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    for train in train_data:
        train_step(cvae, train, optimizer)
        
    end = time.time()
    if(epoch%10 == 0):
         cvae.save()

    loss = tf.keras.metrics.Mean()
    for test in list(test_data.as_numpy_iterator()):
        loss(loss_ELBO(cvae, test, 8))
    elbo = loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            .format(epoch, elbo, end - start))
    generate_epoch_gnomes(cvae, epoch, test_sample)

cvae.save()