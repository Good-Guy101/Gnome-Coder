import os, time, sys
import numpy as np 
import cv2
from matplotlib import pyplot as plt
import imageio
import tensorflow as tf
from IPython import display
import dearpygui.dearpygui as dpg
from PIL import Image

print("Loading Keras...")
import keras
from gnencoder import *

cvae = CVAE(25)
sample = np.empty([1,25], dtype="float32")
img = np.zeros([256,256, 3])
scale = 2


sliders = []

if(os.path.exists("./encoder.index")):
    cvae.load()

dpg.create_context()
dpg.create_viewport(title="Gnencoder", width=900, height=500)

def slider_callback(sender, app_data):
	for i in range(len(sliders)):
		sample[:,i] = dpg.get_value(sliders[i])
	img = cvae.sample(sample).numpy()
	dpg.set_value("texture_tag", img)

def rand_callback(sender, app_data):
	for i in range(len(sliders)):
		dpg.set_value(sliders[i], np.random.uniform(-2, 2))
		sample[:,i] = dpg.get_value(sliders[i])
	img = cvae.sample(sample).numpy()
	dpg.set_value("texture_tag", img)

def zero_callback(sender, app_data):
	for i in range(len(sliders)):
		dpg.set_value(sliders[i], 0)
		sample[:,i] = dpg.get_value(sliders[i])
	img = cvae.sample(sample).numpy()
	print(img.shape)
	dpg.set_value("texture_tag", img)

def save_callback(sender, app_data):
	for i in range(len(sliders)):
		sample[:,i] = dpg.get_value(sliders[i])
	img = cvae.sample(sample).numpy()
	print(img.shape)
	im = Image.fromarray((img))
	im.save("user_made.jpeg")

with dpg.texture_registry():
	dpg.add_raw_texture(width=256, 
						height=256, 
						default_value=img, 
						format=dpg.mvFormat_Float_rgb, 
						tag="texture_tag")

with dpg.window(label="Sliders", width=460):
	dpg.add_text("Sample Sliders")
	with dpg.group(width=10, horizontal=True):
		for i in range(0,25):
			sliders.append(dpg.add_slider_float(default_value=0, 
												min_value=-1*scale, 
												max_value=1*scale, 
												vertical=True, 
												callback=slider_callback))
	"""with dpg.group(width=10, horizontal=True):
		for i in range(0, 25):
			sliders.append(dpg.add_slider_float(default_value=0, 
												min_value=-1*scale, 
												max_value=1*scale, 
												vertical=True, 
												callback=slider_callback))"""
	with dpg.group(horizontal=True):
		dpg.add_button(label="Randomize", callback=rand_callback)
		dpg.add_button(label="Zero", callback=zero_callback)
		dpg.add_button(label="Save", callback=save_callback)


with dpg.window(label="Gnome", width=256):
	dpg.add_image("texture_tag")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()




"""
rand_sample = cvae.sample()
for i in range(rand_sample.shape[0]):
	plt.subplot(4, 4, i + 1)
	plt.imshow(rand_sample[i])
	plt.axis("off")
plt.savefig("./rand_sample.png")
plt.show()
plt.close()
"""