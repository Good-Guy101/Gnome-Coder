import os, time, sys
import numpy as np 
import cv2
import tensorflow as tf
import dearpygui.dearpygui as dpg

print("Loading Keras...")
import keras
from gnencoder import *

cvae = CVAE(25)
sample = np.empty([1,25], dtype="float32")
img = np.zeros([256,256, 3])
scale = 2

count = 0

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
	dpg.set_value("texture_tag", img)

def save_callback(sender, app_data):
	global count
	for i in range(len(sliders)):
		sample[:,i] = dpg.get_value(sliders[i])
	img = cvae.sample(sample).numpy()
	img  = img.reshape([256, 256, 3])
	cv2.imwrite("user_generated"+ str(count) +".png", cv2.cvtColor((img * 255), cv2.COLOR_BGR2RGB))
	count += 1

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
	with dpg.group(horizontal=True):
		dpg.add_button(label="Randomize", callback=rand_callback)
		dpg.add_button(label="Zero", callback=zero_callback)
		dpg.add_button(label="Save", callback=save_callback)

with dpg.window(label="Gnome", width=256, pos=[900-256,0]):
	dpg.add_image("texture_tag")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()