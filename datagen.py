import os
import numpy as np 
import cv2

#Data augmentation functions
#Gaussian noise injection
def gauss_inject(img):
	output = img.copy()
	mean = 0.0
	var = 10
	sigma = var*0.5
	gauss = np.random.normal(mean, sigma, (output.shape[:2]))
	gauss = gauss.reshape(output.shape[:2])
	output[:, :, 0] = output[:, :, 0] + gauss
	output[:, :, 1] = output[:, :, 1] + gauss
	output[:, :, 2] = output[:, :, 2] + gauss
	cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX, dtype=-1)
	output = output.astype(np.uint8)
	return output

#Salt and Pepper injection for RGB img
#Modified from https://gist.github.com/lucaswiman/1e877a164a69f78694f845eab45c381a
def SP_inject(img, prob):
	output = img.copy()
	black = np.array([0, 0, 0], dtype="uint8")
	white = np.array([255, 255, 255], dtype="uint8")
	probs = np.random.random(output.shape[:2])
	output[probs < (probs / 2)] = black
	output[probs > 1 - (prob / 2)] = white
	return output

#Random Saturation change
def saturation(img):
	imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
	(h, s, v) = cv2.split(imghsv)
	s = s * np.random.uniform(low=0.0, high=5.0)
	s = np.clip(s, 0, 255)
	imghsv = cv2.merge([h, s, v])
	output = cv2.cvtColor(imghsv.astype("uint8"),cv2.COLOR_HSV2RGB)
	return output
#Translation
def translate(img, range):
	output = img.copy()
	tx = np.random.randint(low=-range, high=range)
	ty = np.random.randint(low=-range, high=range)
	T = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
	output = cv2.warpAffine(output, T, output.shape[:2], 
		borderMode=cv2.BORDER_CONSTANT, 
		borderValue=(255, 255, 255))
	return output

#Generate Augmented Data
SRC_PATH = "./gnomesCleaned0.2"
NEW_DIR = "./gnomesAugmented"

count = 0

dataset = np.empty((len(os.listdir(SRC_PATH))*5, 256, 256, 3), dtype=np.uint8)

for file in os.listdir(SRC_PATH):
	img = cv2.imread(os.path.join(SRC_PATH, file))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), img)
	dataset[count] = np.asarray(img)
	count += 1
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), SP_inject(img, 0.1))
	dataset[count] = np.asarray(SP_inject(img, 0.1))
	count += 1
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), cv2.flip(img, 1))
	dataset[count] = np.asarray(cv2.flip(img, 1))
	count += 1
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), saturation(img))
	dataset[count] = np.asarray(saturation(img))
	count += 1
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), translate(img, 25))
	dataset[count] = np.asarray(translate(img, 25))
	count += 1

np.save("train_data.npy", dataset)

count = 0

dataset = np.empty((len(os.listdir(SRC_PATH)), 256, 256, 3), dtype=np.uint8)

for file in os.listdir(SRC_PATH):
	img = cv2.imread(os.path.join(SRC_PATH, file))
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
	#cv2.imwrite(os.path.join(NEW_DIR, str(count) + ".jpg"), img)
	dataset[count] = np.asarray(img)
	count += 1

np.save("test_data.npy", dataset)

"""
img = cv2.imread("./gnomesCleaned0.2/26g.jpg")
cv2.imshow("Original", img)
gauss = gauss_inject(img)
saltpep = SP_inject(img, 0.05)
sat = saturation(img)
trans = translate(img, 25)
cv2.imshow("gauss", gauss)
cv2.imshow("saltpep", saltpep)
cv2.imshow("saturation", sat)
cv2.imshow("translate", trans)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""