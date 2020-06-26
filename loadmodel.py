import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from PIL import Image

#Load the model
model = load_model('cnn_from_scratch_fruits.hdf5')
model.summary()

def get_latest_image(dirpath, validext = ('jpg', 'JPG')):
	valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
	valid_files = [f for f in valid_files if '.' in f and \
				   f.rsplit('.', 1)[-1] in validext and os.path.isfile(f)]
	if not valid_files:
		raise ValueError("No valid images in %s" % dirpath)

	return max(valid_files, key= os.path.getmtime)

def get_data(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels

# preprocessing the train, validation, and test images by flattening 
def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

testimg= '/Users/sarvesh/Desktop/avo'
picspath = testimg + '/' + 'pics/'
dsstorefilesintwo = [f for f in os.listdir(picspath) if f.endswith(".DS_Store")]
for fiile in dsstorefilesintwo:
  os.remove(os.path.join(picspath, fiile))

for img in os.listdir(picspath):
	realImg = get_latest_image(picspath)
	if picspath + img != realImg:
		os.remove(os.path.join(picspath, img))
print(os.listdir(picspath))

for img in os.listdir(picspath):
  width, height = Image.open(picspath + img).size
  if width*height == 1000:
  	continue
  else:
  	print(img)
  	image = Image.open(picspath+img)
  	image.resize((100, 100))
  	base, ext = os.path.splitext(img)
  	newname = base + 'new'
  	fullname = newname+ext
  	image.save(picspath + fullname)
  	os.remove(os.path.join(picspath, img))
    
X_test, Y_test,_ = get_data(testimg)
print(X_test)
X_test = np.array(convert_image_to_array(X_test))
X_test = X_test.astype('float32')/255

# using model to predict on test data
Y_pred = model.predict(X_test)
fig = plt.figure(figsize=(20, 15))
for i, idx in enumerate(np.random.choice(X_test.shape[0], size=1, replace=False)):
    ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_test[idx]))
    pred_idx = np.argmax(Y_pred[idx])
    true_idx = np.argmax(Y_test[idx])
    ax.set_title("{}".format(labels[pred_idx]),
                 color=("green"))