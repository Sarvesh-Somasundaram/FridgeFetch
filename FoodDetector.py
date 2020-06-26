

###Written by Sarvesh Somasundaram###

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

base_dir = '/Users/sarvesh/Desktop/FoodDetection/'
print(base_dir)

train_dir = os.path.join(base_dir, 'train/')
train_dir_folders = os.listdir(train_dir)

val_dir = os.path.join(base_dir, 'val/')
val_dir_folders = os.listdir(val_dir)


dsstorefilesintrain = [f for f in train_dir_folders if f.endswith(".DS_Store")]
for file in dsstorefilesintrain:
	os.remove(os.path.join(train_dir, file))
print(dsstorefilesintrain)

dsstorefilesinval = [f for f in val_dir_folders if f.endswith(".DS_Store")]
print(dsstorefilesinval)
for file in dsstorefilesinval:
	os.remove(os.path.join(val_dir, file))
print(dsstorefilesinval)

#Directories of each food for training
foodTrain = sorted([f for f in train_dir_folders], key = str.lower)

foodTrainDir = []
for file in foodTrain:
	foodTrainDir.append(os.path.join(train_dir, file + '/'))

#Directories of each food for validation

foodVal = sorted([f for f in val_dir_folders], key = str.lower)

foodValDir = []
for file in foodVal:
	foodValDir.append(os.path.join(val_dir, file + '/'))

#Setting up the matplotlib figure
nrows = 46
ncols = 45

figure = plt.gcf()
figure.set_size_inches(ncols*6, nrows*6)
pic_index = 100

train_fnames_list = []
for f in foodTrainDir:
	train_fnames_list.append(os.listdir(f))

next_train_pix = []
for x in range(len(foodTrainDir)):

	next_train_pix.append([os.path.join(foodTrainDir[x], fname) 
                	for fname in train_fnames_list[x][pic_index-8:pic_index] 
               		])

oldList = next_train_pix[0]
for i in range(len(next_train_pix)-1):
	newlist = oldList + next_train_pix[i+1]
	oldList = newlist

print(len(next_train_pix))

for i, img_path in enumerate(oldList):
  # Set up subplot
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), # Shape of our images
                                include_top = False, # Leave out the last fully connected layer
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
	layer.trainable = False

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.959):
      print("\n" + 'Reached 99.9%' + ' accuracy so stopping the training!!!')
      self.model.stop_training = True

# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU
x = layers.Dense(1024, activation = 'relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 25,
                                                    class_mode = 'categorical', 
                                                    target_size = (150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  val_datagen.flow_from_directory( val_dir,
                                                          batch_size  = 25,
                                                          class_mode  = 'categorical', 
                                                          target_size = (150, 150))


callbacks = myCallback()
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()