from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Images/train'
valid_path = 'Images/test'

# add preprocessing layer to the front of VGG
#+ [3] in 1st arg is RGB channel , Weights we are taking are of Imagenet , , We are not Including the Top layer , bcoz it may have many inputs but we only some input   
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
	layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('Images/train/*')
  
x = Flatten()(vgg.output)

prediction = Dense(len(folders) , activation= "softmax")(x)

model = Model(inputs=vgg.input , outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255 , shear_range=0.2,zoom_range=0.2 , horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Images/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Images/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# Fit the Model
r = model.fit_generator(
	training_set,
	validation_data=test_set,
	epochs=5,
	steps_per_epoch=len(training_set),
	validation_steps=len(test_set)
)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')
