import os
import json
import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from util import load_image, load_labels, visualise
from neural_network import build_model, FaceTracker, localization_loss


#LOADING THE DATA AND LOADING IT ONTO THE PIPELINE  
train_images=tf.data.Dataset.list_files('./aug_data/train/images/*jpg', shuffle=False)
train_images=train_images.map(load_image)
train_images=train_images.map(lambda x: tf.image.resize(x, (120, 120 )))
train_images=train_images.map(lambda x:x/255)
test_images=tf.data.Dataset.list_files('./aug_data/test/images/*jpg', shuffle= False)
test_images=test_images.map(load_image)
test_images=test_images.map(lambda x: tf.image.resize(x, (120, 120 )))
test_images=test_images.map(lambda x:x/255)
val_images=tf.data.Dataset.list_files('./aug_data/val/images/*jpg', shuffle = False)
val_images=val_images.map(load_image)
val_images=val_images.map(lambda x: tf.image.resize(x, (120, 120 )))
val_images=val_images.map(lambda x:x/255)


#LOADING THE LABELS
train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*json', shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*json', shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels  = tf.data.Dataset.list_files('aug_data/val/labels/*json', shuffle = False)
val_labels  = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

'''
#NUMBER OF DATA VALUES
print(len(train_images))
print(len(test_images))
print(len(val_images))
print(len(train_labels))
print(len(test_labels))
print(len(val_labels))

o/p
4400
700
1200
4400
700
1200

'''


train=tf.data.Dataset.zip((train_images,train_labels))
train=train.shuffle(5000)
train=train.batch(8)
train= train.prefetch(4)


test = tf.data.Dataset.zip((test_images,test_labels))
test = test.shuffle(100)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip( (val_images, val_labels))
val = val.shuffle(1500)
val = val.batch(8)
val = val.prefetch(4)

data_samples = train.as_numpy_iterator()


'''
#TO VISUALISE THE IMAGES
res = data_samples.next()
visualise(res)
'''

batches_per_epoch = len(train)
lr_decay = (1.0/0.75 - 1)/batches_per_epoch

opt = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0001, decay = lr_decay)
class_loss = tf.keras.losses.BinaryCrossentropy()
regress_loss = localization_loss
facetracker = build_model()

model = FaceTracker(facetracker)
model.compile(opt , classloss=class_loss, localizationloss= regress_loss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logdir)

hist = model.fit(train, epochs=40, validation_data = val, callbacks = [tensorboard_callback])



