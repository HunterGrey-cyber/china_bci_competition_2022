import os

import tensorflow as tf
from tensorflow import keras

from Algorithm.EEGModels import *

def create_model():
    
    model = EEGNet(nb_classes = 3, Chans = 64, Samples = Samples,F1 = 4,
               dropoutRate = 0.5)
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])
    return model

model = create_model()
model.save('saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')