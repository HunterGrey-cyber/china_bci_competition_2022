import mne
import numpy as np
import os

# Mention the file path to the dataset
filename = 'BCI_Data/A01T.gdf'

raw = mne.io.read_raw_gdf(filename)

events, _ = mne.events_from_annotations(raw)

raw.load_data()

raw.filter(7., 35., fir_design='firwin')

raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

tmin, tmax = 1., 4.
IMAGES_PATH = 'bci'
event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Getting labels and changing labels from 7,8,9,10 -> 1,2,3,4
y = epochs.events[:,-1] - 6

X = epochs.get_data()

import numpy as np
import datetime
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from Algorithm.EEGModels import *
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import pandas as pd


# tools for plotting confusion matrices
from matplotlib import pyplot as plt

kernels, chans, samples = 1, 22, 751

X_train      = X[0:144,]
Y_train      = y[0:144]
X_validate   = X[144:216,]
Y_validate   = y[144:216]
X_test       = X[216:,]
Y_test       = y[216:]

Y_train      = np_utils.to_categorical(Y_train-1)
Y_validate   = np_utils.to_categorical(Y_validate-1)
Y_test       = np_utils.to_categorical(Y_test-1)

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
   
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples,
               dropoutRate = 0.25)

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

fittedModel = model.fit(X_train, Y_train, batch_size = 20,epochs = 2000,
                        verbose = 2, validation_data=(X_validate, Y_validate))

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

pd.DataFrame(fittedModel.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
save_fig("keras_learning_curves_plot")
plt.show()