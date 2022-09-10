import pandas as pd
import mne
import numpy as np
from tensorflow.keras import utils as np_utils
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

import datetime

from Algorithm.EEGModels import *

ch_types = []
for i in range(64):
    ch_types.append('eeg')
ch_types.append('stim')

IMAGES_PATH = 'img'

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

def fetch_data_label(pkl_path):
    
    obj = pd.read_pickle(pkl_path)
    obj['ch_names'] = obj['ch_names'] + ('stim',)

    raw = mne.io.RawArray(obj["data"],mne.create_info(obj["ch_names"],250,ch_types=ch_types))
    raw.filter(2, None, method='iir')  # replace baselining with high-pass
    
    tmin, tmax = -0.2,3.9
    events = mne.find_events(raw)
    event_dict = {'hand/left': 201, 'hand/right': 202, 'feet': 203}
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       )
    epochs = mne.Epochs(raw, events, event_dict,tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    X = epochs.get_data()*1000
    y = np_utils.to_categorical(labels - 201 )
    return X,y


def get_data(id):
    
    X_train_1,Y_train_1 = fetch_data_label('train/S0'+str(id) + '/block_1.pkl')
    X_train_2,Y_train_2 = fetch_data_label('train/S0'+str(id) + '/block_2.pkl')
    X_train_3,Y_train_3 = fetch_data_label('train/S0'+str(id) + '/block_3.pkl')
    X = np.concatenate((X_train_1,X_train_2,X_train_3))
    Y = np.concatenate((Y_train_1,Y_train_2,Y_train_3))
    # X,Y = shuffle(X,Y)
    return X,Y

X = np.empty((0,64,1026))
Y = np.empty((0,3))

for i in range(1,6):
    X_tmp,Y_tmp = get_data(i)
    X = np.concatenate((X,X_tmp))
    Y = np.concatenate((Y,Y_tmp))

print(X.shape)