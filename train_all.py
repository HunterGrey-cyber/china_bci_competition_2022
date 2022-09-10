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
from tensorboard.plugins.hparams import api as hp
from scipy import signal

samples = 100
# tools for plotting confusion matrices
from matplotlib import pyplot as plt

import datetime

from Algorithm.EEGModels import *

ch_types = []
for i in range(64):
    ch_types.append('eeg')
ch_types.append('stim')

IMAGES_PATH = 'img'

filter = 'high'

ch_names = []
for i in range(64):
    ch_names.append('eeg' + str(i))
    
def preprocess_high(data):

    filtered = mne.filter.filter_data(data, 250,2,None, method='iir')

    return filtered

def preprocess_original(data):
    # 选择使用的导联
    fs = 250
    f0 = 50
    q = 35
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    filter_data = signal.filtfilt(b,a,data)
    return filter_data

def change_samples(X,y,samples):
    X = X[:,:,:1000]
    X = np.transpose(X,(1,0,2))
    add_num = int(1000/samples)
    axis_1 = int(30 * add_num)
    X = X.reshape(X.shape[0],axis_1,samples)
    X = np.transpose(X,(1,0,2))
    y = np.broadcast_to(y,(add_num,30))
    y = np.transpose(y)
    y = np.reshape(y,axis_1)
    return X,y
    
    
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
    
    tmin, tmax = 0,4
    events = mne.find_events(raw)
    event_dict = {'hand/left': 201, 'hand/right': 202, 'feet': 203}
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       )

    epochs = mne.Epochs(raw, events, event_dict,tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]

    X = epochs.get_data()
    # X,y = change_samples(X,labels - 201,samples)
    y = labels - 201
    y = np_utils.to_categorical(y)
    return X,y

def get_data(id):
    
    X_train_1,Y_train_1 = fetch_data_label('../data/train/S0'+str(id) + '/block_1.pkl')
    X_train_2,Y_train_2 = fetch_data_label('../data/train/S0'+str(id) + '/block_2.pkl')
    X_train_3,Y_train_3 = fetch_data_label('../data/train/S0'+str(id) + '/block_3.pkl')
    X = np.concatenate((X_train_1,X_train_2,X_train_3))
    Y = np.concatenate((Y_train_1,Y_train_2,Y_train_3))
    return X,Y


trails = int(90000 / samples)

X,Y = get_data(1)
# X = np.reshape(X,(trails,64,samples))

for i in range(2,6):
    tmp_x,tmp_Y = get_data(i)
    X = np.concatenate((X,tmp_x))
    Y = np.concatenate((Y,tmp_Y))

    

# if filter == 'high':
#     for i in range(trails):
#         X[i,:,:] = preprocess_high(X[i,:,:])

# if filter == 'original':
#     for i in range(trails):
#         X[i,:,:] = preprocess_high(X[i,:,:])
#     for i in range(trails):
#         X[i,:,:] = preprocess_original(X[i,:,:])

X_train,X_rem,Y_train,Y_rem = train_test_split(X,Y,test_size=0.5)
X_validate,X_test,Y_validate,Y_test = train_test_split(X_rem,Y_rem,test_size=0.5)
Samples = X_train.shape[2]

X_train_new = np.concatenate((X_train[:,:,:100],X_train[:,:,100:]),axis=2)
X_train = np.concatenate((X_train,X_train_new))
Y_train = np.concatenate((Y_train,Y_train))

X_train      = X_train.reshape(X_train.shape[0], 64, Samples, 1)
X_validate   = X_validate.reshape(X_validate.shape[0], 64, Samples, 1)
X_test       = X_test.reshape(X_test.shape[0], 64, Samples, 1)

start = datetime.datetime.now()
Samples = X_train.shape[2]

# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 1000
def run(i):

    model = EEGNet(nb_classes = 3, Chans = 64, Samples = Samples,F1 = 4,
               dropoutRate = 0.5)
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])
    
    fittedModel = model.fit(X_train, Y_train, batch_size = 64,epochs = epochs,
                            validation_data=(X_validate, Y_validate))
    result = model.evaluate(X_test, Y_test)

    acc = result[1]
    print("Classification accuracy: %f " % (acc))
    model.save('./saved/model1' +str(i) + '.h5')
    return acc,fittedModel
acc = []

for i in range(1):
    acc,fittedModel = run(i)

print(acc)
pd.DataFrame(fittedModel.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


