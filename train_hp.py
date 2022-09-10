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

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

import datetime

from Algorithm.EEGModels import *

ch_types = []
for i in range(64):
    ch_types.append('eeg')
ch_types.append('stim')

IMAGES_PATH = 'img'

def add_data(X,y):
    X = X[:,:,:1000]
    X = np.transpose(X,(1,0,2))
    add_num = X.shape[0]/X.shape[2]
    X = X.reshape(X.shape[0],3000,10)
    X = np.transpose(X,(1,0,2))
    y = np.broadcast_to(y,(100,30))
    y = np.transpose(y)
    y = np.reshape(y,3000)
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
    X,y = add_data(X,labels - 201)
    y = np_utils.to_categorical(y)
    return X,y

def get_data(id):
    
    X_train_1,Y_train_1 = fetch_data_label('train/S0'+str(id) + '/block_1.pkl')
    X_train_2,Y_train_2 = fetch_data_label('train/S0'+str(id) + '/block_2.pkl')
    X_train_3,Y_train_3 = fetch_data_label('train/S0'+str(id) + '/block_3.pkl')
    X = np.concatenate((X_train_1,X_train_2,X_train_3))
    Y = np.concatenate((Y_train_1,Y_train_2,Y_train_3))
    return X,Y

X,Y = get_data(1)
X = np.reshape(X[:,:,:1000],(9000,64,10))


X_train,X_rem,Y_train,Y_rem = train_test_split(X,Y,test_size=0.5)
X_validate,X_test,Y_validate,Y_test = train_test_split(X_rem,Y_rem,test_size=0.5)
Samples = X_train.shape[2]

X_train_new = np.concatenate((X_train[:,:,:100],X_train[:,:,100:]),axis=2)
X_train = np.concatenate((X_train,X_train_new))
Y_train = np.concatenate((Y_train,Y_train))

X_train      = X_train.reshape(X_train.shape[0], 64, Samples, 1)
X_validate   = X_validate.reshape(X_validate.shape[0], 64, Samples, 1)
X_test       = X_test.reshape(X_test.shape[0], 64, Samples, 1)


Samples = X_train.shape[2]


hp_kernel_size = hp.HParam('kernel_size', hp.Discrete([16, 32]))
hp_dropout = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[hp_kernel_size, hp_dropout],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

epochs = 500
def train_test_model(hparams):

    model = EEGNet(nb_classes = 3, Chans = 64, Samples = Samples,F1 = 4,kernLength = hp_kernel_size,
               dropoutRate = hp_dropout)

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])
    
    fittedModel = model.fit(X_train, Y_train, batch_size = 64,epochs = epochs,
                            validation_data=(X_validate, Y_validate))
    result = model.evaluate(X_test, Y_test)

    acc = result[1]
    print("Classification accuracy: %f " % (acc))
    model.save_weights('./checkpoints/my_checkpoint')
    return acc

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for kernel_size in hp_kernel_size.domain.values:
    for dropout in (hp_dropout.domain.min_value, hp_dropout.domain.max_value):
        hparams = {
            hp_kernel_size: kernel_size,
            hp_dropout: dropout,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1



# pd.DataFrame(fittedModel.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# save_fig("epochs_" + str(epochs) + "_acc_" +str(acc))
# plt.show()
