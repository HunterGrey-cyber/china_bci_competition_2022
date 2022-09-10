import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing

from Algorithm.EEGModels import *
#标准标准化&平衡化- 格式如(25000,1)
def label_transform(labels):
    encoder = preprocessing.LabelEncoder()#标准化
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    return encoded_labels

def fetch_data_label(pkl_path):
    #闭源
    def change_samples(X, y, samples):  # 30,64,1001 samples=200
        num_feature = X.shape[0]
        tmp = 250 * tmax

        X = X[:, :, :tmp]  # 30,64,1000
        X = X.transpose((1, 0, 2))  # 64,30,1000
        add_num = tmp // samples  # 5 30*1000=200*150 (samples*features)
        # axis_1 = int(30 * add_num)#150
        X = X.reshape(X.shape[0], -1, samples)  # 64,150,200
        X = X.transpose((1, 0, 2))  # 150,64,200

        y = np.broadcast_to(y, (add_num, num_feature))  # 30->(5,30)
        y = y.transpose()  # (30,5)
        y = y.reshape(-1)  # (150,)
        return X, y

    obj = pd.read_pickle(pkl_path)
    obj['ch_names'] = obj['ch_names'] + ('stim',)

    raw = mne.io.RawArray(obj["data"], mne.create_info(obj["ch_names"], 250, ch_types=ch_types))

    raw.filter(2, None, method='iir')  # replace baselining with high-pass

    tmin, tmax = 0, 4#4
    events = mne.find_events(raw)

    event_dict = {'hand/left': 201, 'hand/right': 202, 'feet': 203}
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    labels = label_transform(labels)
    X = epochs.get_data()
    # X, y = change_samples(X, labels - 201, samples)
    X, y = change_samples(X, labels, samples)
    # y = np_utils.to_categorical(y)
    return X, y



def get_data(id):
    X_train_1,Y_train_1 = fetch_data_label('../data/train/S0{}/block_1.pkl'.format(id))
    X_train_2,Y_train_2 = fetch_data_label('../data/train/S0{}/block_2.pkl'.format(id))
    X_train_3,Y_train_3 = fetch_data_label('../data/train/S0{}/block_3.pkl'.format(id))
    X = np.concatenate((X_train_1,X_train_2,X_train_3))
    Y = np.concatenate((Y_train_1,Y_train_2,Y_train_3))
    return X,Y


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

if __name__ == '__main__':
    ch_types = []
    for i in range(64):
        ch_types.append('eeg')
    ch_types.append('stim')
    IMAGES_PATH = 'img'

    samples = 10#30
    resulst_log = []

    if not os.path.exists("img/fzy_img"):
        os.makedirs("img/fzy_img")

    for index in [1,2,3,4,5]:
        X,Y = get_data(index)#450,64,200
        num_class = len(set(Y.tolist()))
        Y = np_utils.to_categorical(Y)
        X_train,X_rem,Y_train,Y_rem = train_test_split(X,Y,test_size=0.6)
        X_validate,X_test,Y_validate,Y_test = train_test_split(X_rem,Y_rem,test_size=0.5)

        #数据增强
        X_train = np.concatenate((X_train,X_train))
        Y_train = np.concatenate((Y_train,Y_train))


        X_train = X_train.reshape(X_train.shape[0], 64, -1, 1)
        X_validate = X_validate.reshape(X_validate.shape[0], 64, -1, 1)
        X_test = X_test.reshape(X_test.shape[0], 64, -1, 1)


        epochs = 200
        model = EEGNet(nb_classes = num_class, Chans = 64, Samples = samples,F1 = 4,kernLength = 125)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),metrics = ['accuracy'])



        save_path = './saved/model{}.h5'.format(index)
        checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy',verbose=1,#'val_loss'
                                     validation_freq=1,save_weights_only=True,save_best_only=True)#'val_loss'
        earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=80, verbose=1)

        fittedModel = model.fit(X_train, Y_train, batch_size = 64,epochs = epochs,
                                validation_data=(X_validate, Y_validate),callbacks=[checkpoint])


        plt.plot(fittedModel.history['accuracy'],label='acc')
        plt.plot(fittedModel.history['loss'],label='loss')
        plt.plot(fittedModel.history['val_accuracy'],label='val_acc')
        plt.plot(fittedModel.history['val_loss'],label='val_loss')
        plt.title('rec_{}'.format(index))
        plt.legend()
        plt.savefig("img/fzy_img/rec_{}.png".format(index), dpi=400)
        plt.show()


        test_acc = model.evaluate(X_test, Y_test)[1]
        print("Classification accuracy: %f " % (test_acc))
        resulst_log.append(test_acc)

    for i,acc in enumerate(resulst_log):
        print("第{}个被试的准确率为{:.2%}".format(i+1,acc))





