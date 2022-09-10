import os
import mne
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset, random_split
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam,lr_scheduler
from tqdm import tqdm
from Algorithm.EEGModels import *
from CCAClass import CNNmodel


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

#分割数据集->可加入test-->默认8：1：1
def torch_dataset(data,labels):
    data = torch.Tensor(data)
    data = data.unsqueeze(dim=1)
    labels = torch.Tensor(labels)
    dataset = TensorDataset(data,labels)
    return dataset

def build_dataloader(train_dataset,val_dataset,test_dataset,batch_size):
    train_dataloader = DataLoader(
        train_dataset,  # 训练数据.
        sampler=RandomSampler(train_dataset),  # 打乱顺序
        batch_size=batch_size,
        drop_last=True)
    valid_dataloader = DataLoader(
        val_dataset,  # 验证数据.
        sampler=RandomSampler(val_dataset),  # 打乱顺序
        batch_size=batch_size,
        drop_last=True)
    test_dataloader = DataLoader(
        test_dataset,  # 验证数据.
        sampler=RandomSampler(test_dataset),  # 打乱顺序
        batch_size=batch_size,
        drop_last=False)
    return train_dataloader,valid_dataloader,test_dataloader

if __name__ == '__main__':
    ch_types = []
    for i in range(64):
        ch_types.append('eeg')
    ch_types.append('stim')
    IMAGES_PATH = 'img'

    batch_size = 64
    samples = 10#30
    epochs = 100
    resulst_log = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists("img/fzy_img"):
        os.makedirs("img/fzy_img")

    for index in [1,2,3,4,5]:
        X,Y = get_data(index)#450,64,200
        num_class = len(set(Y.tolist()))
        X_train,X_rem,Y_train,Y_rem = train_test_split(X,Y,test_size=0.6)
        X_val,X_test,Y_val,Y_test = train_test_split(X_rem,Y_rem,test_size=0.5)

        #数据增强
        X_train = np.concatenate((X_train,X_train))
        Y_train = np.concatenate((Y_train,Y_train))


        train_set = torch_dataset(X_train, Y_train)
        val_set = torch_dataset(X_val, Y_val)
        test_set = torch_dataset(X_test, Y_test)
        train_loader, valid_loader, test_loader = build_dataloader(train_set, val_set, test_set, batch_size)

        dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
        dataset_sizes = {'train': len(train_set), 'valid': len(val_set)}


        model = CNNmodel().to(device)
        print(model.parameters)  # 打印模型参数


        history = dict()#记录每轮acc和loss
        history['acc'],history['loss'],history['val_acc'],history['val_loss']=[],[],[],[]
        criterion = CrossEntropyLoss()  # 此损失函数 模型最后不需要softmax
        optimizer = Adam(model.parameters(), lr=1e-3, eps=1e-08)  # clipnorm=1.0, add later
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        best_epoch = 0
        best_acc = 0.0
        best_loss = float("inf")
        model_path = ""

        #训练
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 20)

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders['train']):
                torch.cuda.empty_cache()
                model.train()  # Set model to training mode
                inputs = inputs.to(device)
                labels = labels.long().to(device)  # [24,]=[batch_size,]

                outputs,_ = model(inputs)  # [24,3]=[batch_size,nclass]
                preds = torch.max(outputs, 1)[1]  # [24,]
                loss = criterion(outputs, labels)  # 计算损失

                # train特有
                loss.backward()  # 反向传播
                clip_grad_norm_(model.parameters(), max_norm=1.0)  # clipnorm=1.0, 先剪枝 再更新
                optimizer.step()  # 更新优化器权重
                scheduler.step()  # 更新学习率
                optimizer.zero_grad()  # 清空梯度

                running_corrects += torch.sum(preds == labels)
                running_loss += loss.item()

            epoch_acc = running_corrects / dataset_sizes['train']
            epoch_loss = running_loss / dataset_sizes['train']
            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            history['acc'].append(epoch_acc)
            history['loss'].append(epoch_loss)

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders['valid']):
                torch.cuda.empty_cache()
                model.eval()  # Set model to training mode
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                with torch.no_grad():  # 停用累加梯度
                    outputs,_ = model(inputs)
                    preds = torch.max(outputs, 1)[1]  # [24,]
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes['valid']
            epoch_acc = running_corrects / dataset_sizes['valid']

            print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            history['val_acc'].append(epoch_acc)
            history['val_loss'].append(epoch_loss)

            if epoch_loss < best_loss:
                best_epoch = epoch
                best_acc = epoch_acc
                best_loss = epoch_loss

                # if epoch_acc > best_acc:
                #     best_epoch = epoch
                #     best_acc = epoch_acc
                #     best_loss = epoch_loss
                # 保存模型
                model_path = "saved/best_model_torch_{}.h5".format(index)
                torch.save(model.state_dict(), model_path)
                print("Checkpoint Saved")
            print()


        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Best Epoch of is {}'.format(best_epoch + 1))


        plt.plot(history['acc'],label='acc')
        plt.plot(history['loss'],label='loss')
        plt.plot(history['val_acc'],label='val_acc')
        plt.plot(history['val_loss'],label='val_loss')
        plt.title('rec_{}'.format(index))
        plt.legend()
        plt.savefig("img/fzy_img/rec_torch_{}.png".format(index), dpi=400)
        plt.show()

        #测试
        y_true = np.array([])
        y_pred = np.array([])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)  # 放入显存，如果有
            labels = labels.to(device).long().cpu()
            y_true = np.append(y_true, labels.numpy())
            with torch.no_grad():  # 停用累加梯度
                outputs,_ = model(inputs)
                preds = torch.max(outputs, 1)[1].cpu()
                y_pred = np.append(y_pred, preds.numpy())

        acc_arr = accuracy_score(y_true, y_pred)
        rec_arr = recall_score(y_true, y_pred, average='macro')
        pre_arr = precision_score(y_true, y_pred, average='macro')
        f1_arr = f1_score(y_true, y_pred, average='macro')
        f1_arr_mic = f1_score(y_true, y_pred, average='micro')
        print("Accuracy: ", acc_arr)
        print("Recall: ", rec_arr)
        print("Precision: ", pre_arr)
        print("F1 score: ", f1_arr)
        print("F1 score Micro: ", f1_arr_mic)


        resulst_log.append(acc_arr)#记录每个被试 测试集的acc

    for i,acc in enumerate(resulst_log):
        print("第{}个被试的准确率为{:.2%}".format(i+1,acc))





