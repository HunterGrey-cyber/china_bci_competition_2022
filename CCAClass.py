import numpy as np
import torch, os
from torch.autograd import Variable
import torch.nn as nn



class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        prob = 0
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 10, (1, 23), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),
        #     nn.Conv2d(10, 30, (59, 1), stride=1, padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 10X1X234
        #     nn.Conv2d(30, 30, (1, 17), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 修改了卷积核的尺寸
        #     nn.MaxPool2d((1, 6), stride=(1, 6)),
        #     nn.Conv2d(30, 30, (1, 7), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 修改了卷积核的尺寸
        #     nn.MaxPool2d((1, 6), stride=(1, 6)),
        # )
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, (1, 5), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),#23-->5
            nn.Conv2d(10, 30, (64, 1), stride=1, padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 10X1X234
            nn.Conv2d(30, 30, (1, 3), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 修改了卷积核的尺寸 17-->3
            nn.MaxPool2d((1, 1), stride=(1, 1)),
            nn.Conv2d(30, 30, (1, 2), stride=(1, 1), padding=0), nn.ELU(), nn.Dropout2d(p=prob),  # 修改了卷积核的尺寸
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=30 * 1 * 1, out_features=3),
        )

    def forward(self, img):

        feature_repr = self.model(img)#16,30,1,1
        # print(feature_repr.size())
        feature = feature_repr.view(feature_repr.size(0), -1)
        out_cls = self.classifier(feature)

        return out_cls, feature

class CCAClass:
    def __init__(self):
        self.testModel = CNNmodel()

    def recognize(self, recogdata):
        # odata = np.zeros((1, 1, 59, 100))
        # k = len(recogdata)
        # for i in range(len(recogdata)):
        #     indata = recogdata[i]
        #     indata = indata[np.newaxis, :]
        #     indata = indata[np.newaxis, :]
        #     odata = np.concatenate((odata, indata))
        #
        # D = odata[1:, :, :, :].copy()
        # data = torch.from_numpy(D)

        data = recogdata.unsqueeze(dim=1)
        indata = Variable(data.type(torch.FloatTensor))
        Cls, _ = self.testModel(indata)
        # y_pred = torch.max(Cls, 1)[1]
        # rcls = list(y_pred.detach().numpy())
        # print(rcls)
        # maxlabel = max(rcls, key=rcls.count)

        # result = maxlabel + 201
        return Cls

if __name__ == '__main__':
    model = CCAClass()
    x= torch.randn((16,64, 10))
    result = model.recognize(x)
    print(result)
