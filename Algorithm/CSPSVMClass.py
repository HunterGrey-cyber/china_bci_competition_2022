import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
# from Algorithm.EEGModels import EEGNet
import warnings
warnings.filterwarnings("ignore")

rootdir = os.path.dirname(os.path.abspath(__file__))

# 仅供测试所用
class CSPSVMClass:
    def band_Filter(self, data, bandID):
        bandVue = np.array([[1, 48], [8,30]])
        b, a = signal.butter(6, [2*bandVue[bandID-1, 0]/250, 2*bandVue[bandID-1, 1]/250], 'bandpass', analog=True)
        for iChan in range(data.shape[0]):
            data[iChan, :] = signal.filtfilt(b, a, data[iChan, :])
        return data

    def func_CSP(self, data, CSPMatrix, nbFilterPairs):
        features = np.zeros((2 * nbFilterPairs))
        Filter = CSPMatrix[np.r_[:nbFilterPairs, -nbFilterPairs:0]]
        projectedTrial = np.dot(Filter, data)
        variances = np.var(projectedTrial, 1)
        for i in range(len(variances)):
            features[i] = np.log(variances[i])
        return features

    def getModel(self, personID):
        mod_max_path = rootdir + '/model'
        mod_max_list = os.listdir(mod_max_path)
        # mod_max_list.sort(key=lambda x: int(x[6:7]))
        modPath = os.path.join(mod_max_path, mod_max_list[personID-1])
        mod = joblib.load(modPath)
        return mod

    def recognize(self, data, personID):
        # 通道选择
        chans = list(range(1, 60))
        chans = [i - 1 for i in chans]
        data = data[chans, :]
        data = self.band_Filter(data, personID)
        mod = self.getModel(personID)
        test_feature12 = self.func_CSP(data, mod[0], 3)
        test_feature13 = self.func_CSP(data, mod[1], 3)
        test_feature23 = self.func_CSP(data, mod[2], 3)
        pro12 = mod[3].predict_proba(test_feature12.reshape(1, -1))
        pro13 = mod[4].predict_proba(test_feature13.reshape(1, -1))
        pro23 = mod[5].predict_proba(test_feature23.reshape(1, -1))
        pro1 = pro12[0, 0] + pro13[0, 0]
        pro2 = pro12[0, 1] + pro23[0, 0]
        pro3 = pro13[0, 1] + pro23[0, 1]
        pro = [pro1, pro2, pro3]
        result = pro.index(max(pro)) + 201
        return result


# class EEgnetClass:
#     def band_Filter(self,data,bandID):
#         bandVue = np.array([[1,48],[8,30]])
#         b, a = signal.butter(6, [2*bandVue[bandID-1, 0]/250, 2*bandVue
#                                  [bandID-1, 1]/250], 'bandpass', analog=True)
#         for iChan in range(data.shape[0]):
#             data[iChan, :] = signal.filtfilt(b, a, data[iChan, :])
#         return data
    
#     def recognize(self,data,personID):
#         model = EEGNet(nb_classes = 3, Chans = 64, Samples = 1000,
#                dropoutRate = 0.5)
        
#         model.compile(loss='categorical_crossentropy', optimizer='adam', 
#               metrics = ['accuracy'])
        
#         weights_path = ''
#         model.load_weights(weights_path)
#         probs  = model.predict(data)
#         preds  = probs.argmax(axis = -1)
#         return preds
