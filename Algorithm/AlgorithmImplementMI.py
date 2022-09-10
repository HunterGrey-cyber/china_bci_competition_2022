from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel
from Algorithm.CSPSVMClass import CSPSVMClass
from scipy import signal
import numpy as np
import math


class AlgorithmImplementMI(AlgorithmInterface):
    # 类属性：范式名称
    PARADIGMNAME = 'MI'
    
    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        samp_rate = 250
        # 选择导联编号
        self.select_channel = list(range(1, 60))
        self.select_channel = [i - 1 for i in self.select_channel]
        # 想象开始的trigger（由于240作为trial开始被占用，这里使用系统预留trigger:249）
        self.trial_stimulate_mask_trig = 249
        # trial结束trigger
        self.trial_end_trig = 241
        # 计算时间
        cal_time = 4
        # 计算长度
        self.cal_len = cal_time * samp_rate
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        #初始化方法
        self.method = CSPSVMClass()
        
    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        while not end_flag:
            data_model = self.task.get_data()
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)
            else:
                # 计算模式，则进行处理
                cal_flag, result = self.__cal_proc(data_model)
                # 如果有结果，则进行报告
                if result is not None:
                    report_model = ReportModel()
                    report_model.result = result
                    self.task.report(report_model)
                    # 清空缓存
                    self.__clear_cache()
            end_flag = data_model.finish_flag

    def __idle_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        # 脑电数据
        eeg_data = data[0: -1, :]
        if len(trigger_idx) > 0:
            # 有trial开始trigger则进行计算
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos: eeg_data.shape[1]]
        else:
            # 没有trial开始trigger则
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        personID = data_model.subject_id
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 如果trigger为空，表示依然在当前试次中，根据数据长度判断是否计算
        if len(trigger_idx) == 0:
            # 当已缓存的数据大于等于所需要使用的计算数据时，进行计算
            if self.cache_data.shape[1] >= self.cal_len:
            # 获取所需计算长度的数据
                self.cache_data = self.cache_data[:, 0: int(self.cal_len)]
                # 滤波处理
                use_data = self.__preprocess(self.cache_data)
                # 开始计算，返回计算结果
                result = self.method.recognize(use_data, personID)
                # 停止计算模式
                cal_flag = False
            else:
                # 反之继续采集数据
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                result = None
                cal_flag = True
        # 下一试次已经开始,需要强制结束计算
        else:
            # 下一个trial开始trigger的位置
            next_trial_start_trig_pos = trigger_idx[0]
            # 如果拼接该数据包中部分的数据后，可以满足所需要的计算长度，则拼接数据达到所需要的计算长度
            # 如果拼接完该trial的所有数据后仍无法满足所需要的数据长度，则只能使用该trial的全部数据进行计算
            use_len = min(next_trial_start_trig_pos, self.cal_len - self.cache_data.shape[1])
            self.cache_data = np.append(self.cache_data, eeg_data[:, 0: use_len], axis=1)
            # 滤波处理
            use_data = self.__preprocess(self.cache_data)
            # 开始计算
            result = self.method.recognize(use_data, personID)
            # 开始新试次的计算模式
            cal_flag = True
            # 清除缓存的数据
            self.__clear_cache()
            # 添加新试次数据
            self.cache_data = eeg_data[:, next_trial_start_trig_pos: eeg_data.shape[1]]
        return cal_flag, result


    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __clear_cache(self):
        self.cache_data = np.zeros((64, 0))


    def __preprocess(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        filter_data = signal.filtfilt(self.filterB, self.filterA, data)
        return filter_data
