import sys
import os

sys.path.append('.')

from Algorithm.AlgorithmImplementMI import AlgorithmImplementMI
from Framework.AlgorithmSystemManager import AlgorithmSystemManager
from Task.TaskManagerProxy import TaskManagerProxy
from Task.TaskManagerMI import TaskManagerMI
from Framework.loadData import load_data


if __name__ == '__main__':
    # 系统框架实例
    algorithm_sys_mng = AlgorithmSystemManager()
    # MI赛题实例
    task_mng = TaskManagerMI()
    # 向系统框架注入MI赛题
    algorithm_sys_mng.add_task(task_mng)
    # 创建该赛题的代理对象
    task_mng_proxy = TaskManagerProxy(task_mng)
    # 向系统框架注入赛题代理对象
    algorithm_sys_mng.add_task_proxy(task_mng_proxy)
    # 加载MI数据
    data_path = os.path.join(os.getcwd(), 'TrainData')
    mi_data_path = os.path.join(data_path, 'MI')
    # 读取所有被试者的MI数据
    subject_data_model_set = load_data(mi_data_path)
    # 向赛题注入MI数据
    algorithm_sys_mng.add_data(subject_data_model_set)
    # 算法实例
    algorithm_impl_mi = AlgorithmImplementMI()
    # # 向系统框架注入算法
    algorithm_sys_mng.add_algorithm(algorithm_impl_mi)
    print("1")
    # # 执行算法
    algorithm_sys_mng.run()
    print("2")
    # 清除算法
    algorithm_sys_mng.clear_algorithm()
    # 清除数据
    algorithm_sys_mng.clear_data()
    # 清除赛题
    algorithm_sys_mng.clear_task()
