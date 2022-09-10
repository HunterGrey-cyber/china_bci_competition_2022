from abc import abstractmethod, ABCMeta


class AlgorithmInterface(metaclass=ABCMeta):
    def __init__(self):
        self.task = None

    @abstractmethod
    def run(self):
        pass

    def set_task(self, task_mng):
        self.task = task_mng
