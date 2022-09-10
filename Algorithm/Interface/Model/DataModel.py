class DataModel:
    def __init__(self):
        # 数据
        self.data = None

        # 当前数据块相对本次Session起点位置(每名受试者一次采集视为一个Session)
        # 重置为0, 则认为开始新的block
        self.start_pos = 0

        # 受试者id
        self.subject_id = 0

        # 程序终止标志
        self.finish_flag = False
