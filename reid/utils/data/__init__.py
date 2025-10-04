from __future__ import absolute_import

from .base_dataset import BaseImageDataset
from .preprocessor import Preprocessor
from .dataset import Dataset


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)
        # 为新的 epoch 重新初始化迭代器
        # 在每个 epoch 开始时调用，确保从数据的起始位置重新迭代（避免上一个 epoch 未迭代完的残留状态）

    def next(self):
        try:
            return next(self.iter) # 尝试从当前迭代器获取下一个数据批次
        except: # 若迭代器耗尽（抛出 StopIteration），重新创建迭代器并获取第一个批次
            self.iter = iter(self.loader)
            return next(self.iter)
