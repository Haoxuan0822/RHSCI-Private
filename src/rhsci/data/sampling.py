"""按 seed/epoch 重建顺序，不消耗模型或 DataLoader 的随机数流。"""

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler):
    """返回样本索引；include_epoch=True 时显式返回 (epoch,index) 给数据集。"""

    def __init__(self, data_source, seed, *, include_epoch=False):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.include_epoch = include_epoch

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self), generator=generator).tolist()
        if self.include_epoch:
            return iter((self.epoch, index) for index in indices)
        return iter(indices)


class EpochSampler(EpochRandomSampler):
    """ACT/GR00T 数据集使用的显式 epoch 索引接口。"""

    def __init__(self, dataset, seed):
        super().__init__(dataset, seed, include_epoch=True)
