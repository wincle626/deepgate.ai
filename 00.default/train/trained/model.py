import torch.nn as nn

from dg.base_model import DLGModel
from dg.dtype import DType
from dg.layer import (
    Flatten,
    Norm,
    QuantAvgPool2d,
    QuantConv2d,
    QuantDepthwiseConv2d,
    QuantLinear,
)


class Model(DLGModel):
    NORM_MEAN = -1.6787   # MFCC mean over training set
    NORM_STD = 7.6337     # MFCC std over training set

    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def build_model_graph(self):
        return [
            Norm(mean=self.NORM_MEAN, std=self.NORM_STD, dtype_in=DType.FLOAT32),
            QuantConv2d(1, 64, (10, 4), (2, 2), (1, 1, 4, 5), bias=True, act_func="relu", bn=True),
            *self._ds_block(64),
            *self._ds_block(64),
            *self._ds_block(64),
            *self._ds_block(64),
            QuantAvgPool2d((25, 5), (25, 5), (0, 0)),
            Flatten(),
            QuantLinear(64, self.num_classes, bias=True),
        ]

    def _ds_block(self, channels):
        return [
            QuantDepthwiseConv2d(channels, (3, 3), (1, 1), (1, 1), bias=True, act_func="relu", bn=True),
            QuantConv2d(channels, channels, (1, 1), (1, 1), (0, 0), bias=True, act_func="relu", bn=True),
        ]
