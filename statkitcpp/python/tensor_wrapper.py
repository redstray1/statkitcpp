from __future__ import absolute_import
from lsst.utils import TemplateMeta
from _statkitcpp import Tensor32, Tensor64
from _statkitcpp import DataType

__all__ = []

class Tensor(metaclass=TemplateMeta):
    pass

Tensor.register(DataType.Float32, Tensor32)
Tensor.register(DataType.Float64, Tensor64)

Tensor.alias("F", Tensor32)
Tensor.alias("D", Tensor64)