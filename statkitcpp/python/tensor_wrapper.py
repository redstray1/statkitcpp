# from __future__ import absolute_import
# from lsst.utils import TemplateMeta
# from _statkitcpp import Tensor32, Tensor64
# from _statkitcpp import Float32, Float64

# __all__ = []

# class Tensor(metaclass=TemplateMeta):
#     pass

# Tensor.register(Float32, Tensor32)
# Tensor.register(Float64, Tensor64)
# Tensor.alias(float, Tensor64)
# try:
#     import numpy as np
#     Tensor.alias(np.float64, Tensor64)
#     Tensor.alias(np.float32, Tensor32)
# except:
#     pass