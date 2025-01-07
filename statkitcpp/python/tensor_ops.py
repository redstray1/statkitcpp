from _statkitcpp import DataType
from _statkitcpp import full32, full64, ones32, ones64, zeros32, zeros64
from .tensor_wrapper import Tensor

def full(shape, value, dtype: DataType = DataType.Float32) -> Tensor:
    if dtype == DataType.Float32:
        return full32(shape, value)
    elif dtype == DataType.Float64:
        return full64(shape, value)
    else:
        raise NotImplementedError

def ones(shape, dtype: DataType = DataType.Float32) -> Tensor:
    if dtype == DataType.Float32:
        return ones32(shape)
    elif dtype == DataType.Float64:
        return ones64(shape)
    else:
        raise NotImplementedError

def zeros(shape, dtype: DataType = DataType.Float32) -> Tensor:
    if dtype == DataType.Float32:
        return zeros32(shape)
    elif dtype == DataType.Float64:
        return zeros64(shape)
    else:
        raise NotImplementedError