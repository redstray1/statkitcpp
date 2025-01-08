from _statkitcpp import Float32, Float64
from _statkitcpp import full32, full64, ones32, ones64, zeros32, zeros64
from .tensor_wrapper import Tensor

def full(shape, value, dtype = Float32) -> Tensor:
    if dtype == Float32:
        return full32(shape, value)
    elif dtype == Float64 or dtype == float:
        return full64(shape, value)
    else:
        raise NotImplementedError

def ones(shape, dtype = Float32) -> Tensor:
    if dtype == Float32:
        return ones32(shape)
    elif dtype == Float64 or dtype == float:
        return ones64(shape)
    else:
        raise NotImplementedError

def zeros(shape, dtype = Float32) -> Tensor:
    if dtype == Float32:
        return zeros32(shape)
    elif dtype == Float64 or dtype == float:
        return zeros64(shape)
    else:
        raise NotImplementedError