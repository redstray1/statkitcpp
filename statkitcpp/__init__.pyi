"""

        C++ library with various statistical algorithms with Python integration
    
"""
from __future__ import annotations
import typing
import typing_extensions
__all__ = ['Float32', 'Float64', 'Tensor', 'full', 'ones', 'zeros']
class Float32:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class Float64:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class Tensor:
    requires_grad: bool
    shape: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, *, shape: list[int], dtype: str = 'float32', requires_grad: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def broadcastable_to(self, other: Tensor) -> bool:
        ...
    def mean(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    def reshape(self, new_shape: list[int]) -> None:
        ...
    def sum(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    @property
    def dtype(self) -> str:
        ...
    @property
    def itemsize(self) -> int:
        ...
    @property
    def nbytes(self) -> int:
        ...
    @property
    def ndim(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
    @property
    def strides(self) -> list[int]:
        ...
def full(*, shape: list[int], value: typing.Any, dtype: str = 'float32') -> Tensor:
    ...
def ones(*, shape: list[int], dtype: str = 'float32') -> Tensor:
    ...
def zeros(*, shape: list[int], dtype: str = 'float32') -> Tensor:
    ...
