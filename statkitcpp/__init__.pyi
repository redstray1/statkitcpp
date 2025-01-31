"""

        C++ library with various statistical algorithms with Python integration
    
"""
from __future__ import annotations
import typing
import typing_extensions
__all__ = ['Scalar', 'ScalarType', 'Tensor', 'arange', 'bool', 'float32', 'float64', 'full', 'int16', 'int32', 'int64', 'int8', 'ones', 'zeros']
class Scalar:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: float) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: float) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: bool) -> None:
        ...
class ScalarType:
    """
    Members:
    
      int8
    
      int16
    
      int32
    
      int64
    
      bool
    
      float32
    
      float64
    """
    __members__: typing.ClassVar[dict[str, ScalarType]]  # value = {'int8': <ScalarType.int8: 0>, 'int16': <ScalarType.int16: 1>, 'int32': <ScalarType.int32: 2>, 'int64': <ScalarType.int64: 3>, 'bool': <ScalarType.bool: 6>, 'float32': <ScalarType.float32: 4>, 'float64': <ScalarType.float64: 5>}
    bool: typing.ClassVar[ScalarType]  # value = <ScalarType.bool: 6>
    float32: typing.ClassVar[ScalarType]  # value = <ScalarType.float32: 4>
    float64: typing.ClassVar[ScalarType]  # value = <ScalarType.float64: 5>
    int16: typing.ClassVar[ScalarType]  # value = <ScalarType.int16: 1>
    int32: typing.ClassVar[ScalarType]  # value = <ScalarType.int32: 2>
    int64: typing.ClassVar[ScalarType]  # value = <ScalarType.int64: 3>
    int8: typing.ClassVar[ScalarType]  # value = <ScalarType.int8: 0>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Tensor:
    requires_grad: bool
    shape: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __add__(self, arg0: Scalar) -> Tensor:
        ...
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, *, shape: list[int], dtype: ScalarType = ..., requires_grad: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        ...
    @typing.overload
    def __mul__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __mul__(self, arg0: Scalar) -> Tensor:
        ...
    @typing.overload
    def __pow__(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def __pow__(self, other: Scalar) -> Tensor:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __sub__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __sub__(self, arg0: Scalar) -> Tensor:
        ...
    @typing.overload
    def __truediv__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __truediv__(self, arg0: Scalar) -> Tensor:
        ...
    @typing.overload
    def add(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def add(self, other: Scalar) -> Tensor:
        ...
    def backward(self, grad_output: Tensor | None = None, output: Tensor | None = None, retain_graph: bool = False) -> None:
        ...
    def broadcastable_to(self, other: Tensor) -> bool:
        ...
    @typing.overload
    def div(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def div(self, other: Scalar) -> Tensor:
        ...
    def exp(self) -> Tensor:
        ...
    def log(self) -> Tensor:
        ...
    def mean(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    @typing.overload
    def mul(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def mul(self, other: Scalar) -> Tensor:
        ...
    def neg(self) -> Tensor:
        ...
    @typing.overload
    def pow(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def pow(self, other: Scalar) -> Tensor:
        ...
    def prod(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    def reshape(self, shape: list[int]) -> Tensor:
        ...
    def sqrt(self) -> Tensor:
        ...
    @typing.overload
    def sub(self, other: Tensor) -> Tensor:
        ...
    @typing.overload
    def sub(self, other: Scalar) -> Tensor:
        ...
    def sum(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    def var(self, dim: int = -1, *, keepdims: bool = False) -> Tensor:
        ...
    @property
    def dtype(self) -> ScalarType:
        ...
    @property
    def grad(self) -> Tensor | None:
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
def arange(*, start: Scalar, end: Scalar, step: Scalar = ..., dtype: ScalarType = ...) -> Tensor:
    ...
def full(*, shape: list[int], value: Scalar, dtype: ScalarType = ...) -> Tensor:
    ...
def ones(*, shape: list[int], dtype: ScalarType = ...) -> Tensor:
    ...
def zeros(*, shape: list[int], dtype: ScalarType = ...) -> Tensor:
    ...
bool: ScalarType  # value = <ScalarType.bool: 6>
float32: ScalarType  # value = <ScalarType.float32: 4>
float64: ScalarType  # value = <ScalarType.float64: 5>
int16: ScalarType  # value = <ScalarType.int16: 1>
int32: ScalarType  # value = <ScalarType.int32: 2>
int64: ScalarType  # value = <ScalarType.int64: 3>
int8: ScalarType  # value = <ScalarType.int8: 0>
