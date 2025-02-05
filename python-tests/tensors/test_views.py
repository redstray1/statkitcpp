import pytest
import dataclasses
import typing as tp
import numpy as np
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    shape: tp.Tuple[int, ...]
    new_shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(shape=(10, 10), new_shape=(100,)),
    Case(shape=(5,5,2), new_shape=(5,10)),
    Case(shape=(2,3,5,6,7), new_shape=(6,30,7))
]

TEST_CASES_TRANSPOSED = [
    Case(shape=(10, 10), new_shape=(100,10)),
    Case(shape=(5,5,2), new_shape=(5,10)),
    Case(shape=(10,10,2,3,10), new_shape=(10)),
    Case(shape=(10, 5, 5, 10), new_shape=(1,))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_reshape(t: Case):
    for _ in range(10):
        a = np.random.random(t.shape)
        
        a1 = skpp.Tensor(a)
        a1 = a1.reshape(t.new_shape)

        a = a.reshape(t.new_shape)

        assert np.allclose(np.array(a1), a)

@pytest.mark.parametrize('t', TEST_CASES_TRANSPOSED, ids=str)
def test_transpose(t: Case):
    for _ in range(10):
        a = np.random.random(t.shape)
        
        a1 = skpp.Tensor(a)
        b1 = a1.transpose()
        b = a.transpose([i for i in range(a.ndim - 2)] + [a.ndim - 1, a.ndim - 2])

        assert np.allclose(np.array(b1), b)