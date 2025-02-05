import pytest
import dataclasses
import typing as tp
import numpy as np
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    shape: tp.Tuple[int, ...]
    index: tp.Tuple[int | slice, ...]

TEST_CASES = [
    Case(shape=(10, 10, 10), index=(1, 2, slice(1, 2))),
    Case(shape=(5,7,32), index=(3)),
    Case(shape=(1,1,2), index=(0,0)),
    Case(shape=(5,5,3), index=(slice(0,5))),
    Case(shape=(3,3,5,10), index=(0,1,slice(0,5))),
    Case(shape=(3,5,10,2,2,4), index=(2,4,9,slice(0,1))),
    Case(shape=(10,5,100,2,4), index=(9,slice(4,5))),
    Case(shape=(5,5,5,5,10), index=(0,1,2,3))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_indexing(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape) * 10
        a1 = skpp.Tensor(a)

        assert np.allclose(np.array(a1[t.index]), a[t.index])