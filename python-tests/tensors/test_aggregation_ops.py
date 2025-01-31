import pytest
import dataclasses
import typing as tp
import numpy as np
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    shape: tp.Tuple[int, ...]
    dim: int

TEST_CASES = [
    Case(shape=(1,1), dim=-1),
    Case(shape=(2,2), dim=-1),
    Case(shape=(1,), dim=-1),
    Case(shape=(1,1,3), dim=2),
    Case(shape=(4,5,15,2,7), dim=2),
    Case(shape=(100,20,5,2), dim=0),
    Case(shape=(100,20,5,2), dim=3),
    Case(shape=(5,),dim=0),
    Case(shape=(6,1,6),dim=1),
    Case(shape=(5,5,3,5,4), dim=0),
    Case(shape=(4,4,7,11,1,1,3), dim=0),
    Case(shape=(4,4,7,11,1,1,3), dim=1),
    Case(shape=(4,4,7,11,1,1,3), dim=2),
    Case(shape=(4,4,7,11,1,1,3), dim=3),
    Case(shape=(4,4,7,11,1,1,3), dim=4),
    Case(shape=(4,4,7,11,1,1,3), dim=5),
    Case(shape=(4,4,7,11,1,1,3), dim=-1),
    Case(shape=(5,1,1,15),dim=-1),
    Case(shape=(2,2,3,1,3,1,1),dim=1)
]

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_sum(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape)
        a1 = skpp.Tensor(a)
        assert np.allclose(a.sum(axis=t.dim), a1.sum(t.dim))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_mean(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape)
        a1 = skpp.Tensor(a)
        assert np.allclose(a.mean(axis=t.dim), a1.mean(t.dim))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_var(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape)
        a1 = skpp.Tensor(a)
        if t.shape[t.dim] <= 1:
            with pytest.raises(RuntimeError):
                a1.var(t.dim)
            continue
        assert np.allclose(a.var(axis=t.dim), a1.var(t.dim))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_prod(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape)
        a1 = skpp.Tensor(a)
        assert np.allclose(a.prod(axis=t.dim), a1.prod(t.dim))