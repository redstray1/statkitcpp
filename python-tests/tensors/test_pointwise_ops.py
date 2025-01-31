import pytest
import dataclasses
import typing as tp
import numpy as np
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(shape=(1,)),
    Case(shape=(2,)),
    Case(shape=(3,3)),
    Case(shape=(3,1,3)),
    Case(shape=(1,3,2)),
    Case(shape=(5,3,3,3,2)),
    Case(shape=(5,1,3,2,1,1)),
    Case(shape=(100,2,15,1,2,4)),
    Case(shape=(6,5,2,23,4))
]

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_exp(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10
        a_skpp = skpp.Tensor(a_np)

        assert np.allclose(np.exp(a_np), a_skpp.exp())

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_log(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10 + 5
        a_skpp = skpp.Tensor(a_np)

        assert np.allclose(np.log(a_np), a_skpp.log())

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_sqrt(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10 + 5
        a_skpp = skpp.Tensor(a_np)

        assert np.allclose(np.sqrt(a_np), a_skpp.sqrt())

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_neg(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10
        a_skpp = skpp.Tensor(a_np)

        assert np.allclose(np.negative(a_np), a_skpp.neg())