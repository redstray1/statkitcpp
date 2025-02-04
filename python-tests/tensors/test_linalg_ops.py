import pytest
import dataclasses
import typing as tp
import numpy as np
import torch
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    a_shape: tp.Tuple[int, ...]
    b_shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(a_shape=(5,), b_shape=(5,)),
    Case(a_shape=(1,), b_shape=(1,)),
    Case(a_shape=(10,10),b_shape=(10,)),
    Case(a_shape=(10,2), b_shape=(2,5)),
    Case(a_shape=(5,1), b_shape=(1,5)),
    Case(a_shape=(10,10), b_shape=(10,10)),
    Case(a_shape=(10, 13), b_shape=(13, 7)),
    Case(a_shape=(2, 89), b_shape=(89, 2)),
    Case(a_shape=(2,3,10,10), b_shape=(10,)),
    Case(a_shape=(100,), b_shape=(2,3,100,100)),
    Case(a_shape=(10,2),b_shape=(3,2,5)),
    Case(a_shape=(10,2),b_shape=(4,5,3,2,5))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_dot(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape[0])
        b: np.ndarray = np.random.random(t.a_shape[0])

        a1: skpp.Tensor = skpp.Tensor(a)
        b1: skpp.Tensor = skpp.Tensor(b)

        assert np.allclose(np.array(a1.dot(b1)), a.dot(b))

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_matmul(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape)
        b: np.ndarray = np.random.random(t.b_shape)

        a1: skpp.Tensor = skpp.Tensor(a)
        b1: skpp.Tensor = skpp.Tensor(b)

        a2: torch.Tensor = torch.tensor(a)
        b2: torch.Tensor = torch.tensor(b)

        assert np.allclose(np.array(a1 @ b1), np.array(torch.matmul(a2, b2)))