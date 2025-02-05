import pytest
import dataclasses
import typing as tp
import numpy as np
import torch
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(shape=(1,)),
    Case(shape=(2,)),
    Case(shape=(2,2)),
    Case(shape=(3,3)),
    Case(shape=(3,1,3)),
    Case(shape=(1,3,2)),
    # Case(shape=(5,3,3,3,2)),
    # Case(shape=(5,1,3,2,1,1)),
    # Case(shape=(10,2,15,1)),
    # Case(shape=(6,5,2,23,4))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_scalar1(t: Case) -> None:
    for _ in range(10):
        a = np.random.random(t.shape)

        a1 = skpp.Tensor(a)
        a1.requires_grad = True
        a2 = torch.tensor(a, requires_grad=True)

        b1 = a1 * 2 + 3
        b2 = a2 * 2 + 3

        b1.backward()
        b2.backward(torch.ones_like(b2))

        assert np.allclose(np.array(a1.grad), a2.grad.numpy())
