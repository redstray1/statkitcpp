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
        a_skpp.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)

        b_skpp = a_skpp.exp()
        b_torch = a_torch.exp()

        b_torch.backward(torch.ones_like(b_torch))
        b_skpp.backward()

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_log(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10
        a_skpp = skpp.Tensor(a_np)
        a_skpp.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)

        b_skpp = a_skpp.log()
        b_torch = a_torch.log()

        b_torch.backward(torch.ones_like(b_torch))
        b_skpp.backward()

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_sqrt(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10
        a_skpp = skpp.Tensor(a_np)
        a_skpp.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)

        b_skpp = a_skpp.sqrt()
        b_torch = a_torch.sqrt()

        b_torch.backward(torch.ones_like(b_torch))
        b_skpp.backward()

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_neg(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape) * 10
        a_skpp = skpp.Tensor(a_np)
        a_skpp.requires_grad = True
        a_torch = torch.tensor(a_np, requires_grad=True)

        b_skpp = a_skpp.neg()
        b_torch = a_torch.neg()

        b_torch.backward(torch.ones_like(b_torch))
        b_skpp.backward()

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))