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
    Case(a_shape=(1,1), b_shape=(1)),
    Case(a_shape=(5, 1), b_shape=(1)),
    Case(a_shape=(1, 1), b_shape=(2, 1)),
    Case(a_shape=(5, 5, 5), b_shape=(1, 1, 5)),
    Case(a_shape=(1, 4, 2), b_shape=(1, 1)),
    Case(a_shape=(4, 2, 4), b_shape=(2, 4)),
    Case(a_shape=(1,2,3,5), b_shape=(2,1,5)),
    Case(a_shape=(15,3,5), b_shape=(3,5)),
    Case(a_shape=(8,1,6,1),b_shape=(7,1,5))
]

@pytest.mark.parametrize('t',TEST_CASES,ids=str)
def test_autograd_add(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.a_shape)
        b_np = np.random.random(t.b_shape)

        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)

        a_skpp = skpp.Tensor(a_np)
        b_skpp = skpp.Tensor(b_np)

        a_skpp.requires_grad = True
        b_skpp.requires_grad = True

        c_skpp = a_skpp + b_skpp
        c_skpp.backward()
        c_torch = a_torch + b_torch
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
        assert np.allclose(b_torch.grad.numpy(), np.array(b_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES,ids=str)
def test_autograd_mul(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.a_shape)
        b_np = np.random.random(t.b_shape)

        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)

        a_skpp = skpp.Tensor(a_np)
        b_skpp = skpp.Tensor(b_np)

        a_skpp.requires_grad = True
        b_skpp.requires_grad = True

        c_skpp = a_skpp * b_skpp
        c_skpp.backward()
        c_torch = a_torch * b_torch
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
        assert np.allclose(b_torch.grad.numpy(), np.array(b_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES,ids=str)
def test_autograd_div(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.a_shape)
        b_np = np.random.random(t.b_shape)

        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)

        a_skpp = skpp.Tensor(a_np)
        b_skpp = skpp.Tensor(b_np)

        a_skpp.requires_grad = True
        b_skpp.requires_grad = True

        c_skpp = a_skpp / b_skpp
        c_skpp.backward()
        c_torch = a_torch / b_torch
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
        assert np.allclose(b_torch.grad.numpy(), np.array(b_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES,ids=str)
def test_autograd_pow(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.a_shape)
        b_np = np.random.random(t.b_shape)

        a_torch = torch.tensor(a_np, requires_grad=True)
        b_torch = torch.tensor(b_np, requires_grad=True)

        a_skpp = skpp.Tensor(a_np)
        b_skpp = skpp.Tensor(b_np)

        a_skpp.requires_grad = True
        b_skpp.requires_grad = True

        c_skpp = a_skpp ** b_skpp
        c_skpp.backward()
        c_torch = a_torch ** b_torch
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
        assert np.allclose(b_torch.grad.numpy(), np.array(b_skpp.grad))