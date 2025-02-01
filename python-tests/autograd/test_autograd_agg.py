import pytest
import dataclasses
import typing as tp
import numpy as np
import torch
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
        a_np = np.random.random(t.shape)
        a_skpp = skpp.Tensor(a_np)
        a_torch = torch.tensor(a_np, requires_grad=True)

        a_skpp.requires_grad = True
        
        c_skpp = a_skpp.sum(t.dim)
        c_torch = a_torch.sum(t.dim)
        c_skpp.backward()
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
        
@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_mean(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape)
        a_skpp = skpp.Tensor(a_np)
        a_torch = torch.tensor(a_np, requires_grad=True)

        a_skpp.requires_grad = True
        
        c_skpp = a_skpp.mean(t.dim)
        c_torch = a_torch.mean(t.dim)
        c_skpp.backward()
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))

@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_var(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape)
        a_skpp = skpp.Tensor(a_np)
        a_torch = torch.tensor(a_np, requires_grad=True)

        a_skpp.requires_grad = True
        if t.shape[t.dim] > 1:
            c_skpp = a_skpp.var(t.dim)
            c_torch = a_torch.var(t.dim)
        else:
            with pytest.raises(RuntimeError):
                c_skpp = a_skpp.var(t.dim)
                c_torch = a_torch.var(t.dim)
            continue
        c_skpp.backward()
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))


@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_max(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape)
        a_skpp = skpp.Tensor(a_np)
        a_torch = torch.tensor(a_np, requires_grad=True)

        a_skpp.requires_grad = True
        
        c_skpp = a_skpp.max(t.dim)
        c_torch = a_torch.max(t.dim)[0]
        c_skpp.backward()
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))


@pytest.mark.parametrize('t',TEST_CASES, ids=str)
def test_min(t: Case) -> None:
    for _ in range(10):
        a_np = np.random.random(t.shape)
        a_skpp = skpp.Tensor(a_np)
        a_torch = torch.tensor(a_np, requires_grad=True)

        a_skpp.requires_grad = True
        
        c_skpp = a_skpp.min(t.dim)
        c_torch = a_torch.min(t.dim)[0]
        c_skpp.backward()
        c_torch.backward(torch.ones_like(c_torch))

        assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))