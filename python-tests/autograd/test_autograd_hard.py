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
    c_shape: tp.Tuple[int, ...]
    d_shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(a_shape=(10, 10), b_shape=(10, 10), c_shape=(10, 10), d_shape=(10, 10)),
    Case(a_shape=(10, 1), b_shape=(1, 10), c_shape=(10, 10), d_shape=(1, 1)),
    Case(a_shape=(2,3,1,5), b_shape=(1,1,3,5), c_shape=(1,3,3,1), d_shape=(2,1,1,5))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case1(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape)
        b = np.random.random(t.b_shape)
        c = np.random.random(t.c_shape)
        d = np.random.random(t.d_shape)

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 + b1) * (c1 / d1)
        e2 = (a2 + b2) * (c2 / d2)

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case2(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape)
        b = np.random.random(t.b_shape)
        c = np.random.random(t.c_shape)
        d = np.random.random(t.d_shape)

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 - b1) * (c1 / d1).exp()
        e2 = (a2 - b2) * (c2 / d2).exp()

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case3(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape)
        b = np.random.random(t.b_shape)
        c = np.random.random(t.c_shape)
        d = np.random.random(t.d_shape)

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 + b1) / (c1 ** d1)
        e2 = (a2 + b2) / (c2 ** d2)

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case4(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape)
        b = np.random.random(t.b_shape)
        c = np.random.random(t.c_shape)
        d = np.random.random(t.d_shape)

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 + b1).mean(keepdims=True) * (c1 / d1)
        e2 = (a2 + b2).mean(dim=-1,keepdim=True) * (c2 / d2)

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case5(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape) + 2
        b = np.random.random(t.b_shape) + 2
        c = np.random.random(t.c_shape) + 2
        d = np.random.random(t.d_shape) + 2

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 + b1).mean() + (c1 / d1).mean()
        e2 = (a2 + b2).mean(dim=-1) + (c2 / d2).mean(dim=-1)

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_basic_hard_case6(t: Case):
    for _ in range(10):
        a = np.random.random(t.a_shape) + 2
        b = np.random.random(t.b_shape) + 2
        c = np.random.random(t.c_shape) + 2
        d = np.random.random(t.d_shape) + 2

        a1 = skpp.Tensor(a)
        b1 = skpp.Tensor(b)
        c1 = skpp.Tensor(c)
        d1 = skpp.Tensor(d)

        a1.requires_grad = True
        b1.requires_grad = True
        c1.requires_grad = True
        d1.requires_grad = True

        a2 = torch.tensor(a, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)
        c2 = torch.tensor(c, requires_grad=True)
        d2 = torch.tensor(d, requires_grad=True)

        e1 = (a1 * b1).mean() + (c1 * d1).var()
        e2 = (a2 * b2).mean(dim=-1) + (c2 * d2).var(dim=-1)

        e1.backward()
        e2.backward(torch.ones_like(e2))

        assert np.allclose(a2.grad.numpy(), np.array(a1.grad))
        assert np.allclose(b2.grad.numpy(), np.array(b1.grad))
        assert np.allclose(c2.grad.numpy(), np.array(c1.grad))
        assert np.allclose(d2.grad.numpy(), np.array(d1.grad))

def test_linear_layer_case():
    for _ in range(10):
        X = np.random.random((100, 20)).astype(np.float32)
        A = np.random.random((20, 50)).astype(np.float32)
        b = np.random.random((50,)).astype(np.float32)

        X1 = skpp.Tensor(X)
        A1 = skpp.Tensor(A)
        b1 = skpp.Tensor(b)

        A1.requires_grad = True
        b1.requires_grad = True

        X2 = torch.tensor(X)
        A2 = torch.tensor(A, requires_grad=True)
        b2 = torch.tensor(b, requires_grad=True)

        Y1 = X1 @ A1 + b1
        Y2 = X2 @ A2 + b2

        Y1.backward()
        Y2.backward(torch.ones_like(Y2))

        assert np.allclose(np.array(A2.grad.numpy()), np.array(A1.grad))
        assert np.allclose(np.array(b2.grad.numpy()), np.array(b1.grad))
