
import pytest
import dataclasses
import typing as tp
import numpy as np
import statkitcpp as skpp

@dataclasses.dataclass
class Case:
    a_shape: tp.Tuple[int, ...]
    b_shape: tp.Tuple[int, ...]

TEST_CASES = [
    Case(a_shape=(5, 1), b_shape=(1)),
    Case(a_shape=(1, 1), b_shape=(2, 1)),
    Case(a_shape=(5, 5, 5), b_shape=(1, 1, 5)),
    Case(a_shape=(1, 4, 2), b_shape=(1, 1)),
    Case(a_shape=(4, 2, 4), b_shape=(2, 4)),
    Case(a_shape=(1,2,3,5), b_shape=(2,1,5))
]

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_binary_ops(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape)
        b: np.ndarray = np.random.random(t.b_shape)

        a1: skpp.Tensor = skpp.Tensor(a)
        b1: skpp.Tensor = skpp.Tensor(b)

        assert np.allclose(np.array(a1 + b1), a + b)
        assert np.allclose(np.array(a1 * b1), a * b)
        assert np.allclose(np.array(a1 ** b1), a ** b)
        assert np.allclose(np.array(a1 - b1), a - b)


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_binary_ops_tough(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape) * 10
        b: np.ndarray = np.random.random(t.b_shape) * 10 - 5

        a1: skpp.Tensor = skpp.Tensor(a)
        b1: skpp.Tensor = skpp.Tensor(b)

        assert np.allclose(np.array(a1 + b1), a + b)
        assert np.allclose(np.array(a1 * b1), a * b)
        assert np.allclose(np.array(a1 ** b1), a ** b)
        assert np.allclose(np.array(a1 - b1), a - b)

@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_pointwise_ops(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape)
        a1: skpp.Tensor = skpp.Tensor(a)
        assert np.allclose(np.array(a1.exp()), np.exp(a))
        assert np.allclose(np.array(a1.neg()), np.negative(a))
        assert np.allclose(np.array(a1.log()), np.log(a))
        assert np.allclose(np.array(a1.sqrt()), np.sqrt(a))


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_random_pointwise_ops_tough(t: Case) -> None:
    for _ in range(10):
        a: np.ndarray = np.random.random(t.a_shape) * 100
        a1: skpp.Tensor = skpp.Tensor(a)
        assert np.allclose(np.array(a1.exp()), np.exp(a))
        assert np.allclose(np.array(a1.neg()), np.negative(a))
        assert np.allclose(np.array(a1.log()), np.log(a))
        assert np.allclose(np.array(a1.sqrt()), np.sqrt(a))
