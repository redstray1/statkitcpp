import numpy as np
import statkitcpp as skpp

for _ in range(10):
    a = np.random.random((5, 1))
    b = np.random.random((1, 5))
    a1 = skpp.Tensor(a)
    b1 = skpp.Tensor(b)
    assert np.allclose(np.array(a1.exp()), np.exp(a))
    assert np.allclose(np.array(a1 + b1), a + b)
