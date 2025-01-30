import statkitcpp as skpp
import numpy as np
import sys

a = np.random.random((5, 1))

a1 = skpp.Tensor(a)

a = np.random.random((5, 1))

a1 = skpp.Tensor(a)
print(a1.__class__, a1.size, a1.dtype, a1.ndim, a1.itemsize, a1.nbytes, a1.__repr__)

print(np.array(a1.exp()))