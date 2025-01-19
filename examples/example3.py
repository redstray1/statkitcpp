import numpy as np
import statkitcpp as skpp

aa = np.eye(5, dtype=np.int8)

a = skpp.Tensor(aa)
print(a.dtype, aa.dtype)
print(a.strides, a.shape, a.size)
print(a.sum(1), aa.sum(axis=1))
print(np.array(a.sum()) == np.sum(aa, axis=-1))
