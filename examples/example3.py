import numpy as np
import statkitcpp as skpp

aa = np.random.random((2, 3, 2))

a = skpp.Tensor(aa)
print(a.strides, a.shape)
print(a.sum(2))
print(np.array(a.sum()) == np.sum(aa, axis=-1))
