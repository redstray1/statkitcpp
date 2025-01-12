import numpy as np
import statkitcpp as skpp

a = skpp.ones(shape=[2, 4], dtype='float64')
b = np.array(a)
c = np.array(a)
print(skpp.Tensor(b + 2 * c))