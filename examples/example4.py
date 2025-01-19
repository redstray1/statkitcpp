import statkitcpp as skpp
import numpy as np

a = skpp.arange(start=skpp.Scalar(0), end=skpp.Scalar(5))
#a = skpp.ones(shape=[5])
b = skpp.full(shape=[1, 5], value=skpp.Scalar(0.33))
print(a, b)
print(np.array(a.div(b)) == np.array(a) / np.array(b))