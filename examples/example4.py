import statkitcpp as skpp
import numpy as np

a = skpp.arange(start=skpp.Scalar(0), end=skpp.Scalar(5))
#a = skpp.ones(shape=[5])
b = skpp.arange(start=skpp.Scalar(0), end=skpp.Scalar(5)).reshape([5, 1])
c = a.add(b)
d = c.exp()
e = d.mul(b)
print(a, b, c, d, e, sep='\n')