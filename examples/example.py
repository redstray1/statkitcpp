import statkitcpp as skpp
import sys

a = skpp.Tensor(shape=[2, 2], dtype=skpp.ScalarType.float32)
print(a.requires_grad, a.__class__)
a.requires_grad = False
print(a.requires_grad)
print(a.shape)
print(a.size)
print(a.dtype)
print(skpp.full(shape=[2, 2], value=skpp.Scalar(2)))
b = a
print(b)
print(a.shape, b.shape)
print(a.broadcastable_to(b))
print(b.requires_grad)
b.requires_grad = True
print(b.requires_grad)
print(sys.getsizeof(b), b.nbytes, b.itemsize)
c = skpp.full(shape=[10, 10, 10], value=skpp.Scalar(2), dtype=skpp.ScalarType.float64)
print(sys.getsizeof(c), c.nbytes, c.itemsize, c)
print(a.broadcastable_to(c))