import statkitcpp as skpp
import numpy as np

a = skpp.arange(start=0, end=5)
b = skpp.arange(start=0, end=5).reshape([5, 1])
b.requires_grad = True

a.requires_grad = True
d = a + b
# d = skpp.ones(shape=[5,5])
# d.requires_grad = True
c = d.exp()
c = c.mean()
c.backward(retain_graph=True)
print(c.grad, d.grad, sep='\n')
print(c)

