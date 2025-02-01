import numpy as np
import statkitcpp as skpp

a_np = np.random.random()

a = skpp.arange(start=0, end=5)
b = skpp.arange(start=0, end=5).reshape([5, 1])

a.requires_grad = True
b.requires_grad = True

c = (a + b).max()

c.backward()

print(a.grad)
