import statkitcpp as skpp
import numpy as np

a = skpp.arange(start=0, end=5)
b = skpp.arange(start=0, end=5).reshape([5, 1])
c = a + b
d = c.exp()
e = d * b
print(a, b, c, d, e, sep='\n')

print((a + b) * 5 + 2)

print((a + b) ** 2 + 3)

