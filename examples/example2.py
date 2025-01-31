import statkitcpp as skpp
import numpy as np
import sys

a = skpp.full(shape=[1,1], value=0.0762168)
b = skpp.full(shape=[1], value=0.06522323)

a.requires_grad = True
b.requires_grad = True

c = a / b

c.backward()

print(a.grad)
