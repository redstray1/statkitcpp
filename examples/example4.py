import statkitcpp as skpp
import numpy as np
import torch

a_np = np.random.random((1))
b_np = np.random.random((1))

# a_torch = torch.tensor(a_np, requires_grad=True)
# b_torch = torch.tensor(b_np, requires_grad=True)

a_skpp = skpp.Tensor(a_np)
b_skpp = skpp.Tensor(b_np)

a_skpp = skpp.full(shape=[1], value=float(a_np[0]))
#b_skpp = skpp.full(shape=[1], value=float(b_np[0]))

print(a_skpp, a_np, b_skpp, b_np)
#assert np.allclose(np.array(a_skpp), 
print(skpp.Tensor.div(a_skpp, b_skpp), skpp.Tensor.div(b_skpp, a_skpp))
print(a_skpp.div(b_skpp), b_skpp.div(a_skpp))

a_skpp.requires_grad = True
b_skpp.requires_grad = True

c_skpp = a_skpp / b_skpp
print(c_skpp, a_np / b_np)
assert np.allclose(np.array(c_skpp), a_np / b_np)
c_skpp.backward()
# c_torch = a_torch / b_torch
# c_torch.backward(torch.ones_like(c_torch))

print(a_np, b_np)
# print(a_torch.grad)
print(a_skpp.grad)
# print(b_torch.grad, b_skpp.grad, sep='\n')

# assert np.allclose(a_torch.grad.numpy(), np.array(a_skpp.grad))
# assert np.allclose(b_torch.grad.numpy(), np.array(b_skpp.grad))

