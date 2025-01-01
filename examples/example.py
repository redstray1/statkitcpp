import statkitcpp as m

a = m.Tensor(shape=[2, 2])
print(a.requires_grad)
a.requires_grad = False
print(a.requires_grad)
print(a.shape)
print(a.size)