import statkitcpp as skpp

a = skpp.Tensor(shape=[2, 2], dtype=skpp.DataType.Float32)
print(a.requires_grad)
a.requires_grad = False
print(a.requires_grad)
print(a.shape)
print(a.size)
print(a.dtype)
print(skpp.full(shape=[2, 2], value=2))
b = a
print(b)
print(a.shape, b.shape)
print(a.broadcastable_to(b))
print(b.requires_grad)
b.requires_grad = True
print(b.requires_grad)