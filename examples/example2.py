import statkitcpp as skpp

a = skpp.ones(shape=[2, 3, 2, 2, 4], dtype='float32')
print(a)
print(a.strides)
print()
