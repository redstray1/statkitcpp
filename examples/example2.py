import statkitcpp as skpp

a = skpp.ones(shape=[2, 2, 2], dtype='float32')
print(a)
a.reshape([4, 2])
print(a)
a.reshape([1, 4, 2, 1])
print(a)
