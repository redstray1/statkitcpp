import statkitcpp as skpp

a = skpp.arange(0, 125, dtype=skpp.float64).reshape([5, 5, 5])

print(a)

a[2,3] = a[2,3] + 2.0

print(a)