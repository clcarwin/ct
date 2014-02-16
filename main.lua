require 'init'
require 'sys'

A = ct.rand(2, 3)
B = ct.rand(3, 4)
C = ct.empty(2, 4)

print(torch.mm(A:float(), B:float()))

ct.cublas_init()
ct.sgemm(A, B, C)
print(C)
