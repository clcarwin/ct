require 'torch'
require 'init'
require 'sys'

test = 'reduce'

if test == 'sgemm' then
   A = ct.rand(3, 100)
   B = ct.rand(100, 4)
   C = ct.rand(3, 4)

   ct.cublas_init()
   ct.sgemm(A, B, C)
   print(C)
   print(torch.mm(A:float(), B:float()))

elseif test == 'sigmoid' then
   A = ct.zeros(3, 4)
   
   ct.sigmoid(A)
   print(A)

elseif test == 'mult_by_sigmoid_grad' then
   A = ct.rand(3, 4)
   B = ct.rand(3, 4)

   ct.mult_by_sigmoid_grad(A, B)
   print(A)

elseif test == 'exp' then
   A = ct.ones(3, 4)
   
   ct.exp(A)
   print(A)

elseif test == 'add_mat_vect' then
   A = ct.ones(3, 4)
   b = ct.rand(4, 1)

   print(A)
   print(b)
   ct.add_mat_vect(A, b, 0)
   print(A)

   A = ct.ones(3, 4)
   b = ct.rand(3, 1)
   print(A)
   print(b)
   ct.sub_mat_vect(A, b, 1)
   print(A)

elseif test == 'add' then
   A = ct.ones(3, 4)
   B = ct.ones(3, 4)
   C = ct.empty(3, 4)

   ct.add(A, B, C)
   print(C)

elseif test == 'reduce' then
   A = ct.rand(3, 4)
   b = ct.empty(4, 1)
   c = ct.empty(3, 1)

   ct.max(A, b, 0)
   ct.max(A, c, 1)
   
   print(A)
   print(b)
   print(c)
end
