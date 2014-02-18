-- B = softmax(A), x is temporary storage
function ct.softmax(A, B, x)
   ct.max(A, x, 0)
   ct.sub_mat_vect(A, x, 0)
   ct.exp(A, B)
   ct.sum(B, x, 0)
   ct.div_mat_vect(B, x, 0)
end

function ct.sub(x, y, z)
   ct.add(x, y, z, -1)
end

function ct.rand(n, m)
   return ct.empty(n, m):uniform()
end

function ct.randn(n, m)
   return ct.empty(n, m):normal()
end

function ct.zeros(n, m)
   return ct.empty(n, m):zero()
end

function ct.ones(n, m)
   return ct.empty(n, m):fill(1)
end

function ct.empty(n, m)
   return torch.Tensor():cuda():resize(m, n):t()
end

function ct.emptyAs(A)
   return ct.empty(A:size(1), A:size(2))
end
