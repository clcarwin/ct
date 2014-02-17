function ct.rand(n, m)
   return ct.empty(n, m):uniform()
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
