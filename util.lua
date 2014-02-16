function ct.rand(n, m)
   return ct.empty(n, m):uniform()
end

function ct.empty(n, m)
   return torch.Tensor():cuda():resize(m, n):t()
end
