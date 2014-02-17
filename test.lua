require 'ct'
require 'Test'

test = {}
function test.Linear()
   m = ct.Linear(5, 6)
   i = ct.randn(5, 2)

   assert(testJacobian(m, i) < 1e-2)
   assert(testJacobianParameters(m, i) < 1e-3)
end

function test.Tanh()
   m = ct.Tanh()
   i = ct.randn(5, 2)

   assert(testJacobian(m, i) < 1e-3)
end

for k, v in pairs(test) do
   print('Testing ' .. k)
   v()
end
