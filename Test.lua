eps = 1e-3

function testJacobian(module, input, x, dx)
   module:forward(input)

   x = x or input

   local sx = torch.CudaTensor(x:storage())
   local soutput = torch.CudaTensor(module.output:storage())
   local jacobian = torch.Tensor(sx:nElement(), soutput:nElement())
   local jacobian_hat = torch.Tensor(sx:nElement(), soutput:nElement())

   -- Build Jacobian from module's updateGradInput
   soutput:zero()
   for i = 1,soutput:nElement() do
      soutput[i] = 1
      module:updateGradInput(input, module.output)
      if dx then
         dx:zero()
         module:accGradParameters(input, module.output)
         jacobian:select(2, i):copy(dx)
      else
         jacobian:select(2, i):copy(module.gradInput:t())
      end
      soutput[i] = 0
   end

   -- Numerically estimate the Jacobian
   for i = 1,sx:nElement() do
      orig = sx[i]
      sx[i] = orig + eps
      module:forward(input)
      local f1 = module.output:clone()

      sx[i] = orig - eps
      module:forward(input)
      local f2 = module.output:clone()

      jacobian_hat:select(1, i):copy(f1:add(-1, f2):div(2 * eps):t())
      sx[i] = orig
   end

   return jacobian:add(-1, jacobian_hat):abs():max()
end

function testJacobianParameters(module, input)
   x, dx = module:getParameters()
   return testJacobian(module, input, x, dx)
end

function testCriterion(module, input, target)
   module:forward(input, target)
   module:backward(input, target)

   local sinput = torch.Tensor(input:storage())
   local grad_hat = torch.Tensor(sinput:nElement())
   for i = 1,sinput:nElement() do
      orig = sinput[i]
      sinput[i] = orig + eps
      module:forward(input, target)
      local f1 = module.output

      sinput[i] = orig - eps
      module:forward(input, target)
      local f2 = module.output

      grad_hat[i] = (f1 - f2) / (2 * eps)
      sinput[i] = orig
   end

   return module.gradInput:add(-1, grad_hat):abs():max()
end
