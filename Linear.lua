local Linear, parent = torch.class('ct.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize, batchSize)
   parent.__init(self)

   self.weight = ct.empty(outputSize, inputSize)
   self.bias = ct.empty(1, outputSize)
   self.gradWeight = ct.empty(outputSize, inputSize)
   self.gradBias = ct.empty(1, outputSize)

   self.output = ct.empty(outputSize, batchSize)
   self.gradInput = ct.empty(inputSize, batchSize)

   self:reset()
end

function Linear:reset()
   stdv = 1 / math.sqrt(self.weight:size(2))
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function Linear:updateOutput(input)
   ct.dot(self.weight, input, self.output)
   ct.add_mat_vect(self.output, self.bias, 1)
   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   ct.dot(self.weight, gradOutput, self.gradInput, 1)
   return self.gradInput
end

function Linear:accGradParameters(input, gradOutput)
   ct.dot(gradOutput, input, self.gradWeight, 0, 1)
   ct.sum(gradOutput, self.gradBias, 1)
end