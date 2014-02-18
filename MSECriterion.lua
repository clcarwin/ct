require 'cutorch'

local MSECriterion, parent = torch.class('ct.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.gradInput = nil
   self.d = nil
end

function MSECriterion:updateOutput(input, target)
   self.d = self.d or ct.emptyAs(input)
   ct.sub(input, target, self.d)
   self.output = self.d:dot(self.d)
   return self.output
end

function MSECriterion:updateGradInput(input, target)
   self.gradInput = self.gradInput or ct.emptyAs(input)
   ct.smul(self.d, self.gradInput, 2)
   return self.gradInput
end
