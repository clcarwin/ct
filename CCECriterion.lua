-- Categorical crossentropy criterion
local CCECriterion, parent = torch.class('ct.CCECriterion', 'nn.Criterion')

function CCECriterion:__init()
   parent.__init(self)
   self.softmax_out = nil
   self.tmp_softmax = nil
   self.tmp_cce = nil
   self.gradInput = nil
end

function CCECriterion:updateOutput(input, target)
   self.softmax_out = self.softmax_out or ct.emptyAs(input)
   self.tmp_softmax = self.tmp_softmax or ct.empty(1, input:size(2))
   self.tmp_cce = self.tmp_cce or ct.emptyAs(input)

   ct.softmax(input, self.softmax_out, self.tmp_softmax)
   self.output = ct.cce(self.softmax_out, target, self.tmp_cce)

   return self.output
end

function CCECriterion:updateGradInput(input, target)
   self.gradInput = self.gradInput or ct.emptyAs(input)
   ct.sub(self.softmax_out, target, self.gradInput)
   return self.gradInput
end
