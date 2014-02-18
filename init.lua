require 'torch'
require 'nn'
require 'cutorch'

require 'libct'

include('Linear.lua')
include('Tanh.lua')
include('CCECriterion.lua')

include('util.lua')

ct.cublas_init()
