require 'torch'
require 'cutorch'

require 'torch_datasets'
require 'ct'

X, y, X_te, y_te = torch_datasets.mnist()

num_train = X:size(1)
batch_size = 100
num_batches = num_train / batch_size
learning_rate = 0.1
num_hidden = 1000
epochs = 5

-- initialize cublas
ct.cublas_init()

-- put trainig data on GPU (data has to be transposed for cuBLAS)
dev_train = X:t():cuda():t()
dev_test = X_te:t():cuda():t()
dev_lbl = y:resize(num_train, 1):float():cuda()

