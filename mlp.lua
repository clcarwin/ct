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

ct.cublas_init()

X = X:cuda():t()
X_te = X_te:cuda():t()
y = torch.eye(10):index(1, y):cuda():t()

i0, i1, i2 = 784, num_hidden, 10

w_bound = math.sqrt(6 / (i0 + i1))
W1 = ct.empty(i1, i0):uniform(-w_bound, w_bound)
b1 = ct.zeros(1, i1)

w_bound = math.sqrt(6 / (i1 + i2))
W2 = ct.empty(i2, i1):uniform(-w_bound, w_bound)
b2 = ct.zeros(1, i2)

-- temp storage
a2 = ct.empty(i1, batch_size)
a3 = ct.empty(i2, batch_size)
d3 = ct.empty(i2, batch_size)
d2 = ct.empty(i1, batch_size)
dW1 = ct.empty(i1, i0)
db1 = ct.empty(1, i1)
dW2 = ct.empty(i2, i1)
db2 = ct.empty(1, i2)
softmax_tmp = ct.empty(1, batch_size)

for i = 1,epochs do
   print(i)
   for batch = 1,num_batches do
      s = (batch - 1) * batch_size

      X_tr = X:narrow(2, s + 1, batch_size)
      y_tr = y:narrow(2, s + 1, batch_size)

      -- forward pass
      ct.dot(W1, X_tr, a2)
      ct.add_mat_vect(a2, b1, 1)
      ct.tanh(a2)

      ct.dot(W2, a2, a3)
      ct.add_mat_vect(a3, b2, 1)
      ct.softmax(a3, softmax_tmp)

      -- backward pass
      ct.sub(a3, y_tr, d3)
      ct.dot(W2, d3, d2, 1)
      ct.mult_by_tanh_grad(d2, a2)

      ct.dot(d3, a2, dW2, 0, 1)
      ct.dot(d2, X_tr, dW1, 0, 1)
      ct.sum(d3, db2, 1)
      ct.sum(d2, db1, 1)

      -- update params
      ct.add(W1, dW1, W1, -learning_rate / batch_size)
      ct.add(b1, db1, b1, -learning_rate / batch_size)
      ct.add(W2, dW2, W2, -learning_rate / batch_size)
      ct.add(b2, db2, b2, -learning_rate / batch_size)
   end
end

a2t = ct.empty(i1, X_te:size(2))
a3t = ct.empty(i2, X_te:size(2))
softmax_tmpt = ct.empty(1, X_te:size(2))

ct.dot(W1, X_te, a2t)
ct.add_mat_vect(a2t, b1, 1)

ct.dot(W1, X_te, a2t)
ct.add_mat_vect(a2t, b1, 1)
ct.tanh(a2t)

ct.dot(W2, a2t, a3t)
ct.add_mat_vect(a3t, b2, 1)
ct.softmax(a3t, softmax_tmpt)

_, p = a3t:float():max(1)
print(p:ne(y_te):float():mean())
