"""
Simplistic implementation of the two-layer neural network.
Training method is stochastic minibatch gradient descent with momentum.

As an example it computes XOR for given input.

Some details:
- tanh activation for hidden layer
- sigmoid activation for output layer
- cross-entropy loss

Less than 100 lines of active code.

"""

import numpy as np
import time

n_hidden = 10
n_in = 10
n_out = 10
n_samples = 300

batch_size = 30

learning_rate = 0.01
momentum = 0.9

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return  1 - np.tanh(x)**2

def train(x, t, V, W, bv, bw):

    # forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(Ew, W.T)

    dW = np.dot(Z.T, Ew)
    dV = np.dot(x.T, Ev)

    # first sum over the sample error
    # then average over the batch sample errors
    loss = np.mean (
        -np.sum( t * np.log(Y) + (1 - t) * np.log(1 - Y), axis=1) )

    # note that biases gradients are summations sample errors over the batch

    return  loss, (dV, dW, Ev.sum(axis=0), Ew.sum(axis=0))

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Setup initial parameters
# Note that initialization is cruxial for the first-order methods!

V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

# Generate some data

X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1

n_batches = X.shape[0]/batch_size

# Train
for epoch in range(900):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(n_batches):
        s = slice(batch_size * i, batch_size * (i+1))
        loss, grad = train(X[s], T[s], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print "Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 )

# Try to predict something

x = np.random.binomial(1, 0.5, n_in)
print "XOR prediction:"
print x
print predict(x, *params)

