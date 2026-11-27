from math import exp, log, tanh


from typing import Sequence
import numpy as np

class Initializer:

  def init_weights(self, n_in, n_out):
    raise NotImplementedError

  def init_bias(self, n_out):
    raise NotImplementedError

class Var:
    """
    A variable which holds a float and enables gradient computations.
    """

    def __init__(self, val: float, grad_fn=lambda: []):
        assert type(val) == float
        self.v = val
        self.grad_fn = grad_fn
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for input, grad in self.grad_fn():
            input.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v + other.v, lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Var') -> 'Var':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return self + (-other)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return self * other ** -1

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)

    def sigmoid(self):
        return Var(1.0 / (1.0 + exp(-self.v)), lambda: [(self, self.v * (1.0 - self.v))])

    def tanh(self):
        t = tanh(self.v)
        return Var(t, lambda: [(self, 1.0 - t ** 2)])
    
    def identity(self):
        return Var(self.v, lambda: [(self, 1.0)])

    def relu(self):
        return Var(self.v if self.v > 0.0 else 0.0, lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])
    
    def exp(self):
        return Var(exp(self.v), lambda: [(self, exp(self.v))])
    
    def log(self):
        return Var(log(self.v), lambda: [(self, self.v ** -1)])

def nparray_to_Var(x):
  if x.ndim==1:
    y = [[Var(float(x[i]))] for i in range(x.shape[0])] # always work with list of list
  else:
    y = [[Var(float(x[i,j])) for j in range(x.shape[1])] for i in range(x.shape[0])]
  return y

def Var_to_nparray(x):
  y = np.zeros((len(x),len(x[0])))
  for i in range(len(x)):
    for j in range(len(x[0])):
      y[i,j] = x[i][j].v
  return y

import random

def squared_loss(t, y):

  # add check that sizes agree

  def squared_loss_single(t, y):
    Loss = Var(0.0)
    for i in range(len(t)): # sum over outputs
      Loss += (t[i]-y[i]) ** 2
    return Loss

  Loss = Var(0.0)
  for n in range(len(t)): # sum over training data
    Loss += squared_loss_single(t[n],y[n])
  return Loss

class NormalInitializer(Initializer):

  def __init__(self, mean=0, std=0.1):
    self.mean = mean
    self.std = std

  def init_weights(self, n_in, n_out):
    return [[Var(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)]

  def init_bias(self, n_out):
    return [Var(0.0) for _ in range(n_out)]

class ConstantInitializer(Initializer):

  def __init__(self, weight=1.0, bias=0.0):
    self.weight = weight
    self.bias = bias

  def init_weights(self, n_in, n_out):
    return [[Var(self.weight) for _ in range(n_out)] for _ in range(n_in)]

  def init_bias(self, n_out):
    return [Var(self.bias) for _ in range(n_out)]


class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer = NormalInitializer()):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn

    def __repr__(self):
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Var]:
      params = []
      for r in self.weights:
        params += r

      return params + self.bias

    def forward(self, single_input: Sequence[Var]) -> Sequence[Var]:
        # self.weights is a matrix with dimension n_in x n_out. We check that the dimensionality of the input
        # to the current layer matches the number of nodes in the current layer
        assert len(self.weights) == len(single_input), "weights and single_input must match in first dimension"
        weights = self.weights
        out = []
        # For some given data point single_input, we now want to calculate the resulting value in each node in the current layer
        # We therefore loop over the (number of) nodes in the current layer:
        for j in range(len(weights[0])):
            # Initialize the node value depending on its corresponding parameters.
            node = self.bias[j]
            # We now finish the linear transformation corresponding to the parameters of the currently considered node.
            for i in range(len(single_input)):
                node += weights[i][j] * single_input[i]
            node = self.act_fn(node)
            out.append(node)

        return out

def data_generator(noise=0.1, n_samples=300, D1=True):
    # Create covariates and response variable
    if D1:
        X = np.linspace(-3, 3, num=n_samples).reshape(-1,1) # 1-D
        np.random.shuffle(X)
        y = np.random.normal((0.5*np.sin(X[:,0]*3) + X[:,0]), noise) # 1-D with trend
    else:
        X = np.random.multivariate_normal(np.zeros(3), noise*np.eye(3), size = n_samples) # 3-D
        np.random.shuffle(X)
        y = np.sin(X[:,0]) - 5*(X[:,1]**2) + 0.5*X[:,2] # 3-D

    # Stack them together vertically to split data set
    data_set = np.vstack((X.T,y)).T

    train, validation, test = np.split(data_set, [int(0.35*n_samples), int(0.7*n_samples)], axis=0)

    # Standardization of the data, remember we do the standardization with the training set mean and standard deviation
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)

    train = (train-train_mu)/train_sigma
    validation = (validation-train_mu)/train_sigma
    test = (test-train_mu)/train_sigma

    x_train, x_validation, x_test = train[:,:-1], validation[:,:-1], test[:,:-1]
    y_train, y_validation, y_test = train[:,-1], validation[:,-1], test[:,-1]

    return x_train, y_train,  x_validation, y_validation, x_test, y_test


def forward(input, network):
    
  def forward_single(x, network):
    for layer in network:
        x = layer.forward(x)
    return x

  output = [ forward_single(input[n], network) for n in range(len(input))]
  return output

def parameters(network):
  params = []
  for layer in range(len(network)):
    params += network[layer].parameters()
  return params

def update_parameters(params, learning_rate=0.01):
  for p in params:
    p.v -= learning_rate*p.grad

def zero_gradients(params):
  for p in params:
    p.grad = 0.0

# Smaller dataset (21 train, 21 val, 18 test) so it runs fast and actually overfits
np.random.seed(42)
x_tr, y_tr, x_va, y_va, x_te, y_te = data_generator(noise=0.5, n_samples=60, D1=True)
x_tr, y_tr = nparray_to_Var(x_tr), nparray_to_Var(y_tr)
x_va, y_va = nparray_to_Var(x_va), nparray_to_Var(y_va)
x_te, y_te = nparray_to_Var(x_te), nparray_to_Var(y_te)

def model(x_train, y_train, x_validation, y_validation, x_test, y_test, NN, EPOCHS=200, LEARN_R=2e-3):
    train_loss, val_loss = [], []

    for e in range(EPOCHS):
        Loss = squared_loss(y_train, forward(x_train, NN))
        Loss.backward()
        update_parameters(parameters(NN), LEARN_R)
        zero_gradients(parameters(NN))

        train_loss.append(Loss.v / len(y_train))
        val_loss.append(squared_loss(y_validation, forward(x_validation, NN)).v / len(y_validation))

    test_loss = squared_loss(y_test, forward(x_test, NN)).v / len(y_test)
    print(f"Final Train MSE: {train_loss[-1]:.4f} | Val MSE: {val_loss[-1]:.4f} | Test MSE: {test_loss:.4f}")
    return train_loss, val_loss, test_loss

losses = []

# 1. Underfit: Linear model, stopped early (high train, val, and test MSE)
NN_under = [DenseLayer(1, 1, lambda x: x.identity())]
print("Underfit:")
losses.append(model(x_tr, y_tr, x_va, y_va, x_te, y_te, NN_under, EPOCHS=10, LEARN_R=2e-3))

# 2. Just right: Moderate capacity (lowest val & test MSE)
NN_right = [
    DenseLayer(1, 6, lambda x: x.tanh(), initializer=NormalInitializer(0, 0.5)),
    DenseLayer(6, 1, lambda x: x.identity(), initializer=NormalInitializer(0, 0.5))
]
print("Just right:")
losses.append(model(x_tr, y_tr, x_va, y_va, x_te, y_te, NN_right, EPOCHS=150, LEARN_R=3e-3))

# 3. Overfit: High capacity + longer training (low train MSE, higher val & test MSE)
NN_over = [
    DenseLayer(1, 30, lambda x: x.tanh(), initializer=NormalInitializer(0, 1.0)),
    DenseLayer(30, 1, lambda x: x.identity(), initializer=NormalInitializer(0, 0.3))
]
print("Overfit:")
losses.append(model(x_tr, y_tr, x_va, y_va, x_te, y_te, NN_over, EPOCHS=250, LEARN_R=4e-3))
names = ["Underfit (1-1)", "Just right (1-6-1)", "Overfit (1-30-1)"]
nets  = [NN_under, NN_right, NN_over]

import matplotlib.pyplot as plt

# --- Figure 1: Loss curves ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name, (tr, va, te) in zip(axes, names, losses):
    ax.plot(tr, label="Train MSE", lw=2)
    ax.plot(va, label="Validation MSE", lw=2, linestyle="--")
    ax.axhline(te, color="k", linestyle=":", label=f"Test MSE ({te:.3f})")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curves.png", bbox_inches="tight")
plt.show()

# --- Figure 2: Model fits ---
x_tr_np, y_tr_np = Var_to_nparray(x_tr), Var_to_nparray(y_tr)
x_te_np, y_te_np = Var_to_nparray(x_te), Var_to_nparray(y_te)
x_grid = np.linspace(np.min(x_tr_np), np.max(x_tr_np), 150).reshape(-1, 1)
x_grid_v = nparray_to_Var(x_grid)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, name, net in zip(axes, names, nets):
    y_pred = Var_to_nparray(forward(x_grid_v, net))
    ax.scatter(x_tr_np, y_tr_np, alpha=0.7, s=30, label="Train")
    ax.scatter(x_te_np, y_te_np, alpha=0.7, s=30, label="Test")
    ax.plot(x_grid, y_pred, color="red", lw=2, label="Model fit")
    ax.set_title(name)
    ax.set_xlabel("$x$")
    if ax == axes[0]:
        ax.set_ylabel("$y$")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fit_comparison.png", bbox_inches="tight")
plt.show()
