from scipy.integrate import solve_ivp
from matplotlib.pyplot import *
from numpy import *



#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
a = 4.
b = 1.
c = 0.5
N = 400


def ode(t, SIR):
    S, I, R = SIR
    dSdt = c*R - (a*S*I/N)
    dIdt = (a*S*I/N) - b*I
    dRdt = b*I - c*R
    return [dSdt, dIdt, dRdt]

C0 = [300, 100, 0]

sol = solve_ivp(ode, (0, 10), C0,t_eval=linspace(0,10,100))

import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

plt.plot(sol.t, sol.y.T)
plt.legend(['S', 'I', 'R'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(labelsize=20, direction='in',top=True,right=True)
plt.xlabel('Time',fontsize=20)
plt.ylabel('Population',fontsize=20)
plt.tight_layout()
#plt.savefig(fname='fig/solve_ivp_solution.png' ,bbox_inches="tight")
#plt.show()

#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

















#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
import autograd.numpy as np
from autograd import grad, elementwise_grad, jacobian
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))

def C(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)
    return outputs

# initial guess for the weights and biases
params = init_random_params(0.1, layer_sizes=[1, 8, 3])

def objective_soln(params, step):
    return np.sum((sol.y.T - C(params, sol.t.reshape([-1, 1])))**2)

params = adam(grad(objective_soln), params,
              step_size=0.1, num_iters=500)

plt.plot(sol.t.reshape([-1, 1]), C(params, sol.t.reshape([-1, 1])),
         sol.t, sol.y.T, 'o')
plt.legend(['Ann', 'Bnn', 'Cnn','A', 'B', 'C'])
plt.xlabel('Time')
plt.ylabel('C')
#plt.show()
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa














#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


jacC = jacobian(C, 1)
jacC(params, sol.t.reshape([-1, 1])).shape

i = np.arange(len(sol.t))
plt.plot(jacC(params, sol.t.reshape([-1, 1]))[i, 0, i, 0],   c*sol.y[2] - (a*sol.y[0]*sol.y[1]/N), 'ro')
plt.plot(jacC(params, sol.t.reshape([-1, 1]))[i, 1, i, 0],   (a*sol.y[0]*sol.y[1]/N) - b*sol.y[1], 'bo')
plt.plot(jacC(params, sol.t.reshape([-1, 1]))[i, 2, i, 0],   b*sol.y[1] - c*sol.y[2], 'go')
plt.show()




# Derivatives
jac = jacobian(C, 1)

def dCdt(params, t):
    i = np.arange(len(t))
    return jac(params, t)[i, :, i].reshape((len(t), 3))

#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa























#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
plt.clf()
t = np.linspace(0, 10, 25).reshape((-1, 1))
params = init_random_params(0.01, layer_sizes=[1, 1000, 3])
i = 0    # number of training steps
N = 701  # epochs for training
et = 0.0 # total elapsed time

def objective(params, step):
    S, I, R = C(params, t).T
    dSdt, dIdt, dRdt = dCdt(params, t).T

    # dSdt = c*R - (a*S*I/N)
    # dIdt = (a*S*I/N) - b*I
    # dRdt = b*I - c*R

    z1 = np.sum((dSdt - (c*R - (a*S*I/N))  )**2)
    z2 = np.sum((dIdt - ((a*S*I/N) - b*I)  )**2)
    z3 = np.sum((dRdt - (b*I - c*R)        )**2)
    ic = np.sum((np.array([S[0], I[0], R[0]]) - C0)**2)  # initial conditions
    print(z1 + z2 + z3 + ic)
    return z1 + z2 + z3 + ic

def callback(params, step, g):
    if step % 100 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))

objective(params, 0)  # make sure the objective is scalar

import time
t0 = time.time()

params = adam(grad(objective), params,
              step_size=0.01, num_iters=N, callback=callback)

i += N
t1 = (time.time() - t0) / 60
et += t1

fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

plt.plot(t, C(params, t), sol.t, sol.y.T, 'o')
plt.legend(['Ann', 'Bnn', 'Cnn', 'A', 'B', 'C'])
plt.xlabel('Time',fontsize=20)
plt.ylabel('C',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(labelsize=20, direction='in',top=True,right=True)

print(f'{t1:1.1f} minutes elapsed this time. Total time = {et:1.2f} min. Total epochs = {i}.')
plt.plot(t, np.sum(dCdt(params, t), axis=1))
plt.xlabel('Time')
plt.ylabel(r'$\Sigma dC/dt$')
plt.tight_layout()
#plt.savefig(fname='fig/NeuralNetwork.png' ,bbox_inches="tight")

plt.show()



ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

plt.plot(t, np.sum(dCdt(params, t), axis=1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(labelsize=20, direction='in',top=True,right=True)

plt.xlabel('Time',fontsize=20)
plt.ylabel(r'$\Sigma dC/dt$',fontsize=20)
plt.tight_layout()
#plt.savefig(fname='fig/Error.png' ,bbox_inches="tight")
plt.show()




#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa