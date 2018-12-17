from autograd import grad, elementwise_grad, jacobian
from autograd.misc.optimizers import adam
from scipy.integrate import solve_ivp
import autograd.numpy.random as npr
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import autograd.numpy as np
from numpy import *

def MSE_error(y_computed,y_exact):
    """
    MSE, simple calculates the MSE for the inputs, then returns MSE
    """
    MSE = 0
    y_exact = y_exact.ravel()
    y_computed = y_computed.ravel()
    for y_computed_i,y_exact_i in zip(y_computed,y_exact):
        MSE += (y_computed_i-y_exact_i)**2
    return MSE/len(y_exact)

def R2_error(y_computed,y_exact):
    """
    R2, simple calculates the R2 for the inputs, then returns R2
    """
    #ravel to two long lists
    y_exact = y_exact.ravel()
    y_computed = y_computed.ravel()

    #define sums and mean-value
    numerator = 0
    denominator = 0
    y_mean = np.mean(y_exact)

    #calculate the sums
    for y_computed_i,y_exact_i in zip(y_computed,y_exact):
        numerator += (y_computed_i-y_exact_i)**2
        denominator += (y_exact_i-y_mean)**2
    return 1 - (numerator/denominator)

#initial values
a = 4.; b = 1.; c = 0.5

N = 400             #people
S0 = 300            #susceptible 
I0 = 100            #infected
R0 = 0              #recovered


def ode(t, SIR):
    """ 
    coupled ordinary differential equations
    """
    S, I, R = SIR
    dSdt = c*R - (a*S*I/N)
    dIdt = (a*S*I/N) - b*I
    dRdt = b*I - c*R
    return [dSdt, dIdt, dRdt]

SIR0 = [S0, I0, R0]

sol = solve_ivp(ode, (0, 10), SIR0,t_eval=linspace(0,10,100))






def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "activation function"
    return x / (1.0 + np.exp(-x))


def C(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = activation_function(outputs)
    return outputs

# initial guess for the weights and biases



def objective_soln(params, step):
    return np.sum((sol.y.T - C(params, sol.t.reshape([-1, 1])))**2)


def simulation(func,n,e,s):
    activation_function = func
    nodes = n
    epoch = e
    step = s

    params = init_random_params(0.1, layer_sizes=[1, nodes, 3])
    params = adam(grad(objective_soln), params,
                  step_size=step, num_iters=epoch)

    neural_solution = C(params, sol.t.reshape([-1, 1]))

    ax = subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    plot(sol.t.reshape([-1, 1]), neural_solution)
    plot(sol.t, sol.y.T,"o")
    xlabel("Time [day]",fontsize=20)
    ylabel("People [#]",fontsize=20)
    xticks(fontsize=20)
    yticks(fontsize=20)
    tick_params(labelsize=20, direction='in',top=True,right=True,left=True,bottom=True,length=5)
    tick_params(labelsize=20, direction='in',left=True,which="minor",length=3)
    tight_layout()
    legend(['Snn', 'Inn', 'Rnn'],loc="best",fontsize=12)
    savefig("%s_%s_%s_%s.pdf" % (activation_function.__name__,nodes,epoch,str(step).replace(".","")) ,bbox_inches="tight")
    clf()

    return MSE_error(sol.y,neural_solution),R2_error(sol.y,neural_solution)








if __name__ == '__main__':
    func_ = [swish,np.tanh,np.arctan]
    n_ = [100,1000]
    e_ = [100,1000,10000]
    s_ = [0.1,0.01,0.001]

    for func in func_:
        for n in n_:
            for e in e_:
                for s in s_:
                    activation_function = func
                    mse,r2 = simulation(func,n,e,s)
                    print("%s %s %s %s %.3f %.7f" % (activation_function.__name__,n,e,str(s).replace(".",""),mse,r2))















