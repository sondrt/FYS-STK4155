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






a = 4.
b = 1.
c = 0.5
N = 400
S0 = 300
I0 = 100
R0 = 0

def ode(t, SIR):
    S, I, R = SIR
    dSdt = c*R - (a*S*I/N)
    dIdt = (a*S*I/N) - b*I
    dRdt = b*I - c*R
    return [dSdt, dIdt, dRdt]

C0 = [S0, I0, R0]

sol = solve_ivp(ode, (0, 10), C0,t_eval=linspace(0,10,100))



def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))

activation_function = swish

def C(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = activation_function(outputs)
    return outputs


# initial guess for the weights and biases
params = init_random_params(0.1, layer_sizes=[1, 100, 3])

def objective_soln(params, step):
    return np.sum((sol.y.T - C(params, sol.t.reshape([-1, 1])))**2)

params = adam(grad(objective_soln), params,
              step_size=0.01, num_iters=500)



jacC = jacobian(C, 1)
jacC(params, sol.t.reshape([-1, 1])).shape

i = np.arange(len(sol.t))

# Derivatives
jac = jacobian(C, 1)

def dCdt(params, t):
    i = np.arange(len(t))
    return jac(params, t)[i, :, i].reshape((len(t), 3))












outfile = open("unsupervised.txt","w")

func_ = [swish,np.tanh,np.arctan]
N = 101  # epochs for training
n_ = [100,1000]
e_ = [100,1000,10000]
s_ = [0.1,0.01,0.001]

for func in func_:
    for n in n_:
        for N in e_:
            for s in s_:
                activation_function = func
                plt.clf()
                t = np.linspace(0, 10, 25).reshape((-1, 1))
                params = init_random_params(0.1, layer_sizes=[1, n, 3])
                i = 0    # number of training steps
                et = 0.0 # total elapsed time

                def objective(params, step):
                    S, I, R = C(params, t).T
                    dCadt, dCbdt, dCcdt = dCdt(params, t).T
                    z1 = np.sum((dCadt - (c*R - (a*S*I/N))  )**2)
                    z2 = np.sum((dCbdt - ((a*S*I/N) - b*I)  )**2)
                    z3 = np.sum((dCcdt - (b*I - c*R)        )**2)
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
                              step_size=s, num_iters=N, callback=callback)

                i += N
                t1 = (time.time() - t0) / 60
                et += t1
                print(f'{t1:1.1f} minutes elapsed this time. Total time = {et:1.2f} min. Total epochs = {i}.')

                nn_solution = C(params, t)

                ax = subplot(111)
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(1.5)
                plot(t, C(params, t), sol.t, sol.y.T, 'o')
                xlabel("Time [day]",fontsize=20)
                ylabel("People [#]",fontsize=20)
                xticks(fontsize=20)
                yticks(fontsize=20)
                tick_params(labelsize=20, direction='in',top=True,right=True,left=True,bottom=True,length=5)
                tick_params(labelsize=20, direction='in',left=True,which="minor",length=3)
                tight_layout()
                legend(['Snn', 'Inn', 'Rnn'],loc="best",fontsize=12)
                savefig("unsupervised_%s_%s_%s_%s.pdf" % (activation_function.__name__,n,N,str(s).replace(".","")) ,bbox_inches="tight")
                clf()
                mse = MSE_error(sol.y,nn_solution)
                r2  = R2_error(sol.y,nn_solution)
                outfile.write("%s %s %s %s %.3f %.7f\n" % (activation_function.__name__,n,N,str(s).replace(".",""),mse,r2))
                outfile.write(f'{t1:1.1f} minutes elapsed this time. Total time = {et:1.2f} min. Total epochs = {i}.\n')
outfile.close()














