#RK4_SIRS_function.py
from numpy import *
from matplotlib.pyplot import *

# INITIALIZE
a = 4 		# infection rate
b = 3 		# recovery rate
c = 0.5		# immunity lose

N = 400

S0 = 300
I0 = 100
R0 = 0

T = 365*1
dt = 0.01

t = linspace(0,T,T)

S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)


#IC
S[0] = S0
R[0] = R0
I[0] = I0


#Define help funcitons
def dSdt( t, S, I, R):
	return(c * R - a * S * I / N )


def dIdt( t, S, I):
	return(a * S * I / N - b * I)

def dRdt( t, I, R): #here for the lolz
	return(b * I - c * R)

def RK4SIRS( S0, I0, R0, dt):
	"""Solves the SIRS diff eq. with 4th order runge-kutta. """
	N = S0 + I0 + R0

	for i in range(len(t)):
		Si = S[i]
		Ii = I[i]  
		Ri = R[i]

		SK1 = dSdt(i, Si, Ii, Ri)
		IK1 = dIdt(i, Si, Ii)
		RK1 = dRdt(i, Ii, Ri)

		SK2 = dSdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1, Ri + dt / 2 * RK1)
		IK2 = dIdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1)
		RK2 = dRdt( i + dt / 2, Ii + dt/ 2 * IK1, Ri + dt / 2 * RK1 )

		SK3 = dSdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2 , Ri + dt / 2 * RK2)
		IK3 = dIdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2)
		RK3 = dRdt( i + dt / 2, Ii + dt/ 2 * IK2, Ri + dt / 2 * RK2 )

		SK4 = dSdt( i + dt, Si + dt * SK3, Ii + dt * IK3, Ri + dt / 2 * RK2)
		IK4 = dIdt( i + dt, Si + dt * SK3, Ii + dt * IK3)
		RK4 = dRdt( i + dt, Ii + dt * IK3, Ri + dt / 2 * RK2)

		S[i + 1] = Si + dt / 6 * (SK1 + 2 * SK2 + 2 * SK3 + SK4)
		I[i + 1] = Ii + dt / 6 * (IK1 + 2 * IK2 + 2 * IK3 + IK4)
		R[i + 1] = Ri + dt / 6 * (RK1 + 2 * RK2 + 2 * RK3 + RK4)

#	BC
	R_new =  N - S - I 
	return(S, I, R_new)

#run!
S_new, I_new, R_new  = RK4SIRS( S0, I0, R0, dt)

#analytical equilibrium 
Ssteady = b / a 
Isteady =  (1 - b/a)/(1+b/c)
Rsteady = b/c * (1 - b/a)/(1+b/c)

#for easier plotting:
n = zeros(len(t))
V_Ss = linspace(Ssteady * N, Ssteady * N,len(t))
V_Is = linspace(Isteady * N, Isteady * N,len(t))
V_Rs = linspace(Rsteady * N, Rsteady * N,len(t))

# Plot !
fig = figure()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(1.5)


plot(t, S_new[:-1], "b", label = 'S')
plot(t, I_new[:-1], "r", label = 'I')
plot(t, R_new[:-1], "g", label = 'R')

#plotting the equilibrium lines.
plot(t,V_Ss,"--b")
plot(t,V_Is,"--r")
plot(t,V_Rs,"--g")

#plotting the total population N, to see if it changes( and for when it changes. )
#plot(t, (S_new+I_new+R_new)[:-1],"k")

xlabel("Time",fontsize=20)
ylabel("Population",fontsize=20)
xticks(fontsize=20)
yticks(fontsize=20)
tick_params(labelsize=20, direction='in',top=True,right=True)
tight_layout()
legend(loc="right",fontsize=15)
savefig(fname='fig/texfig/RK4_b3T1.png' ,bbox_inches="tight")
show()








