#RK4_SIRS_vitaldynamics_SeasonalVariation.py

#RK4_SIRS_function.py
from numpy import *
import matplotlib.pyplot as plt

# INITIALIZE
#a = 4 		# infection rate
b = 1 		# recovery rate
c = 0.5		# immunity lose

d  = 0.0002 	# death rate
d1 = 0.001	# death rate of infected
e  = 0.0006 	# birth rate

dt = 0.01
N = 400
S0 = 300
I0 = 100
R0 = 0
T = 365*5

t = linspace(0,T,T)

S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)


#IC
S[0] = S0
I[0] = I0
R[0] = R0


#Define help funcitons
def dSdt( t, S, I, R, a):
	return(c * R - a * S * I / N  - d * S + e * N)


def dIdt( t, S, I, a):
	return(a * S * I / N - b * I - d * I - d1 * I)

def dRdt( t, I, R): #here for the lolz
	return(b * I - c * R - d * R)

def seasonalVariation(t):
	a0 = 4
	A = 10
	o_s_n_f = 2*pi/365.
	a = A*cos(t*o_s_n_f) + a0
	return (a)

def RK4SIRS( N, S0, I0, R0, dt):
	"""Solves the SIRS diff eq. with runge-kutta of forth order """
	N = S0 + I0 + R0
	R = zeros(len(t)+1)
	for i in range(len(t)):
		a = seasonalVariation(i)

		Si = S[i]
		Ii = I[i]  
		Ri = R[i]

		SK1 = dSdt(i, Si, Ii, Ri, a)
		IK1 = dIdt(i, Si, Ii, a)
		RK1 = dRdt(i, Ii, Ri)

		SK2 = dSdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1, Ri + dt / 2 * RK1, a)
		IK2 = dIdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1, a)
		RK2 = dRdt( i + dt / 2, Ii + dt/ 2 * IK1, Ri + dt / 2 * RK1 )

		SK3 = dSdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2 , Ri + dt / 2 * RK2, a)
		IK3 = dIdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2, a)
		RK3 = dRdt( i + dt / 2, Ii + dt/ 2 * IK2, Ri + dt / 2 * RK2)

		SK4 = dSdt( i + dt, Si + dt * SK3, Ii + dt * IK3, Ri + dt / 2 * RK2, a)
		IK4 = dIdt( i + dt, Si + dt * SK3, Ii + dt * IK3, a)
		RK4 = dRdt( i + dt, Ii + dt * IK3, Ri + dt / 2 * RK2)

		S[i + 1] = Si + dt / 6 * (SK1 + 2 * SK2 + 2 * SK3 + SK4)
		I[i + 1] = Ii + dt / 6 * (IK1 + 2 * IK2 + 2 * IK3 + IK4)
		R[i + 1] = Ri + dt / 6 * (RK1 + 2 * RK2 + 2 * RK3 + RK4)

	return(S, I, R)

#run!
S_new, I_new, R_new  = RK4SIRS(N, S0, I0, R0, dt)




#plot!
plt.figure(facecolor='w')
plt.plot(t, S_new[:-1], label = 'Susceptible')
plt.plot(t, I_new[:-1], label = 'Infected')
plt.plot(t, R_new[:-1], label = 'Recovered')

plt.plot(t, (S_new+I_new+R_new)[:-1],label = 'Total')

plt.xlabel('Time/days')
plt.ylabel('number')
plt.title('SIRS with RK4, vital dynamics and seasonal variation')
plt.legend(loc="best")
plt.grid(True)
#plt.savefig(fname='fig/newfig/RK4VDSV_Stand_A10T5.pdf')
plt.show()







