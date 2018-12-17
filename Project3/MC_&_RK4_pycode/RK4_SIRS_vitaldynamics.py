#RK4_SIRS_vitaldynamics.py

#RK4_SIRS_function.py
from numpy import *
import matplotlib.pyplot as plt

# INITIALIZE
a = 4 		# infection rate
b = 1 		# recovery rate
c = 0.5		# immunity lose

d = 0.002 	# death rate
d1 = 0.01	# death rate of infected
e = 0.006 	# birth rate

dt = 0.01
N = 400
S0 = 300
I0 = 100
R0 = 0
T = 365*10

t = linspace(0,T,T)

S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)


#IC
S[0] = S0
I[0] = I0
R[0] = R0


#Define help funcitons
def dSdt( t, S, I, R):
	return(c * R - a * S * I / N  - d * S + e * N)


def dIdt( t, S, I):
	return(a * S * I / N - b * I - d * I - d1 * I)

def dRdt( t, I, R): #here for the lolz
	return(b * I - c * R - d * R)

def RK4SIRS( N, S0, I0, R0, dt):
	"""Solves the SIRS diff eq. with runge-kutta of forth order """
	N = S0 + I0 + R0
	R = zeros(len(t)+1)

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

	return(S, I, R)

#run!
S_new, I_new, R_new  = RK4SIRS(N, S0, I0, R0, dt)



#plot!
plt.figure(facecolor='w')
plt.plot(t, S_new[:-1], "b", label = 'Susceptible')
plt.plot(t, I_new[:-1], "r", label = 'Infected')
plt.plot(t, R_new[:-1], "g", label = 'Recovered')

plt.plot(t, (S_new+I_new+R_new)[:-1],"k", label = 'Total')


plt.xlabel('Time/days')
plt.ylabel('number')
plt.title('SIRS with RK4 and vital dynamics')
plt.legend(loc="best")
plt.grid(True)
plt.savefig(fname='fig/newfig/RK4_vitalDynamics_d:002_dI:01_e:006.pdf')
#plt.show()







