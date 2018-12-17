"""
#RK4_SIR_function.py
from numpy import *
import matplotlib.pyplot as plt

# INITIALIZE
a = 4.
b = 1.
c = 0.5
dt = 0.01
N = 400
S0 = 300
I0 = 100
R0 = 0

n = linspace(0,N,N+1)
S = zeros(N+1)
I = zeros(N+1)
R = zeros(N+1)


#IC
S[0] = S0
R[0] = R0
I[0] = I0


#Define help funcitons
def dSdt( t, S, I):
	return( - a * S * I / N )


def dIdt( t, S, I):
	return(a * S * I / N - b * I)

def dRdt( t, I): #here for the lolz
	return(b * I)

def RK4SIRS( N, a, b, S0, I0, R0, dt):
	N = S0 + I0 + R0

	for i in range(N):
		Si = S[i]
		Ii = I[i]  
#		RI = R[I]
		#print(Si)
		SK1 = dSdt(i, Si, Ii)
		IK1 = dIdt(i, Si, Ii)
#		RK1 = dRdt(i, Ii)

		SK2 = dSdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1)
		IK2 = dIdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1)
#		RK2 = dRdt( i + dt / 2, Ii + dt/ 2 * IK1 )

		SK3 = dSdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2)
		IK3 = dIdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2)
#		RK3 = dRdt( i + dt / 2, Ii + dt/ 2 * IK2 )

		SK4 = dSdt( i + dt, Si + dt * SK3, Ii + dt * IK3)
		IK4 = dIdt( i + dt, Si + dt * SK3, Ii + dt * IK3)
#		RK4 = dRdt( i + dt, Ii + dt * IK3)

		S[i + 1] = Si + dt / 6 * (SK1 + 2 * SK2 + 2 * SK3 + SK4)
		I[i + 1] = Ii + dt / 6 * (IK1 + 2 * IK2 + 2 * IK3 + IK4)
#		R[i + 1] = Ri + dt / 6 * (RK1 + 2 * RK2 + 2 * RK3 + RK4)

#	BC
	R =  N - S - I 
	return(S, I, R)

#run!
S,I,R = RK4SIR(N, a, b, S0, I0, R0, dt)




#plot!
plt.figure()
plt.plot(n, S, label = 'Susceptible')
plt.plot(n, I, label = 'Infected')
plt.plot(n, R, label = 'Recovered')

#plt.axis([0,10, 0,400])

plt.xlabel('N')
plt.ylabel('value')
plt.title('SIR')
plt.legend()

plt.show()

"""

#RK4_SIRS_function.py
from numpy import *
import matplotlib.pyplot as plt

# INITIALIZE
a = 4.
b = 1.
c = 0.5
dt = 0.01
N = 400
S0 = 300
I0 = 100
R0 = 0

n = linspace(0,N,N+1)
S = zeros(N+1)
I = zeros(N+1)
R = zeros(N+1)


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

def RK4SIR( N, S0, I0, R0, dt):
	N = S0 + I0 + R0

	for i in range(N):
		Si = S[i]
		Ii = I[i]  
		Ri = R[i]

		SK1 = dSdt(i, Si, Ii)
		IK1 = dIdt(i, Si, Ii)
		RK1 = dRdt(i, Ii)

		SK2 = dSdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1)
		IK2 = dIdt( i + dt / 2, Si + dt / 2 * SK1, Ii + dt / 2 * IK1)
		RK2 = dRdt( i + dt / 2, Ii + dt/ 2 * IK1 )

		SK3 = dSdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2)
		IK3 = dIdt( i + dt / 2, Si + dt / 2 * SK2, Ii + dt / 2 * IK2)
		RK3 = dRdt( i + dt / 2, Ii + dt/ 2 * IK2 )

		SK4 = dSdt( i + dt, Si + dt * SK3, Ii + dt * IK3)
		IK4 = dIdt( i + dt, Si + dt * SK3, Ii + dt * IK3)
		RK4 = dRdt( i + dt, Ii + dt * IK3)

		S[i + 1] = Si + dt / 6 * (SK1 + 2 * SK2 + 2 * SK3 + SK4)
		I[i + 1] = Ii + dt / 6 * (IK1 + 2 * IK2 + 2 * IK3 + IK4)
		R[i + 1] = Ri + dt / 6 * (RK1 + 2 * RK2 + 2 * RK3 + RK4)

#	BC
	R =  N - S - I 
	return(S, I, R)

#run!
S,I,R = RK4SIR(N, S0, I0, R0, dt)




#plot!
plt.figure()
plt.plot(n, S, label = 'Susceptible')
plt.plot(n, I, label = 'Infected')
plt.plot(n, R, label = 'Recovered')

#plt.axis([0,10, 0,400])

plt.xlabel('N')
plt.ylabel('value')
plt.title('SIR')
plt.legend()

plt.show()
























































