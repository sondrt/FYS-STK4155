#MonteCarloSIRS_vitaldynamics_SeasonalVariation.py
from numpy import *
import matplotlib.pyplot as plt
import random
import statistics


#a = 4 			# infection rate
b = 1 			# recovery rate
c = 0.5			# immunity lose

d  = 0.0002 	# death rate
d1 = 0.001		# death rate of infected
e  = 0.0006 	# birth rate
f = 40			# Vaccine, A fixed number of shots per time period.

N = 400
S0 = 300
I0 = 100
R0 = 0

T = 365*5
#dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
t = linspace(0,T,T)

S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)

#Initial conditions
S[0] = S0
I[0] = I0
R[0] = R0

#help functions
def random_number():
	'''still pretty dumb'''
	return(random.uniform(0,1))

def Psi(i,a,dt):
	return(a * S[i] * I[i] / N * dt  )

def Pir(i,dt):
	return(b * I[i] * dt)

def Prs(i,dt):
	return(c * R[i] * dt ) 

def seasonalVariation(t):
	'''Variates the spread of the disease with a period, here sett to one year.'''
	a0 = 4
	A = 4.
	o_s_n_f = 2*pi/365.
	a = A*cos(t*o_s_n_f) + a0
	return (a)

def vital_dynamics(i, d, d1, e):
	''' Includes vital dynamics to the system '''
	S[i] = S[i] + e * N - d * S[i] 
	I[i] = I[i] - d * I[i] - d1*I[i] 
	R[i] = R[i] - d * R[i]
	return(S[i], I[i], R[i])

def vaccination(i):
	''' Include the possibility for vaccinations'''
	S[i] = S[i] - f
	I[i] = I[i]
	R[i] = R[i] + f
	return(S[i],I[i],R[i])

def MC():
	'''The Monte Carlo simulation.'''
	N = 400
	for i in range(len(t)):
		a = seasonalVariation(i)
		dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
#		if i > 0: #People desirve a chance to live one lousy day.
#			vital_dynamics(i, d, d1, e)

		if (i%365 == 0 ):
			vaccination(i)


		'''normalizing stuff, and making sure that one and only one move occures every time'''
		normalizing_everything = Psi(i,a,dt)+Pir(i,dt)+Prs(i,dt)
		P_SI = Psi(i,a,dt)/normalizing_everything
		P_IR = Pir(i,dt)/normalizing_everything
		P_RS = Prs(i,dt)/normalizing_everything

		#a random number
		our_special_number = random_number()


		if S[i] > 0 and P_SI > our_special_number:
			S[i+1] = S[i] - 1
			I[i+1] = I[i] + 1
			R[i+1] = R[i]

		
		elif I[i] > 0 and P_IR + P_SI > our_special_number:
			S[i+1] = S[i]
			I[i+1] = I[i] - 1
			R[i+1] = R[i] + 1


		elif R[i] > 0 and P_RS + P_SI + P_IR > our_special_number:
			S[i+1] = S[i] + 1
			I[i+1] = I[i]
			R[i+1] = R[i] - 1
#	N = S[i] + I[i] + R[i] 
	return(S,I,R)

S_new,I_new,R_new = MC()




#plot!
plt.figure()
plt.plot(t, S_new[:-1], label = 'Susceptible')
plt.plot(t, I_new[:-1], label = 'Infected')
plt.plot(t, R_new[:-1], label = 'Recovered')

plt.plot(t, (S_new+I_new+R_new)[:-1],label = 'Total')

plt.xlabel('Time')
plt.ylabel('population')
plt.title('SIRS with MC')
plt.legend(loc="best")
plt.grid(True)
plt.savefig(fname='fig/newfig/MC_vaccine_SVA=4_T=5f=40_notiming.pdf')
plt.show()




	




