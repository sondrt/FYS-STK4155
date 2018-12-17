#MonteCarloSIRS_vitaldynamics_SeasonalVariation.py
from numpy import *
import matplotlib.pyplot as plt
import random
import statistics


#a = 4 		# infection rate
b = 1 		# recovery rate
c = 0.5		# immunity lose			d001_dI01_e005.pdf

d  = 0.0002 	# death rate
d1 = 0.001	# death rate of infected
e  = 0.0006 	# birth rate

N = 400
S0 = 300
I0 = 100
R0 = 0
T = 365*5


#print(dt)
t = linspace(0,T,T)


S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)



#IC
S[0] = S0
I[0] = I0
R[0] = R0


def seasonalVariation(t):
	a0 = 4
	A = 10
	o_s_n_f = 2*pi/365.
	a = A*cos(t*o_s_n_f) + a0
	return (a)

def random_number():
	return(random.uniform(0,1))

#probability for move
def Psi(i,a,dt):
	return(a * S[i] * I[i] / N * dt  )

def Pir(i,dt):
	return(b * I[i] * dt)

def Prs(i,dt):
	return(c * R[i] * dt ) 

def vital_dynamics(i, d, d1, e):
	S[i] = S[i] + e * N - d * S[i]
	I[i] = I[i] - d * I[i] - d1*I[i] 
	R[i] = R[i] - d * R[i]
	return(S[i], I[i], R[i])

def MC():
	for i in range(len(t)):
		a = seasonalVariation(i)
		dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
		#if i > 0: 
		vital_dynamics(i, d, d1, e)

	
		#normalizing stuff
		normalizing_everything = Psi(i,a,dt)+Pir(i,dt)+Prs(i,dt)
		P_SI = Psi(i,a,dt)/normalizing_everything
		P_IR = Pir(i,dt)/normalizing_everything
		P_RS = Prs(i,dt)/normalizing_everything

		#a random number
		our_special_number = random_number()

			#tesing if empty and if move should happen
		if S[i] > 0 and P_SI > our_special_number:
			S[i+1] = S[i] - 1
			I[i+1] = I[i] + 1
			R[i+1] = R[i]

			#tesing if empty and if move should happen
		elif I[i] > 0 and P_IR + P_SI > our_special_number:
			S[i+1] = S[i]
			I[i+1] = I[i] - 1
			R[i+1] = R[i] + 1

			#tesing if empty and if move should happen
		elif R[i] > 0 and P_RS + P_SI + P_IR > our_special_number:
			S[i+1] = S[i] + 1
			I[i+1] = I[i]
			R[i+1] = R[i] - 1
	return(S,I,R)

S_new,I_new,R_new = MC()


#plot!
plt.figure()
plt.plot(t, S_new[:-1], label = 'Susceptible')
plt.plot(t, I_new[:-1], label = 'Infected')
plt.plot(t, R_new[:-1], label = 'Recovered')

#plotting total population
plt.plot(t, (S_new+I_new+R_new)[:-1],label = 'Total')

plt.xlabel('Time')
plt.ylabel('population')
plt.title('SIRS with MC, vital dynamics and seasonal variations')
plt.legend(loc="best")
plt.grid(True)
plt.savefig(fname='fig/newfig/MC_vitaldynamic_d02_dI10_e006.pdf')

#plt.savefig(fname='fig/newfig/MCVDSV_Stand_A10T5.pdf')
plt.show()

'''
RK4_vitalDynamics_d001_dI01_e05.pdf
RK4_vitalDynamics_d001_dI01_e005.pdf
RK4_vitalDynamics_d001_dI30_e005
RK4_vitalDynamics_d002_dI10_e006
'''
	

	




