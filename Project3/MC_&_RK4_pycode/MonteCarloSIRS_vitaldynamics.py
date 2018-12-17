#MonteCarloSIRS_vitaldynamics.py
from numpy import *
import matplotlib.pyplot as plt
import random
import statistics


#a = A*cos(omega * t) + a0	# infection rate
a = 4		# infection rate
b = 1 		# recovery rate
c = 0.5		# immunity lose

d  = 0.00002 	# death rate
dI = 0.003	# death rate of infected
e  = 0.00006	# birth rate

N = 400
S0 = 300
I0 = 100
R0 = 0
T = 365*10

dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
print(dt)
t = linspace(0,T,T)


S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)



#IC
S[0] = S0
I[0] = I0
R[0] = R0




def random_number():
	return(random.uniform(0,1))

#probability for move
def Psi(i):
	return(a * S[i] * I[i] / N * dt  )

def Pir(i):
	return(b * I[i] * dt)

def Prs(i):
	return(c * R[i] * dt ) 

def vital_dynamics(i, d, dI, e):
	S[i] = S[i] + e * N - d * S[i]
	I[i] = I[i] - d * I[i] - dI*I[i] 
	R[i] = R[i] - d * R[i]
	return(S[i], I[i], R[i])


def MC():


	for i in range(len(t)):		

		vital_dynamics(i, d, dI, e)

		#normalizing stuff
		normalizing_everything = Psi(i)+Pir(i)+Prs(i)
		P_SI = Psi(i)/normalizing_everything
		P_IR = Pir(i)/normalizing_everything
		P_RS = Prs(i)/normalizing_everything

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

		else: 
			S[i+1] = S[i]
			I[i+1] = I[i]
			R[i+1] = R[i]


	return(S,I,R)	

S_new,I_new,R_new = MC()




#plot!
plt.figure()
plt.plot(t, S_new[:-1], label = 'Susceptible')
plt.plot(t, I_new[:-1], label = 'Infected')
plt.plot(t, R_new[:-1], label = 'Recovered')

plt.plot(t, (S_new+I_new+R_new)[:-1],"k", label = 'Total')

plt.xlabel('Time')
plt.ylabel('population')
plt.title('SIRS with MC and vital dynamics')
plt.legend(loc="best")
plt.grid(True)
plt.savefig(fname='fig/newfig/MC_vitaldynamic_d00002_dI003_e00006.pdf')
plt.show()

'''
RK4_vitalDynamics_d001_dI01_e05.pdf
RK4_vitalDynamics_d001_dI01_e005.pdf
RK4_vitalDynamics_d001_dI30_e005
RK4_vitalDynamics_d002_dI10_e006



d  = 0.00002 	# death rate
dI = 0.00004	# death rate of infected
e  = 0.00006	# birth rate

d  = 0.00002 	# death rate
dI = 0.0001	# death rate of infected
e  = 0.00006	# birth rate

d  = 0.00002 	# death rate
dI = 0.001	# death rate of infected
e  = 0.00006	# birth rate

'''



 

