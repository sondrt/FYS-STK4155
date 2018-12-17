#MonteCarloSIRS_vitaldynamics_SeasonalVariation.py
from numpy import *
import matplotlib.pyplot as plt
import random
import statistics
#random.uniform(0,1)
#print(random.uniform(0,1))
a = 4.
b = 1. #1.,2.,3.,4.
c = 0.5

d = 0.0002 	#TEST VALUES
d1 = 0.0004	#TEST VALUES
e = 0.0006 		#TEST VALUES

N = 400
S0 = 300
I0 = 100
R0 = 0
T = 10000

dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
#print(dt)
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

def vital_dynamics(i, d, d1, e):
	S[i] = S[i] + e * N - d * S[i]
	I[i] = I[i] - d * I[i] - d1*I[i] 
	R[i] = R[i] - d * R[i]
	return(S[i], I[i], R[i])

def MC():
	for i in range(len(t)):
		if i > 0: 
			vital_dynamics(i, d, d1, e)

		while True:
			if S[i] > 0 and Psi(i) > random_number():
				S[i+1] = S[i] - 1
				I[i+1] = I[i] + 1
				R[i+1] = R[i]
				break

			
			elif I[i] > 0 and Pir(i) > random_number():
				S[i+1] = S[i]
				I[i+1] = I[i] - 1
				R[i+1] = R[i] + 1
				break

			elif R[i] > 0 and Prs(i) > random_number():
				S[i+1] = S[i] + 1
				I[i+1] = I[i]
				R[i+1] = R[i] - 1
				break
			else:
				S[i+1] = S[i]
				I[i+1] = I[i]
				R[i+1] = R[i]
				break
	return(S,I,R)

S_new,I_new,R_new = MC()
#print(mean(S_new),mean(I_new),mean(R_new))
#S = 0.47 I = 0.10, R = 0.425

#print(statistics.pstdev(S_new),statistics.pstdev(I_new),statistics.pstdev(R_new))
#S = 33.12810317951664  I = 17.67014859256619 R = 28.248528572795212



#plot!
plt.figure()
plt.plot(t, S_new[:-1], label = 'Susceptible')
plt.plot(t, I_new[:-1], label = 'Infected')
plt.plot(t, R_new[:-1], label = 'Recovered')


plt.xlabel('Time')
plt.ylabel('population')
plt.title('SIRS with MC')
plt.legend(loc="best")
plt.grid(True)
#plt.savefig(fname='fig/MCSIRS_b3.pdf')
plt.show()




	




