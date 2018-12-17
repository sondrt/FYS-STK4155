#MonteCarloSIRS.py
from numpy import *
import matplotlib.pyplot as plt
import random
#random.uniform(0,1)
#print(random.uniform(0,1))
a = 4.
b = 8. #1.,2.,3.,4.
c = 0.5
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
I[0] = I0
R[0] = R0



dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
#print(4./(a * N), 1./(b * N) , 1./(c * N))
print("dt = ", dt)

def random_number():
	return(random.uniform(0,1))

#probability for move
def Psi(i):
	return(a * S[i] * I[i] / N * dt)

def Pir(i):
	return(b * I[i] * dt)

def Prs(i):
	return(c * R[i] * dt ) 


def MC():
	for i in range(N):
		c1=0
		c2=0
		c3=0
		if S[i] > 0 and Psi(i) > random_number():
			S[i+1] = S[i] - 1
			I[i+1] = I[i] + 1
			c1 = 1

		elif I[i] > 0 and Pir(i) > random_number():
			I[i+1] = I[i] - 1
			R[i+1] = R[i] + 1
			c2 = 1

		elif R[i] > 0 and Prs(i) > random_number():
			R[i+1] = R[i] - 1
			S[i+1] = S[i] + 1
			c3 = 1

		if c1 == 0 and c3 == 0:
			S[i + 1] = S[i]

		if c2 == 0 and c1 == 0: 
			I[i + 1] = I[i]

		if c3 == 0 and c2 == 0:
			R[i + 1] = R[i]

	return(S,I,R)


S_new,I_new,R_new = MC()
print(S_new)



#plot!
plt.figure()
plt.plot(n, S_new, label = 'Susceptible')
plt.plot(n, I_new, label = 'Infected')
plt.plot(n, R_new, label = 'Recovered')

#plt.axis([0,10, 0,400])

plt.xlabel('N')
plt.ylabel('value')
plt.title('SIRS')
plt.legend(loc="best")
plt.grid(True)
plt.savefig(fname='fig/MCSIRS_b8.pdf')
plt.show()



	




