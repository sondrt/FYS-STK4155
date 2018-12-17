#MonteCarloSIRS.py
from numpy import *
import matplotlib.pyplot as plt
import random
import statistics

# Initialize 

a = 4 		# infection rate
b = 1		# recovery rate
c = 0.5		# immunity lose

N = 400		# Tot population
S0 = 300
I0 = 100
R0 = 0

T = 365*10	# a year * number of years
dt = min(4./(a * N), 1./(b * N) , 1./(c * N))
t = linspace(0,T,T)

S = zeros(len(t)+1)
I = zeros(len(t)+1)
R = zeros(len(t)+1)

#Initial Conditions
S[0] = S0
I[0] = I0
R[0] = R0



#help funksjons 
def random_number():
	'''This is just dumb'''
	return(random.uniform(0,1))


def Psi(i):
	'''probability for move S -> I'''
	return(a * S[i] * I[i] / N * dt)

def Pir(i):
	'''probability for move I -> R'''
	return(b * I[i] * dt)

def Prs(i):
	'''probability for move R -> S'''
	return(c * R[i] * dt ) 

#main function
def MC():
	''' Monte Carlo simulation '''
	for i in range(len(t)):		
		'''normalizing stuff, and making sure that one and only one move occures every time'''
		normalizing_everything = Psi(i)+Pir(i)+Prs(i)
		P_SI = Psi(i)/normalizing_everything
		P_IR = Pir(i)/normalizing_everything
		P_RS = Prs(i)/normalizing_everything

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

#finding mean
Smean, Imean, Rmean = mean(S_new),mean(I_new),mean(R_new)
print(Smean, Imean, Rmean)
#finding standard deviation.
print(statistics.pstdev(S_new),statistics.pstdev(I_new),statistics.pstdev(R_new))
#S = 33.12810317951664  I = 17.67014859256619 R = 28.248528572795212
#S = 33.12  17.67 28.24
# analytical equilibrium 
Ssteady = b / a 
Isteady =  (1 - b / a) / (1 + b / c)
Rsteady = b / c * (1 - b / a) / (1 + b / c)



#for easier plotting:
n = zeros(len(t))
V_Ss = linspace(Ssteady * N, Ssteady * N,len(t))
V_Is = linspace(Isteady * N, Isteady * N,len(t))
V_Rs = linspace(Rsteady * N, Rsteady * N,len(t))

V_Smean = linspace(Smean, Smean, len(t))
V_Imean = linspace(Imean, Imean, len(t))
V_Rmean = linspace(Rmean, Rmean, len(t))

plt.figure()
plt.plot(t, S_new[:-1], "b", label = 'MC Susceptible')
plt.plot(t, I_new[:-1], "r", label = 'MC Infected')
plt.plot(t, R_new[:-1], "g", label = 'MC Recovered')




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
S_new2, I_new2, R_new2  = RK4SIRS( S0, I0, R0, dt)



# analytical equilibrium 
Ssteady = b / a 
Isteady =  (1 - b/a)/(1+b/c)
Rsteady = b/c * (1 - b/a)/(1+b/c)


#for easier plotting:
n = zeros(len(t))
V_Ss = linspace(Ssteady * N, Ssteady * N,len(t))
V_Is = linspace(Isteady * N, Isteady * N,len(t))
V_Rs = linspace(Rsteady * N, Rsteady * N,len(t))

# #plot!
# plt.figure()
# plt.plot(t, S_new[:-1], "b", label = 'Susceptible')
# plt.plot(t, I_new[:-1], "r", label = 'Infected')
# plt.plot(t, R_new[:-1], "g", label = 'Recovered')

#plotting the equilibrium lines.
# plt.plot(t,V_Ss,"--b", label = 'S equilibrium')
# plt.plot(t,V_Is,"--r", label = 'I equilibrium')
# plt.plot(t,V_Rs,"--g", label = 'R equilibrium')

# #plotting the total population N, to see if it changes( and for when it changes. )
# plt.plot(t, (S_new+I_new+R_new)[:-1],"k", label = 'Total')

# plt.xlabel('Time')
# plt.ylabel('population')
# plt.title('4th order RK on SIRS')
# plt.legend(loc="best")
# plt.grid(True)
# #plt.savefig(fname='fig/RK4_b4T1.pdf')
# plt.show()














#plot!


plt.plot(t, S_new2[:-1], "b", label = 'RK4 Susceptible')
plt.plot(t, I_new2[:-1], "r", label = 'RK4 Infected')
plt.plot(t, R_new2[:-1], "g", label = 'RK4 Recovered')


#plotting the equilibrium lines.
plt.plot(t,V_Ss,"--b", label = 'S equilibrium')
plt.plot(t,V_Is,"--r", label = 'I equilibrium')
plt.plot(t,V_Rs,"--g", label = 'R equilibrium')

#plotting mean. 
# plt.plot(t,V_Smean,"-.b", label = 'S mean')
# plt.plot(t,V_Imean,"-.r", label = 'I mean')
# plt.plot(t,V_Rmean,"-.g", label = 'R mean')

#plotting total population.
#'plt.plot(t, (S_new+I_new+R_new)[:-1],"k", label = 'Total')

plt.xlabel('Time')
plt.ylabel('population')
plt.title('Monte Carlo simulation on SIRS')
plt.legend(loc="best")
plt.grid(True)
#plt.savefig(fname='fig/newfig/MCRK4_b1T3.pdf')
#plt.show()






	
'''
@book{NumPy,
author = {Oliphant, Travis},
year = {2006},
month = {01},
title = {Guide to NumPy}
}
@article{yang2014comparison,
  title={Comparison of filtering methods for the modeling and retrospective forecasting of influenza epidemics},
  author={Yang, Wan and Karspeck, Alicia and Shaman, Jeffrey},
  journal={PLoS computational biology},
  volume={10},
  number={4},
  year={2014},
  publisher={Public Library of Science}
}
@article{zaman2008stability,
  title={Stability analysis and optimal vaccination of an SIR epidemic model},
  author={Zaman, Gul and Kang, Yong Han and Jung, Il Hyo},
  journal={BioSystems},
  volume={93},
  number={3},
  pages={240--249},
  year={2008},
  publisher={Elsevier}
}
'''



