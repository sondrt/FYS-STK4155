#main to project1

import numpy as np
import matplotlib.pyplot as plt
from FrankFunction import FrankeFunction

n=10
x = sorted(np.random.rand(n)) #add noise with random.normal
y = sorted(np.random.rand(n))


noise = np.random.normal(n) #adder eller multipliser.
#print(x)
#print(y)
xp = np.linspace(0,1,n)

z = sorted(np.polyfit(x,y,deg=2))

x, y = np.meshgrid(x,y)

Frankenumbers = FrankeFunction(x,y)

if x not in  y:
	print(...)

#plt.plot(x,y,'-',xp,y,'--')
#plt.plot()
#plt.show()


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()