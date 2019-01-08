
import numpy as np 
import matplotlib.pyplot as plt 

f = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x - 0.2
h = 0.05
x = np.linspace(0,1,11) 

# Central differences approximation
dfc1 = (f(x+h) - f(x-h))/(2*h)
dfc2 = (f(x+h) - 2*f(x) + f(x-h)) / h**2
print(('f\' ={}').format(dfc1))
print(('f\'\'={}').format(dfc2))

plt.plot(x,f(x),'-k', x,dfc1,'--b', x,dfc2,'-.r')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['f(x)','f \'(x)','f \'\'(x)'])
#plt.grid()
plt.show()
