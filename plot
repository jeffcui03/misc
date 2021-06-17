

=======================================================================================================
Scatter Plots by category

https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(1974)

# Generate Data
num = 20
x, y = np.random.random((2, num))
labels = np.random.choice(['a', 'b', 'c'], num)
df = pd.DataFrame(dict(x=x, y=y, label=labels))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()

plt.show()

=========================================================================================
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
