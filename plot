====================================================================================================
#Remove "," in number
#index_col=False, cannot use index_col=None if there is , at the end of the each line in the input file

stats_df =pd.read_csv('/qau/scratch/cuix/t3/t3_exec_stats.csv', index_col=False, thousands=',')

======================================================================================================
#String nanosecond to ptyhon "tm"

def chTm2Nsec(chTm):
    #subStr=chTm[:-3]
    #print "input=",chTm
    try:
        nsecs = 0
        usecs = 0
        nsecs = chTm[-3:]
        subStr= chTm[:-3]
        #print chTm, subStr, nsecs
        tm = datetime.strptime(subStr, '%Y-%m-%dD%H:%M:%S.%f')
        usecs=tm.hour*3600000000L + tm.minute*60000000L+ tm.second*1000000L +  tm.microsecond
        return usecs*1000L+ long(nsecs)
    except ValueError:
        print("Exception: ", chTm)
        return 0


==========================================================================================================
#How to convert q query result in "nbbo_tbl' to np.array and to pd.DataFrame

nas_K = q('nbbo_tbl')
nasQuoteArr = np.array(nas_K)
nasQuoteArr

nasQuoteDf = pd.DataFrame(np.array(nas_K))
nasQuoteDf.head()



=======================================================================================================
Plot by Order Time

Method 1:
bbjd_x_df  = pd.DataFrame(dict(x=bbjd_df.ordtime, y=bbjd_df.ordqty, label=bbjd_df.side))

error: float() argument must be a string or a number not 'datetime" 
x is of type "datetime.time"
How to fix this issue?  set the column to index 
bbjd_x_df.set_index(['x'], inplace=True)	


Method 2
	time	bprice	sprice
0	0 days 09:00:30.020810	64.18	64.23
1	0 days 09:00:30.361300	64.18	64.22
2	0 days 09:00:30.363338	64.16	64.22
3	0 days 09:00:30.363384	64.18	64.22
4	0 days 09:00:30.364367	64.18	64.22
type(nasQuoteDf['time'][0])
pandas._libs.tslibs.timedeltas.Timedelta
plot_start = datetime.datetime(2021, 1, 27, 0, 0, 0)
#plot_start
nasQuoteDf['time_pd'] = nasQuoteDf['time'].apply(lambda x: plot_start + x) 

title_ticker='PLUG'
plt.figure(figsize=(20, 10))
plt.plot(nasQuoteDf['time_pd'], nasQuoteDf['sprice'], color='r', alpha=0.3)
plt.plot(nasQuoteDf['time_pd'], nasQuoteDf['bprice'], color='g', alpha=0.3)
#all trades happen at the 14:59:00 timestamp, needs more granularity
plt.title(title_ticker+" Continous Trading Near Open")


=========================================================================================================




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
