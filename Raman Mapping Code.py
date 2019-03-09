import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpld3
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic(u'matplotlib')
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'qt')
#mpld3.enable_notebook()


# In[17]:


##   Import ASCII data file (full path of the of ASCII file)

data = np.genfromtxt(r'Path of th File')

m=np.shape(data)
print(data)
print(m)


# In[8]:


p=0
for i in range(1,m[0]):     #range for i will be 1 to 37 i.e. range(1,37)
    plt.plot(data[0, 2:m[1]], data[i, 2:m[1]])
    p=p+1
print(p)
plt.show()


# In[4]:


for i in range(2,m[1]):
    if data[0,i]>=380 and data[0,i]<=380.5:
        temp1=i
    if data[0,i]>=410 and data[0,i]<=415:
        temp2=i
print(temp1,temp2)


# In[18]:


posx=[]
posy=[]
p=0
for i in range(1,m[0]):     #range for i will be 1 to 37 i.e. range(1,37)
    posx.append(data[i,0])
    posy.append(data[i,1])
    p =p+1
    plt.plot(data[0, temp1:temp2], data[i, temp1:temp2])
#print(posx,posy)
n=len(data[0, temp1:temp2])
print(n,p)
plt.show()


# In[19]:

### Fitting of the peak using Gaussian and Lorentzian Function

def gaussian(x,mean,amp,sigma):
    
    #    Gaussian Function 
    f=np.exp((-(x-mean)**2)/(2*sigma**2))*amp   
    
    #Lorentzian Function
    #f= (sigma**2 / ((x-mean)**2 + sigma**2) ) *amp
    return f

# In[20]:


from scipy import optimize
import peakutils 
posx = []
posy = []
v1 = []
v2 = []
delta_v = []


for i in range(1,m[0]):     #m[0]
    posx.append(data[i,0])
    posy.append(data[i,1])
    y = np.asarray(data[i,temp1:temp2])
    base = peakutils.baseline(y, deg=None, max_it=None, tol=None)  #baseline correction
    corr_data = data[i,temp1:temp2]-base
    #print(corr_data)
    
    
    #plt.plot(data[0,temp1:temp2],base)
    def cost(parameters):
        g_0 = parameters[:3]
        g_1 = parameters[3:6]
        #p_0 = parameters[6:9]
        return np.sum(np.power(gaussian(data[0,temp1:temp2], *g_0) + gaussian(data[0,temp1:temp2], *g_1) - corr_data, 2)) /len(data[0, temp1:temp2])
       
    
    initial_guess = [385,100,2,405,100,5]
    result = optimize.minimize(cost, initial_guess)#, bounds='bnds',tol=1e-15)
    
    # Fitted Peak Position 
    v1.append(result.x[0])
    v2.append(result.x[3])
    
    # Fitting Parameters 
    
    print('steps', result.nit, result.fun)
    print(f'g_0: mean: {result.x[0]:3.3f} amplitude: {result.x[1]:3.3f} sigma: {result.x[2]:3.3f}')
    print(f'g_1: mean: {result.x[3]:3.3f} amplitude: {result.x[4]:3.3f} sigma: {result.x[5]:3.3f}') # offset: {result.x[6]:3.3f}')
    print(f'p_0: a: {result.x[6]:3.3f} b: {result.x[7]:3.3f} c: {result.x[8]:3.3f}')# d: {result.x[9]:3.3f}')
    
    
    fitted_data = gaussian(data[0,temp1:temp2],*result.x[:3])+gaussian(data[0,temp1:temp2],*result.x[3:6])
    #print(fitted_data)
    
    peak_area = np.trapz(fitted_data,data[0,temp1:temp2],dx=0.005)
    #print(peak_area)
    
    
    if peak_area>5.0 and (result.x[3]-result.x[0])>15.0 and (result.x[3]-result.x[0])<25:
        delta_v.append(result.x[3]-result.x[0])
    else:
        delta_v.append(15)
        #plt.plot( data[0,temp1:temp2], fitted_data)
        #plt.plot(data[0,temp1:temp2],corr_data)
        #print(peak_area)
                  
    plt.plot( data[0,temp1:temp2], fitted_data, label = 'Fitted Curve')
    plt.plot(data[0,temp1:temp2],corr_data,'.', 'Actual Data point')
plt.legend()
plt.show()


# In[13]:


#print(np.shape(delta_v))
d = np.reshape(delta_v,(25,23))
#d = np.reshape(delta_v,(6,6))

#print(d)

x = np.linspace(-5,5,23)
y = np.linspace(-5,5,25)
X,Y = np.meshgrid(x,y)
plt.contourf(X,Y,d, 500,cmap=cm.hot,)
plt.colorbar(label='Separation of $E{^1}_2g$ and $A_{1g}$')
plt.show()
