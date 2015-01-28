
# coding: utf-8

# # SYDE 556
# # Assignment 1: Representations in Populations of Neurons
# 
# **Author:** *Jonathan Johnston*
# 
# **Course Instructor:** *Professor C. Eliasmith*
# 
# This assignment is an introduction into the basic behaviours of a set of neuron models. It describes some ways in which neuron populations may hold a representation of a given stimuli, as well as the error associated in representation.
# 
# The assignment corresponds to the document hosted at:
# 
# http://nbviewer.ipython.org/github/celiasmith/syde556/blob/master/Assignment%201.ipynb

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, show, figure, suptitle, table, subplot, legend, axis
import matplotlib.pyplot as plt
import matplotlib as mp

def rmse(x, y):
   return np.sqrt(np.mean(np.power(x - y,2)))


# ## Section 1: Representation of Scalars
# ### 1.1 Basic Encoding and Decoding
# In this section we define a rectified linear neuron model which will be used in subsequent parts of the assignment.

# In[2]:

class RectifiedLinearNeuron:
    e_vals = [-1,1]

    def __init__(self):
        self.x_int = np.random.uniform(-0.95,0.95)
        self.max_fire_rate = np.random.uniform(100,200)
        self.encoder = np.random.choice(self.e_vals)
        self.alpha = (self.max_fire_rate)/abs(self.encoder - self.x_int)
        self.j_bias = -(self.alpha)*(self.x_int)*(self.encoder)

    def print_vars(self):
        print(self.__dict__)

    def rates(self, x):
        rates = []
        for pos in x:
            rate = max(self.alpha*pos*self.encoder+self.j_bias,0) # rectified
            rates.append(rate)
        return rates

def generate_RL_neurons(num, stimuli):
    neurons = []
    rates = []
    for i in range(num):
        n = RectifiedLinearNeuron()
        neurons.append(n)
        rates.append(n.rates(stimuli))
    return neurons, rates


# #### Part A
# In this section we generate a set of random rectified linear neurons and plot them together.

# In[3]:

figure()
suptitle('Random Rectified Linear Neurons')

num_neurons = 16
S = 40 # samples
x = np.linspace(-1,1,S)
neurons, rates = generate_RL_neurons(num_neurons, x)
for neuron, nrate in zip(neurons, rates):
    plot(x, nrate)

neurons[0].print_vars()
xlabel('$x$ (stimuli)')
ylabel('$a$ (Hz)')
show()


# #### Part B
# In this section we find the optimal decoders for the generated neuron population.

# In[4]:

A = np.transpose(rates) # matrix of neuron activities
gamma = np.dot(np.transpose(A),A)/S
upsilon = np.dot(np.transpose(A),x)/S
decoders = np.dot(np.linalg.inv(gamma),upsilon)
print('Decoders: ', decoders)


# #### Part C
# In this section, we decode firing rates into an approximation of the stimuli, which describes how the neurons represent the stimuli.

# In[5]:

x_approx = np.dot(A,decoders)
figure()
suptitle('Neural Representation of Stimuli')
plot(x,x_approx,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error = x - x_approx
figure()
suptitle('Representation Error')
plot(x,x_error)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse = rmse(x, x_approx)
print('Root Mean Squared Error (RMSE): ', x_rmse)


# #### Part D
# We decode under Gaussian noise proportional to the highest firing rate in the population.

# In[6]:

stddev_noise = 0.2*np.amax(A)
gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
A_noisy = A + gauss_noise

x_approx_noisy = np.dot(A_noisy,decoders)
figure()
suptitle('Neural Representation of Stimuli under Gaussian Noise')
plot(x,x_approx_noisy,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error_noisy = x - x_approx_noisy
figure()
suptitle('Representation Error with Gaussian Noise')
plot(x,x_error_noisy)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse_noisy = rmse(x, x_approx_noisy)
print('Root Mean Squared Error under Gaussian Noise (RMSE): ', x_rmse_noisy)


# #### Part E
# We recompute optimal decoders to take Gaussian noise into account.

# In[7]:

gamma_noisy = np.dot(np.transpose(A),A)/S + stddev_noise*np.identity(A.shape[1])
upsilon = np.dot(np.transpose(A),x)/S
decoders_noisy = np.dot(np.linalg.inv(gamma_noisy),upsilon)
print('Decoders under noise: ',decoders_noisy)

x_approx_nd = np.dot(A,decoders_noisy)
figure()
suptitle('Neural Representation of Stimuli with Noise-Optimized Decoders')
plot(x,x_approx_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error_nd = x - x_approx_nd
figure()
suptitle('Representation Error with Noise-Optimized Decoders')
plot(x,x_error_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse_nd = rmse(x, x_approx_nd)
print('Root Mean Squared Error with Noise-Optimized Decoders (RMSE): ', x_rmse_nd)

x_approx_noisy_nd = np.dot(A_noisy,decoders_noisy)
figure()
suptitle('Neural Representation of Stimuli under Gaussian Noise with Noise-Optimized Decoders')
plot(x,x_approx_noisy_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error_noisy_nd = x - x_approx_noisy_nd
figure()
suptitle('Representation Error with Gaussian Noise and Noise-Optimized Decoders')
plot(x,x_error_noisy_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse_noisy_nd = rmse(x, x_approx_noisy_nd)
print('Root Mean Squared Error under Gaussian Noise with Noise-Optimized Decoders (RMSE): ', x_rmse_noisy_nd)


# #### Part F
# 'Based on the RMSE of the variations of neuron noise and decoder noise '
#        'compensation, it appears that the noise introduces a hundred-fold '
#        'increase in the RMSE that the normal decoder produces. Compensating '
#        'for the noise by altering the decoder accordingly results in a better '
#        'result. However, in the non-noisy case, the compensation results in '
#        'higher error. In essence, using the decoder to compensate for noisy '
#        'neurons results in a lower variability in RMSE, which means a better '
#        'error in the worst case, but a worse error in the best case.'

# In[8]:

figure()
suptitle('Comparison of RMSE for variations of neuron noise and noise compensation')
# Remove plot so we can plot a simple table
ax = subplot(111, frame_on=False)
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
table_vals = [[x_rmse, x_rmse_nd],[x_rmse_noisy, x_rmse_noisy_nd]]
row_labels = ['No Noise', 'Gaussian Noise']
col_labels = ['Normal Decoder','Noise-Optimized Decoder']
rmse_table = table(cellText=table_vals, rowLabels=row_labels, colLabels=col_labels, colWidths=[0.4,0.4], loc='center')
for cell in rmse_table.properties()['child_artists']:
    cell.set_height(0.15) # increase cell height
show()


# ### 1.2 Exploring Sources of Error
# #### Part A and B
# Here, we explore various error associated with various numbers of neurons.

# In[9]:

stddev_factor_set = [0.1, 0.01]
S = 40 # samples
x = np.linspace(-1,1,S)
num_runs = 5
num_neuron_set = [4,8,16,32,64,128,256]
print('Number of neurons', num_neuron_set)

for stddev_factor in stddev_factor_set:
    error_dist_set = []
    error_noise_set = []
    for num_neurons in num_neuron_set:
        error_dist = 0
        error_noise = 0
        for run in range(num_runs):
            neurons, rates = generate_RL_neurons(num_neurons, x)
            A = np.transpose(rates)
            stddev_noise = stddev_factor*np.amax(A)
            
            gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
            A_noisy = A + gauss_noise
            
            gamma = np.dot(np.transpose(A_noisy),A_noisy)/S + stddev_noise*np.identity(A.shape[1])
            upsilon = np.dot(np.transpose(A_noisy),x)/S
            decoders = np.dot(np.linalg.inv(gamma),upsilon)
            x_approx = np.dot(A,decoders)
            error_dist += sum(np.power(x - x_approx,2))/S
            error_noise += stddev_noise*sum(np.power(decoders,2))

        
        error_dist_set.append(error_dist/num_runs)
        error_noise_set.append(error_noise/num_runs)
    
    fig = figure()
    suptitle('Log-Log Plot of Neuron Number versus Distortion Error with Noise Std Dev Factor of {SD}'.format(SD=stddev_factor))
    ax = fig.add_subplot(1,1,1)
    err = ax.plot(num_neuron_set, error_dist_set, label='Error')
    n = ax.plot(num_neuron_set, [error_dist_set[0]/N for N in num_neuron_set],'--g', label=r'$1/N$')
    n2 = ax.plot(num_neuron_set, [error_dist_set[0]/(N**2) for N in num_neuron_set],'--r', label=r'$1/N^2$')
    n4 = ax.plot(num_neuron_set, [error_dist_set[0]/(N**4) for N in num_neuron_set],'--y', label=r'$1/N^4$')
    legend(handles=[err,n,n2,n4],labels=[])
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel('Number of neurons')
    ylabel('$E_{dist}$ (distortion error)')
    axis([1,1000,10**(-8),0.1])
    show()
    
    fig = figure()
    suptitle('Log-Log Plot of Neuron Number versus Noise Error with Noise Std Dev Factor of {SD}'.format(SD=stddev_factor))
    ax = fig.add_subplot(1,1,1)
    err = ax.plot(num_neuron_set, error_noise_set, label='Error')
    n = ax.plot(num_neuron_set, [error_noise_set[0]/N for N in num_neuron_set],'--g', label=r'$1/N$')
    n2 = ax.plot(num_neuron_set, [error_noise_set[0]/(N**2) for N in num_neuron_set],'--r', label=r'$1/N^2$')
    n4 = ax.plot(num_neuron_set, [error_noise_set[0]/(N**4) for N in num_neuron_set],'--y', label=r'$1/N^4$')
    legend(handles=[err,n,n2,n4], labels=[])
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlabel('Number of neurons')
    ylabel('$E_{noise}$ (noise error)')
    axis([1,1000,10**(-8),0.1])
    show()
    
    print('Standard deviation factor of noise: ', stddev_factor)
    print('Error due to distortion: ', error_dist_set)
    print('Error due to noise: ', error_noise_set)


# #### Part C
# Comment on differences in parts A and B
# 
# While there is a significant difference in the total error resuling from a change in the standard deviation of the neuron noise, the proportionality between the number of neurons and the total error from distortion and noise is constant. The error due to noise is related to the number of neurons with a 1/N relationship, and the error due to distortion is related to the number of neurons with a 1/N^2 relationship. A massive decrease in error (10^6 fold) can be seen, for example, by increasing neuron population from 4 to 256. Only a modest improvement in error can be seen with a decrease in standard deviation of the noise. Also, at low neuron populations, distortion error is greater but at high neuron populations noise error is higher. Therefore more neurons will encode a stimuli better than fewer neurons, but noise error will not be overcome as easily with more neurons.

# ### 1.3 Leaky Integrate-and-Fire Neurons
# #### Part A
# Here we define a leaky integrate-and-fire neuron model.

# In[10]:

class LIFNeuron:
    e_vals = [-1,1]

    def __init__(self, tau_ref=0.002, tau_rc=0.02):
        self.x_int = np.random.uniform(-1,1)
        self.max_fire_rate = np.random.uniform(100,200)
        self.encoder = np.random.choice(self.e_vals)
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        
        exp_term = 1 - np.exp((tau_ref - 1/self.max_fire_rate)/tau_rc)
        enc_term = self.encoder**2 - self.encoder*self.x_int
        self.alpha = (1 - exp_term)/(exp_term*enc_term)
        self.j_bias = 1 - self.alpha*self.encoder*self.x_int

    def print_vars(self):
        print(self.__dict__)

    def rates(self, x):
        rates = []
        for pos in x:
            rate = 0
            J = self.alpha*pos*self.encoder+self.j_bias
            if J > 1:
                rate = 1/(self.tau_ref - self.tau_rc*np.log(1 - 1/J))
            rates.append(rate)
        return rates

def generate_LIF_neurons(num, stimuli):
    neurons = []
    rates = []
    for i in range(num):
        n = LIFNeuron()
        neurons.append(n)
        rates.append(n.rates(stimuli))
    return neurons, rates

figure()
suptitle('Random Leaky Integrate-and-Fire Neurons')

num_neurons = 16
S = 40 # samples
x = np.linspace(-1,1,S)
neurons, rates = generate_LIF_neurons(num_neurons, x)
for neuron, nrate in zip(neurons, rates):
    plot(x, nrate)

neurons[0].print_vars()
xlabel('$x$ (stimuli)')
ylabel('$a$ (Hz)')
show()


# #### Part B
# Now we find the error due to noise.

# In[11]:

stddev_noise = 0.2*np.amax(A)
A = np.transpose(rates)
gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
A_noisy = A + gauss_noise

gamma_noisy = np.dot(np.transpose(A),A)/S + stddev_noise*np.identity(A.shape[1])
upsilon = np.dot(np.transpose(A),x)/S
decoders_noisy = np.dot(np.linalg.inv(gamma_noisy),upsilon)
print('Noise-optimized decoders: ', decoders_noisy)

x_approx_nd = np.dot(A,decoders_noisy)
figure()
suptitle('Neural Representation of Stimuli with Noise-Optimized Decoders')
plot(x,x_approx_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error_nd = x - x_approx_nd
figure()
suptitle('Representation Error with Noise-Optimized Decoders')
plot(x,x_error_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse_nd = rmse(x, x_approx_nd)
print('Root Mean Squared Error with Noise-Optimized Decoders (RMSE): ', x_rmse_nd)

x_approx_noisy_nd = np.dot(A_noisy,decoders_noisy)
figure()
suptitle('Neural Representation of Stimuli under Gaussian Noise with Noise-Optimized Decoders')
plot(x,x_approx_noisy_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
show()

x_error_noisy_nd = x - x_approx_noisy_nd
figure()
suptitle('Representation Error with Gaussian Noise and Noise-Optimized Decoders')
plot(x,x_error_noisy_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
show()

x_rmse_noisy_nd = rmse(x, x_approx_noisy_nd)
print('Root Mean Squared Error under Gaussian Noise with Noise-Optimized Decoders (RMSE): ', x_rmse_noisy_nd)


# ## Section 2: Representation of Vectors
# ### 2.1 Vector Tuning Curves
# #### Part A
# We now bring the previous LIF model into two dimensions. It is worth noting that the plot is representing the entire grid of -1 to 1 on both the x and y axes. This means that the tuning curve appears as if it hits a higher frequency than is defined in its max firing rate. This is not the case, however, because the activity should stop at the unit circle defined around the origin (0,0).

# In[12]:

class TwoDimLIFNeuron:
    def __init__(self, enc_angle=0, maxrate=100, x_int=[0,0], tau_ref=0.002, tau_rc=0.02):
        self.x_int = x_int
        self.max_fire_rate = maxrate
        self.enc = [np.cos(enc_angle), np.sin(enc_angle)]
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        
        exp_term = 1 - np.exp((tau_ref - 1/self.max_fire_rate)/tau_rc)
        enc_term = np.vdot(self.enc,self.enc) - np.vdot(self.enc,x_int)
        self.alpha = (1 - exp_term)/(exp_term*enc_term)
        self.j_bias = 1 - self.alpha*np.vdot(self.enc,x_int)

    def print_vars(self):
        print(self.__dict__)

    def rates(self, x):
        rates = []
        for pos in x:
            rate = 0
            J = self.alpha*np.vdot(pos,self.enc)+self.j_bias
            if J > 1:
                rate = 1/(self.tau_ref - self.tau_rc*np.log(1 - 1/J))
            rates.append(rate)
        return rates

from mpl_toolkits.mplot3d import Axes3D

angle = -np.pi/4
neuron = TwoDimLIFNeuron(enc_angle=angle)
size = 100
x,y = np.linspace(-1,1,size), np.linspace(-1,1,size)
x,y = np.meshgrid(x,y)

points = []
for i in range(len(x)):
    for j in range(len(x)):
        points.append([x[j][i],y[j][i]])
rates = neuron.rates(points)
rates = np.reshape(rates,(size,size))

figure()
ax = plt.gca(projection='3d')
suptitle('Two-Dimensional Tuning Curve')
xlabel('x')
ylabel('y')
ax.set_zlabel('$a$ (Hz)')
surf = ax.plot_surface(X=x,Y=y,Z=rates,cmap=mp.cm.coolwarm)
show()


# #### Part B
# In this section we use a set of points distributed evenly around the unit circle as stimuli to find the corresponding firing rates.
# 
# We also fit a cosine on the resulting curve. The cosine is a good fit because it approximates the nature of taking values of a sort of exponential function along a circular curve. Where it fails is in its periodicity. The tuning curve is only active in a defined region, and is zero elsewhere, whereas the cosine curve is periodic and has no discontinuities. The cosine is also not ideal because the function being approximated is exponential, and therefore many cosines would be necessary to approximate it perfectly.

# In[13]:

size = 100
points = []
angles = np.linspace(-np.pi,np.pi,size)
for angle in angles:
    point = [np.cos(angle), np.sin(angle)]
    points.append(point)

rates = neuron.rates(points)

from scipy.optimize import curve_fit

def cosfunc(theta,a,b,c,d):
    return a*np.cos(b*theta + c) + d

popt,pcov = curve_fit(cosfunc,angles,rates)

cosfit = []
for angle in angles:
    cosfit.append(cosfunc(angle,popt[0],popt[1],popt[2],popt[3])) 

figure()
suptitle('Tuning Curve Along Unit Circle')
xlabel(r'$\theta$ (radians)')
ylabel('$a$ (Hz)')
tune = plot(angles,rates,'b',label='Tuning curve')
cos = plot(angles,cosfit,'--r',label='Cosine curve fit')
legend(handles=[tune,cos],labels=[])
show()


# ### 2.2 Vector Representation
# #### Part A
# We generate a set of 100 random, uniformly distributed unit vectors as encoders and plot them.

# In[14]:

num_encoders = 100
encoders = []
enc_angles = []
for i in range(num_encoders):
    angle = np.random.uniform(0,2*np.pi)
    unit = [np.cos(angle), np.sin(angle)]
    enc_angles.append(angle)
    encoders.append(unit)

u,v = zip(*encoders)
xy = np.zeros(len(encoders))

figure()
suptitle('Random Set of Encoders as Unit Vectors')
ax = plt.gca()
ax.quiver(xy,xy,u,v,angles='xy',scale_units='xy',scale=1)
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
xlabel('x')
ylabel('y')
show()


# #### Part B
# The encoders all lie on the unit circle, and therefore have magnitude one. The magnitudes of the decoders, on the other hand, are extremely small in comparison. It follows reason that in representing a stimuli that will have at maximum a value of one, a weighted sum of the firing rates of a large number of neurons will result in each neuron's weight being extremely small. In addition, the angles of the decoder vectors are distributed somewhat evenly, as are the angles of the encoder vectors.

# In[15]:

size = 40
x,y = np.linspace(-1,1,size), np.linspace(-1,1,size)
x,y = np.meshgrid(x,y)

stimuli = []
for i in range(len(x)):
    for j in range(len(x)):
        stimuli.append([x[j][i],y[j][i]])
        
neurons = []
rates = []
for enc_angle in enc_angles:
    angle = np.random.uniform(0,2*np.pi)
    x_int = [np.cos(angle), np.sin(angle)]
    maxrate = np.random.uniform(100,200)
    n = TwoDimLIFNeuron(enc_angle=enc_angle,maxrate=maxrate,x_int=x_int)
    neurons.append(n)
    rates.append(n.rates(stimuli))

A = np.transpose(rates)
stddev_noise = 0.2*np.amax(A)
gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
A_noisy = A + gauss_noise

gamma_noisy = np.dot(np.transpose(A_noisy),A_noisy)/S + stddev_noise*np.identity(A.shape[1])
upsilon = np.dot(np.transpose(A_noisy),stimuli)/S
decoders = np.dot(np.linalg.inv(gamma_noisy),upsilon)

u,v = zip(*decoders)
xy = np.zeros(len(decoders))

figure()
suptitle('Decoder Vectors for 2-D LIF Neuron Models')
ax = plt.gca()
plot(u,v,'.')
ax.set_xlim([-6*10**(-4),6*10**(-4)])
ax.set_ylim([-6*10**(-4),6*10**(-4)])
xlabel('x')
ylabel('y')
show()


# #### Part C
# Now we look at the stimuli as it is represented by the neuron population, and compare with the actual stimuli. As we can see, the accuracy of using the decoders to represent the stimuli is very good.

# In[16]:

stimuli = []
for i in range(20):
    angle = np.random.uniform(0,2*np.pi)
    radius = np.random.uniform(0,1)
    point = [radius*np.cos(angle), radius*np.sin(angle)]
    stimuli.append(point)

rates = []
for n in neurons:
    rates.append(n.rates(stimuli))

A = np.transpose(rates)
stim_approx = np.dot(A,decoders)

u,v = zip(*stimuli)
Ax,Ay = zip(*stim_approx)

figure()
suptitle('Decoded Stimuli for 2-D LIF Neuron Models')
ax = plt.gca()
s = plot(u,v,'.b',label='Original stimuli')
a = plot(Ax,Ay,'r^',label='Approximated stimuli')
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])
xlabel('x')
ylabel('y')
legend(handles=[s,a],labels=[])
show()

diffs = np.power(stimuli - stim_approx,2)
mean = np.mean([diff for diff in diffs])
rmse = np.sqrt(mean)

print('RMSE: ',rmse)


# #### Part D
# Using the encoders as decoders, we repeat Part C normally. As well, we repeat C and D whilst ignoring the magnitude of the representation and stimuli vectors.
# 
# As we can see, without normalizing the result, the error corresponding with the use of encoders as decoders is extremely high. Using the decoders themselves results in a much lower error.
# 
# When we normalize the result of decoding with the encoders, however, the RMSE is much lower than previously, and is only about 7 orders of magnitude greater than using the decoders as intended. The interesting part is that when we normalize the result of the **decoders** being used as intended, the RMSE is much higher than is acceptable.
# 
# This result is interesting because it implies that the computational effort expended when determining decoders is not entirely necessary if a slightly higher error is acceptable. This means that there exists a speed-accuracy trade-off when considering the use of encoders or decoders to decode the firing rates of neural populations in order to represent a stimulus.

# In[17]:

# Use the encoders instead of decoders
stim_enc_approx = np.dot(A,encoders)

u,v = zip(*stimuli)
Ax,Ay = zip(*stim_enc_approx)

figure()
suptitle('Decoded Stimuli for 2-D LIF Neuron Models using Encoders as Decoders')
ax = plt.gca()
s = plot(u,v,'.b',label='Original stimuli')
a = plot(Ax,Ay,'r^',label='Approximated stimuli')
xlabel('x')
ylabel('y')
legend(handles=[s,a],labels=[])
show()

diffs = np.power(stimuli - stim_enc_approx,2)
mean = np.mean([diff for diff in diffs])
rmse = np.sqrt(mean)

print('RMSE: ',rmse)

# Ignore magnitudes of resulting vectors
stimuli_norm = np.array([s/np.vdot(s,s) for s in stimuli])
stim_approx_norm = np.array([s/np.vdot(s,s) for s in stim_approx])
diffs = np.power(stimuli_norm - stim_approx_norm,2)
mean = np.mean([diff for diff in diffs])
rmse_norm = np.sqrt(mean)
print('RMSE, normalizing vectors: ',rmse_norm)

stim_enc_approx_norm = np.array([s/np.vdot(s,s) for s in stim_enc_approx])
diffs = np.power(stimuli - stim_enc_approx_norm,2)
mean = np.mean([diff for diff in diffs])
rmse_enc_norm = np.sqrt(mean)
print('RMSE for use of encoder as decoder, normalizing vectors: ',rmse_enc_norm)

