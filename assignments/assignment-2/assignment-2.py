
# coding: utf-8

# # SYDE 556: Simulating Neurobiological Systems
# # Assignment 2: Spiking Neurons
# 
# **Author:** *Jonathan Johnston*
# 
# **Course Instructor:** *Professor C. Eliasmith*
# 
# This assignment details implementation of neuron models for representing temporal stimuli.
# 
# The assignment corresponds to the document hosted at:
# 
# http://nbviewer.ipython.org/github/celiasmith/syde556/blob/master/Assignment%202.ipynb

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp


# In[2]:

def calc_rms(x):
    '''Calculate root mean squared power of a signal x'''
    return np.sqrt(np.mean(np.power(x,2)))

# rms test
assert calc_rms([1,1,1]) == 1.0
assert calc_rms([3,3,3]) == 3.0


# ## Section 1: Generating a Random Input Signal
# 
# ### Section 1.1: Gaissuan White Noise

# In[3]:

def generate_signal(T,dt,rms,limit,seed=0):
    '''
    Return a randomly generated white noise signal limited
    to a specified frequency cutoff, as an array in both 
    the time and frequency domains (x, X)
    
    Keyword Arguments:
    T - length of signal in seconds
    dt - time step in seconds
    rms - root mean square power level of the signal
    limit - maximum frequency for the signal (in Hz)
    seed - the random number seed to use
    '''
    
    if seed != 0:
        np.random.seed(seed=seed)
    
    t = np.arange(0,T,dt) #time scale
    N = len(t) #samples
    # Need to reorder into linear scale
    f = np.fft.fftfreq(N,dt) #freq range (Hz)
    X = np.zeros(len(f)).tolist() #freq signal
    
    for i, freq in enumerate(f):
        if X[i] != 0: #skip filled values
            continue
        elif abs(freq) <= limit: #skip for freqs past limit
            real, imag = np.random.normal(), np.random.normal()*1j
            X[i] = real + imag
            if -freq in f and freq != 0: #symmetric values
                idx = np.where(f == -freq)
                X[idx[0]] = real - imag
    
    # Rescale the signal to rms threshold
    x = np.fft.ifft(X).real
    sig_rms = calc_rms(x)
    x = [sig*rms/(sig_rms) for sig in x]
    
    X = np.fft.fft(x) #convert updated signal to freq domain
    f, X = (list(t) for t in zip(*sorted(zip(f,X))))
    
    return x, X


# #### Part A
# 
# Plot three randomly generated signals of frequency 5, 10, and 20 Hertz.

# In[20]:

# time ranges and params
T, dt, rms = 1, 0.001, 0.5
t = np.arange(0,T,dt)

# generate 3 time plots
limits = [5,10,20]
for limit in limits:
    x, X = generate_signal(T=T,dt=dt,rms=rms,limit=limit)
    plt.plot(t,x)
    plt.title('Gaussian White Noise Signal at %d Hertz' % limit)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('$x$')
    plt.show()
    print('RMS:',calc_rms(x))


# #### Part B
# 
# Now we take 100 different seeds, generate 100 different signals, and average out the signals at each frequency by using the norm.

# In[21]:

# time ranges and params
num_seeds = 100
t = np.arange(0,T,dt)
N = len(t) #samples
T, dt, rms, limit = 1, 0.001, 0.1, 10

f = sorted(np.fft.fftfreq(N,dt)) #freqs (Hz)
w = [2*np.pi*freq for freq in f] #convert to radians
seeds = np.arange(1,num_seeds,1) #100 unique PRNG seeds

X_sigs = []
for seed in seeds:
    x,X = generate_signal(T=T,dt=dt,rms=rms,limit=limit,seed=seed)
    X_sigs.append(X)

X_sigs = np.array(X_sigs)
X_sigs_norm = [np.absolute(freq_vals) for freq_vals in X_sigs[:,]][0]

# Plot the average magnitudes
plt.plot(w,X_sigs_norm)
plt.title('Average Magnitude Over %d Signals' % num_seeds)
plt.xlabel('$\omega$ (radians)')
plt.ylabel('$|X(\omega)|$')
plt.xlim([-2*np.pi*limit - 20, 2*np.pi*limit + 20])
plt.show()


# ### Section 1.2: Gaussian Power Spectrum Noise

# In[7]:

def generate_dropoff_signal(T,dt,rms,bandwidth,seed=0):
    '''
    Return a randomly generated white noise signal with 
    dropoff at a specified bandwidth, as an array
    in both the time and frequency domains (x, X)
    
    Keyword Arguments:
    T - length of signal in seconds
    dt - time step in seconds
    rms - root mean square power level of the signal
    bandwidth - dropoff frequency for the signal (in Hz)
    seed - the random number seed to use
    '''
    
    if seed != 0:
        np.random.seed(seed=seed)
    
    t = np.arange(0,T,dt) #time scale
    N = len(t) #samples
    # Need to reorder into linear scale
    f = np.fft.fftfreq(N,dt) #freq range (Hz)
    X = np.zeros(len(f)).tolist() #freq signal
    
    for i, freq in enumerate(f):
        stddev = np.exp(-np.power(freq,2)/(2*np.power(bandwidth,2)))
        if X[i] != 0: #skip filled values
            continue
        elif stddev != 0.0:
            real = np.random.normal(scale=stddev)
            imag = np.random.normal(scale=stddev)*1j
            X[i] = real + imag
            if -freq in f and freq != 0: #symmetric values
                idx = np.where(f == -freq)
                X[idx[0]] = real - imag
    
    # rescale the signal to rms threshold
    x = np.fft.ifft(X).real
    sig_rms = calc_rms(x)
    x = [sig*(rms/sig_rms) for sig in x]
    
    X = np.fft.fft(x) #convert updated signal to freq domain
    
    return x, X


# #### Part A
# 
# Plot three randomly generated signals of frequency 5, 10, and 20 Hertz; however this time we use a signal generated with a frequency dropoff bandwidth.

# In[23]:

# time ranges and params
T, dt, rms = 1, 0.001, 0.5
t = np.arange(0,T,dt)

# generate 3 time plots
bandwidths = [5,10,20]
for bandwidth in bandwidths:
    x, X = generate_dropoff_signal(T=T,dt=dt,rms=rms,bandwidth=bandwidth)
    plt.plot(t,x)
    plt.title('GWN Bandwidth-Dropoff Signal at %d Hertz' % bandwidth)
    plt.xlabel('$t$ (seconds)')
    plt.ylabel('$x$')
    plt.show()
    print('RMS:',calc_rms(x))


# #### Part B
# 
# Now we take 100 different seeds, generate 100 different signals, and average out the signals at each frequency by using the norm; however this time we use the signal generated with a frequency dropoff bandwidth.

# In[9]:

# time ranges and params
num_seeds = 100
t = np.arange(0,T,dt)
N = len(t) #samples
T, dt, rms, bandwidth = 1, 0.001, 0.1, 10

f = np.fft.fftfreq(N,dt) #freqs (Hz)
w = [2*np.pi*freq for freq in f] #convert to radians
seeds = np.arange(1,num_seeds,1) #100 unique PRNG seeds

X_sigs = []
for seed in seeds:
    x,X = generate_dropoff_signal(T=T,dt=dt,rms=rms,bandwidth=bandwidth,seed=seed)
    X_sigs.append(X)

X_sigs = np.array(X_sigs)
X_sigs_norm = [np.absolute(freq_vals) for freq_vals in X_sigs[:,]]

# Fix the ordering
w, X_sigs_norm = (list(t) for t in zip(*sorted(zip(w,X_sigs_norm[0]))))
# Plot the average magnitudes
plt.plot(w,X_sigs_norm)
plt.xlabel('$\omega$ (radians)')
plt.ylabel('$|X(\omega)|$ (average of %s signals)' % num_seeds)
plt.xlim([-2*np.pi*bandwidth - 80, 2*np.pi*bandwidth + 80])
plt.show()


# ## Section 2: Simulating a Spiking Neuron

# In[10]:

class SpikingNeuron:
    def __init__(self, enc=1, tau_ref=0.002, tau_rc=0.02):
        self.min_rate = 40 # Hz, x = 0
        self.max_rate = 150 # Hz, x = {1,-1}; coincides with enc
        self.enc = enc
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        
        self.j_bias = 1/(1-np.exp((self.tau_ref-(1/self.min_rate))/self.tau_rc))
        self.alpha = 1/(1-np.exp((self.tau_ref-(1/self.max_rate))/self.tau_rc))-self.j_bias
    
    def print_vars(self):
        print(self.__dict__)
    
    def spikes(self, x, dt):
        '''Calculate voltage spikes for time-variant stimuli'''
        
        num_ref_steps = np.floor(self.tau_ref/dt)
        ref_count = 0
        spike_count = 0
        voltages = []
        v = 0
        v_next = 0
        for stim in x:
            J = self.alpha*stim*self.enc + self.j_bias
            if ref_count > 0:
                v = 0
                ref_count -= 1
            elif v >= 1: #spike
                v = 1.5 #constant spike voltage
                ref_count = num_ref_steps
                spike_count +=1
            elif v < 0: #keep positive
                v = 0
            
            v_next = v + dt*(1/self.tau_rc)*(J - v)
            voltages.append(v)
            v = v_next
        
        return voltages, spike_count


# #### Part A
# 
# 

# In[11]:

dt = 0.001
T = 1
t = np.arange(0,T,dt)
n = SpikingNeuron()

# Constant reference signal of x = 0
x = [0 for time in t]
spikes,num_spikes = n.spikes(x,dt)
print('Number of spikes:',num_spikes)

plt.plot(t, spikes, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.ylim(-0.2,1.6)
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.legend()
plt.show()

# Constant reference signal of x = 1
x = [1 for time in t]
spikes,num_spikes = n.spikes(x,dt)
print('Number of spikes:',num_spikes)

plt.plot(t, spikes, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.ylim(-0.2,1.6)
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.legend()
plt.show()


# #### Part B
# 
# Number of spikes observed is exact for the x = 0 case (40 spikes in 1 second); however the x = 1 case is limited to 143 spikes in 1 second, which is short of its theoretical 150 Hz rate at x = 1. In this simulation we used a time step of 0.001 seconds; however if we decrease this to a smaller time step, say by a factor of 10 to 0.0001 seconds, then we see the spike rate jump to 149 spikes in 1 second for the reference signal of x = 1. Further, another decrease in time step magnitude by a factor of 10 results in 150 Hz neuron activity for x = 1. This implies that the "clock speed" of our simulation, which we modify to approach continuous time, is a limiting factor in accurately simulating neuronal activity.

# #### Part C
# 
# Now we generate a signal using the Gaussian white noise distribution as in previous sections, generate the spikes using said signal, then overlay them both together.

# In[12]:

T, dt, rms, limit = 1, 0.001, 0.5, 30
x, X = generate_signal(T=T,dt=dt,rms=rms,limit=limit)
t = np.arange(0,T,dt)
n = SpikingNeuron()

# Randomly generated signal
spikes,num_spikes = n.spikes(x,dt)
print('Number of spikes:',num_spikes)

plt.plot(t, x, label='Input Signal',color='g')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$x$')
plt.show()

plt.plot(t, spikes, label='Neuron Voltage')
plt.plot(t, x, label='Input Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$x$')
plt.legend()
plt.show()


# #### Part D
# 
# Now we take a closer look at the signal and spikes from Part C.

# In[13]:

plt.plot(t, spikes, label='Neuron Voltage')
plt.plot(t, x, label='Input Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$x$')
plt.xlim(0.0,0.2)
plt.legend()
plt.show()


# #### Part E
# 
# The first obvious method for improving the accuracy of the spiking neuron simulation is to use a more accurate method for approximation beyond Euler's method. Examples of this includes Runge-Kutta and Heun's method. 
# 
# The next method for improving the accuracy of the spiking neuron simulation is to make the refractory period more accurate. In these areas, interpolation may be used in order to better approximate the next step's value if the time steps do not align well with the refractory periods.
# 
# In this assignment these possible improvements have not been implemented.

# ## Section 3: Simulating Two Spiking Neurons
# 
# #### Part A
# 
# Simulation of two spiking neurons of opposing encoders at x = 0.

# In[14]:

dt = 0.001
T = 1
t = np.arange(0,T,dt)

# Constant reference signal of x = 0
x = [0 for time in t]

# Encoder of +1
n1 = SpikingNeuron(enc=1)
spikes1,num_spikes1 = n1.spikes(x,dt)
print('Number of spikes for neuron with encoder=(1):',num_spikes1)

plt.plot(t, spikes1, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.ylim(-0.2,1.6)
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.legend()
plt.show()

# Encoder of -1
n2 = SpikingNeuron(enc=-1)
spikes2,num_spikes2 = n2.spikes(x,dt)
print('Number of spikes for neuron with encoder=(-1):',num_spikes2)

plt.plot(t, spikes2, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.ylim(-0.2,1.6)
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.legend()
plt.show()


# #### Part B
# 
# Spiking output of two spiking neurons with opposing encoders at x = 1.

# In[15]:

dt = 0.001
T = 1
t = np.arange(0,T,dt)

# Constant reference signal of x = 1
x = [1 for time in t]

# Encoder of +1
spikes1,num_spikes1 = n1.spikes(x,dt)
print('Number of spikes for neuron with encoder=(1):',num_spikes1)

plt.plot(t, spikes1, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-0.2,1.6)
plt.legend()
plt.show()

# Encoder of -1
spikes2,num_spikes2 = n2.spikes(x,dt)
print('Number of spikes for neuron with encoder=(-1):',num_spikes2)

plt.plot(t, spikes2, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-0.2,1.6)
plt.legend()
plt.show()


# #### Part C
# 
# Stimulate the two spiking neurons with a simple sine wave at 5 Hz.

# In[16]:

dt = 0.001
T = 1
t = np.arange(0,T,dt)

# Sine wave of 5 Hz
x = 0.5*np.sin(10*np.pi*t)

# Encoder of +1
spikes1,num_spikes1 = n1.spikes(x,dt)
print('Number of spikes for neuron with encoder=(1):',num_spikes1)

plt.plot(t, spikes1, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-0.5,1.6)
plt.legend()
plt.show()

# Encoder of -1
spikes2,num_spikes2 = n2.spikes(x,dt)
print('Number of spikes for neuron with encoder=(-1):',num_spikes2)

plt.plot(t, spikes2, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-0.5,1.6)
plt.legend()
plt.show()


# #### Part D
# 
# Simulate the two spiking neurons using a Gaussian white noise-generated signal of max frequency 5 Hz.

# In[17]:

T,dt,rms,limit = 2, 0.001, 0.5, 5
t = np.arange(0,T,dt)

# Random signal of max 5 Hz
x,X = generate_signal(T=T,dt=dt,rms=rms,limit=limit,seed=3)

# Encoder of +1
spikes1,num_spikes1 = n1.spikes(x,dt)
print('Number of spikes for neuron with encoder=(1):',num_spikes1)

plt.plot(t, spikes1, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-1.2,1.6)
plt.legend()
plt.show()

# Encoder of -1
spikes2,num_spikes2 = n2.spikes(x,dt)

print('Number of spikes for neuron with encoder=(-1):',num_spikes2)
plt.plot(t, spikes2, label='Neuron Voltage')
plt.plot(t, x, label='Reference Signal')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (voltage) of neuron')
plt.ylim(-1.2,1.6)
plt.legend()
plt.show()


# ## Section 4: Computing an Optimal Filter

# In[18]:

##
## Code adapted from part 2 and 3
##
def two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref):
    '''
    Calculate voltage spikes for time-variant stimuli
    for two spiking neurons with opposing encoders.
    '''
    voltage_set = []
    for enc in [-1,1]:
        num_ref_steps = np.floor(tau_ref/dt)
        ref_count = 0
        voltages = []
        v = 0
        v_next = 0
        for stim in x:
            J = alpha*stim*enc + Jbias
            if ref_count > 0:
                v = 0
                ref_count -= 1
            elif v >= 1: #spike
                v = 1.5 #constant spike voltage
                ref_count = num_ref_steps
            elif v < 0: #keep positive
                v = 0

            v_next = v + dt*(1/tau_rc)*(J - v)
            voltages.append(v)
            v = v_next
        voltage_set.append(voltages)
    voltage_set = np.array(voltage_set)
    return voltage_set[0],voltage_set[1]

##
## Code annotated with square brackets [like this]
##

import numpy

T = 4.0         # length of signal in seconds
dt = 0.001      # time step size

# [Generate white noise (same seed as section 3(d))]
x, X = generate_signal(T=T, dt=dt, rms=0.5, limit=5, seed=3)

Nt = len(x)                # [Number of time samples]
t = numpy.arange(Nt) * dt  # [Time range values (seconds)]

# Neuron parameters
tau_ref = 0.002          # [Refractory period (seconds)]
tau_rc = 0.02            # [Resistor-Capacitor (RC) constant]
x0 = 0.0                 # firing rate at x=x0 is a0
a0 = 40.0
x1 = 1.0                 # firing rate at x=x1 is a1
a1 = 150.0

# [Calculate J_bias and alpha for our neuron models]
eps = tau_rc/tau_ref
r1 = 1.0 / (tau_ref * a0)
r2 = 1.0 / (tau_ref * a1)
f1 = (r1 - 1) / eps
f2 = (r2 - 1) / eps
alpha = (1.0/(numpy.exp(f2)-1) - 1.0/(numpy.exp(f1)-1))/(x1-x0) 
x_threshold = x0-1/(alpha*(numpy.exp(f1)-1))              
Jbias = 1-alpha*x_threshold;

# [Simulate the two spiking neurons (from part 3)]
spikes = two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref)

freq = numpy.arange(Nt)/T - Nt/(2.0*T)   # [Generate set of frequency values]
omega = freq*2*numpy.pi                  # [Convert Hertz to radians]

r = spikes[0] - spikes[1]                # [Spike responses are opposites of each other]
R = np.array(numpy.fft.fftshift(numpy.fft.fft(r))) # [Take response into frequency domain and shift (center) it]
X = np.array(X)                          # [Do data type conversion]

sigma_t = 0.025                          # [Standard deviation of 0.025]
W2 = numpy.exp(-omega**2*sigma_t**2)     # [Gaussian curve over frequencies with stddev of sigma_t]
W2 = W2 / sum(W2)                        # [Normalize to sum to 1 so magnitude of signals aren't affected during convolution]

CP = X*R.conjugate()                  # 
WCP = numpy.convolve(CP, W2, 'same')  # 
RP = R*R.conjugate()                  # [Make response real by multiplying by conjugate]
WRP = numpy.convolve(RP, W2, 'same')  # [Make response continuous and smooth by convolving with Gaussian filter]
XP = X*X.conjugate()                  # [Make stimuli real by multiplying by conjugate]
WXP = numpy.convolve(XP, W2, 'same')  # [Make stimuli continuous and smooth by convolving with Gaussian filter]

H = WCP / WRP                         # [Get the filter]

h = numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.ifftshift(H))).real  # [Take filter into time domain]

XHAT = H*R                            # [Approximate freq-domain representation using filter and response]

xhat = numpy.fft.ifft(numpy.fft.ifftshift(XHAT)).real  # [Get time-domain approximation of stimuli]


import pylab

pylab.figure(1)
pylab.subplot(1,2,1)
pylab.plot(freq, numpy.sqrt(XP), label='Stim. power spectrum')  # [Real part response needs to be sqrt'd after having been 'squared' by conjugate]
pylab.legend()
pylab.xlabel('$\omega$ (radians)')
pylab.ylabel('$|X(\omega)|$')

pylab.subplot(1,2,2)
pylab.plot(freq, numpy.sqrt(RP), label='Spike response spectrum')  # [Real part response needs to be sqrt'd after having been 'squared' by conjugate]
pylab.legend()
pylab.xlabel('$\omega$ (radians)')
pylab.ylabel('$|R(\omega)|$')


pylab.figure(2)
pylab.subplot(1,2,1)
pylab.plot(freq, H.real)   # [Take real part of the filter]
pylab.xlabel('$\omega$ (radians)')
pylab.title('Optimal Filter Spectrum')
pylab.xlim(-50, 50)

pylab.subplot(1,2,2)
pylab.plot(t-T/2, h)       # [Filter in time-domain]
pylab.title('Optimal Fitler in Time-Domain')
pylab.xlabel('$t$ (seconds)')
pylab.xlim(-0.5, 0.5)


pylab.figure(3)
pylab.plot(t, r, color='k', label='Spiking activity', alpha=0.2)  # [Responses]
pylab.plot(t, x, linewidth=2, label='Stimuli (signal)')           # [Input signal]
pylab.plot(t, xhat, label='Neuron representation of signal')      # [Temporal representation]
pylab.title('Representation of Temporal Stimuli')
pylab.legend(loc='best')
pylab.xlabel('$t$ (seconds)')

pylab.show()

