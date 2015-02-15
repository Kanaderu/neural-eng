
# coding: utf-8

# # SYDE 556: Simulating Neurobiological Systems
# # Assignment 2: Spiking Neurons
# 
# **Author:** *Jonathan Johnston*
# 
# **Course Instructor:** *Professor C. Eliasmith*
# 
# This assignment is ... TODO: Description
# 
# The assignment corresponds to the document hosted at:
# 
# http://nbviewer.ipython.org/github/celiasmith/syde556/blob/master/Assignment%202.ipynb

# In[101]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp


# In[136]:

def calc_rms(x):
    '''Calculate root mean squared power of a signal x'''
    return np.sqrt(np.mean(np.power(x,2)))

# rms test
assert calc_rms([1,1,1]) == 1.0
assert calc_rms([3,3,3]) == 3.0


# ## Section 1: Generating a Random Input Signal
# 
# ### Section 1.1: Gaissuan White Noise

# In[174]:

def generate_signal(T,dt,rms,limit,seed=0):
    '''
    Return a randomly generated white noise signal as an array
    in both the time and frequency domains (x, X)
    
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
    f = np.fft.fftfreq(N,dt) #freq range (Hz)
    print(f)
    X = np.zeros(len(w)).tolist() #freq signal
    
    for i, freq in enumerate(f):
        if abs(freq) <= limit:
            real, imag = np.random.normal(), np.random.normal()*1j
            X[i] = real + imag
            if freq != 0:
                X[-i] = real - imag
    
    # rescale the signal to rms threshold
    x = np.fft.ifft(X).real
    sig_rms = calc_rms(x)
    x = [sig + (rms - sig_rms) for sig in x]
    X = np.fft.fft(x)
    
    return x, X


# #### Part A
# 
# Plot three randomly generated signals of frequency 5, 10, and 20 Hertz.

# In[175]:

# time ranges and params
T, dt, rms = 2, 0.01, 0.5
t = np.arange(0,T,dt)

# generate 3 time plots
limits = [5,10,20]
for limit in limits:
    x, X = generate_signal(T=T,dt=dt,rms=rms,limit=limit)
    print(np.shape(x))
    print(len(t))
    plt.plot(t,x)
    plt.show()
    print(calc_rms(x))


# #### Part B
# 
# 

# In[155]:

# time ranges and params
t = np.arange(0,T,dt)
N = len(t) #samples
T, dt, rms, limit = 1, 0.5, 0.1, 10

f = np.fft.fftfreq(N,dt) #freqs (Hz)
w = [2*np.pi*freq for freq in f] #convert to radians
seeds = np.arange(1,2,1) #100 unique PRNG seeds

X_sigs = []
for seed in seeds:
    x,X = generate_signal(T=T,dt=dt,rms=rms,limit=limit,seed=seed)
    X_sigs.append(X)

X_sigs = np.array(X_sigs)
X_sigs_norm = [np.linalg.norm(freq_vals) for freq_vals in X_sigs[:,]]

print(X_sigs)


# In[ ]:



