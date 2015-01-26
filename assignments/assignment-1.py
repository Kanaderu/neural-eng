import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, show, draw, figure, suptitle, table, subplot, legend, axis
from nengutils import print_header, rmse

print_header('SYDE 556: Assignment 1')
print_header('Section 1: Representation of Scalars')
print_header('Section 1.1',1)
print('Define rectified linear neuron model')

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


print_header('Part A',2)
print('Generate set of random rectified linear neurons and plot them')

figure(1)
suptitle('Random Rectified Linear Neurons')

num_neurons = 16
S = 100 # samples
x = np.linspace(-1,1,S)
neurons, rates = generate_RL_neurons(num_neurons, x)
for neuron, nrate in zip(neurons, rates):
    neuron.print_vars()
    plot(x, nrate)

xlabel('$x$ (stimuli)')
ylabel('$a$ (Hz)')
draw()

print_header('Part B',2)
print('Find the optimal decoders for the neurons')

A = np.transpose(rates) # matrix of neuron activities
gamma = np.dot(np.transpose(A),A)/S
upsilon = np.dot(np.transpose(A),x)/S
decoders = np.dot(np.linalg.inv(gamma),upsilon)
print(decoders)

print_header('Part C',2)
print('Decode firing rates into approximation of stimuli')

x_approx = np.dot(A,decoders)
figure(2)
suptitle('Neural Representation of Stimuli')
plot(x,x_approx,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error = x - x_approx
figure(3)
suptitle('Representation Error')
plot(x,x_error)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse = rmse(x, x_approx)
print('Root Mean Squared Error (RMSE): ', x_rmse)

print_header('Part D',2)
print('Decode under Gaussian noise proportional to highest firing rate')

stddev_noise = 0.2*np.amax(A)
gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
A_noisy = A + gauss_noise

x_approx_noisy = np.dot(A_noisy,decoders)
figure(4)
suptitle('Neural Representation of Stimuli under Gaussian Noise')
plot(x,x_approx_noisy,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error_noisy = x - x_approx_noisy
figure(5)
suptitle('Representation Error with Gaussian Noise')
plot(x,x_error_noisy)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse_noisy = rmse(x, x_approx_noisy)
print('Root Mean Squared Error under Gaussian Noise (RMSE): ', x_rmse_noisy)

print_header('Part E',2)
print('Recompute optimal decoders to take Gaussian noise into account')

gamma_noisy = np.dot(np.transpose(A),A)/S + stddev_noise*np.identity(A.shape[1])
upsilon = np.dot(np.transpose(A),x)/S
decoders_noisy = np.dot(np.linalg.inv(gamma_noisy),upsilon)
print(decoders_noisy)

x_approx_nd = np.dot(A,decoders_noisy)
figure(6)
suptitle('Neural Representation of Stimuli with Noise-Optimized Decoders')
plot(x,x_approx_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error_nd = x - x_approx_nd
figure(7)
suptitle('Representation Error with Noise-Optimized Decoders')
plot(x,x_error_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse_nd = rmse(x, x_approx_nd)
print('Root Mean Squared Error with Noise-Optimized Decoders (RMSE): ', x_rmse_nd)

x_approx_noisy_nd = np.dot(A_noisy,decoders_noisy)
figure(8)
suptitle('Neural Representation of Stimuli under Gaussian Noise with Noise-Optimized Decoders')
plot(x,x_approx_noisy_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error_noisy_nd = x - x_approx_noisy_nd
figure(9)
suptitle('Representation Error with Gaussian Noise and Noise-Optimized Decoders')
plot(x,x_error_noisy_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse_noisy_nd = rmse(x, x_approx_noisy_nd)
print('Root Mean Squared Error under Gaussian Noise with Noise-Optimized Decoders (RMSE): ', x_rmse_noisy_nd)

print_header('Part F',2)
print('Show a table with all four RMSE values')

figure(10)
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
draw()

print(('Based on the RMSE of the variations of neuron noise and decoder noise '
       'compensation, it appears that the noise introduces a hundred-fold '
       'increase in the RMSE that the normal decoder produces. Compensating '
       'for the noise by altering the decoder accordingly results in a better '
       'result. However, in the non-noisy case, the compensation results in '
       'higher error. In essence, using the decoder to compensate for noisy '
       'neurons results in a lower variability in RMSE, which means a better '
       'error in the worst case, but a worse error in the best case.'))

print_header('Section 1.2',1)
print_header('Part A and B',2)
print('Explore various error associated with various numbers of neurons')

stddev_factor_set = [0.1, 0.01]
fig_num = 10
S = 100 # samples
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
            A = A + gauss_noise
            
            gamma = np.dot(np.transpose(A),A)/S + stddev_noise*np.identity(A.shape[1])
            upsilon = np.dot(np.transpose(A),x)/S
            decoders = np.dot(np.linalg.inv(gamma),upsilon)
            x_approx = np.dot(A,decoders)
            error_dist += sum(np.power(x - x_approx,2))/S
            error_noise += stddev_noise*sum(np.power(decoders,2))

        
        error_dist_set.append(error_dist/num_runs)
        error_noise_set.append(error_noise/num_runs)
    
    fig = figure(fig_num+1)
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
    draw()
    
    fig = figure(fig_num+2)
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
    draw()
    
    print('Standard deviation factor of noise: ', stddev_factor)
    print('Error due to distortion: ', error_dist_set)
    print('Error due to noise: ', error_noise_set)
    fig_num += 2

print_header('Part C',2)
print('Comment on differences in parts A and B')

print(('While there is a significant difference in the total error resuling '
       'from a change in the standard deviation of the neuron noise, the '
       'proportionality between the number of neurons and the total error '
       'from distortion and noise is constant. The error due to noise is '
       'related to the number of neurons with a 1/N relationship, and '
       'the error due to distortion is related to the number of neurons '
       'with a 1/N^2 relationship. A massive decrease in error (10^6 fold) '
       'can be seen, for example, by increasing neuron population from '
       '4 to 256. Only a modest improvement in error can be seen with a '
       'decrease in standard deviation of the noise. Also, at low neuron '
       'populations, distortion error is greater but at high neuron '
       'populations noise error is higher. Therefore more neurons will '
       'encode a stimuli better than fewer neurons, but noise error '
       'will not be overcome as easily with more neurons'))

print_header('Section 1.3',1)
print('Define a leaky integrate-and-fire neuron model')

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

figure(15)
suptitle('Random Leaky Integrate-and-Fire Neurons')

num_neurons = 16
S = 100 # samples
x = np.linspace(-1,1,S)
neurons, rates = generate_LIF_neurons(num_neurons, x)
for neuron, nrate in zip(neurons, rates):
    neuron.print_vars()
    plot(x, nrate)

xlabel('$x$ (stimuli)')
ylabel('$a$ (Hz)')
draw()

print_header('Part B',2)
print('Find error due to noise')

stddev_noise = 0.2*np.amax(A)
A = np.transpose(rates)
gauss_noise = np.random.normal(scale=stddev_noise,size=np.shape(A))
A_noisy = A + gauss_noise

gamma_noisy = np.dot(np.transpose(A),A)/S + stddev_noise*np.identity(A.shape[1])
upsilon = np.dot(np.transpose(A),x)/S
decoders_noisy = np.dot(np.linalg.inv(gamma_noisy),upsilon)
print(decoders_noisy)

x_approx_nd = np.dot(A,decoders_noisy)
figure(16)
suptitle('Neural Representation of Stimuli with Noise-Optimized Decoders')
plot(x,x_approx_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error_nd = x - x_approx_nd
figure(17)
suptitle('Representation Error with Noise-Optimized Decoders')
plot(x,x_error_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse_nd = rmse(x, x_approx_nd)
print('Root Mean Squared Error with Noise-Optimized Decoders (RMSE): ', x_rmse_nd)

x_approx_noisy_nd = np.dot(A_noisy,decoders_noisy)
figure(18)
suptitle('Neural Representation of Stimuli under Gaussian Noise with Noise-Optimized Decoders')
plot(x,x_approx_noisy_nd,'b')
plot(x,x,'r')
xlabel('$x$ (stimuli)')
ylabel(r'$\hat x$ (approximation)')
draw()

x_error_noisy_nd = x - x_approx_noisy_nd
figure(19)
suptitle('Representation Error with Gaussian Noise and Noise-Optimized Decoders')
plot(x,x_error_noisy_nd)
xlabel('$x$ (stimuli)')
ylabel(r'$x - \hat x$ (error)')
draw()

x_rmse_noisy_nd = rmse(x, x_approx_noisy_nd)
print('Root Mean Squared Error under Gaussian Noise with Noise-Optimized Decoders (RMSE): ', x_rmse_noisy_nd)


# The end
print_header('End of script')
# Plot all figures
show()
