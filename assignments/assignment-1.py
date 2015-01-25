import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, show, draw, figure, suptitle, table, subplot
from nengutils import print_header

print_header('Section 1')
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

print_header('Part A',2)
print('Generate set of random rectified linear neurons and plot them')

figure(1)
suptitle('Random Rectified Linear Neurons')

neurons = []
rates = []
S = 100 # samples
x = np.linspace(-1,1,S)
for i in range(16):
    n = RectifiedLinearNeuron()
    n.print_vars()
    neurons.append(n)
    rates.append(n.rates(x))
    plot(x, rates[i])

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

x_rmse = np.sqrt(np.mean(np.power(x_error,2)))
print('Root Mean Squared Error (RMSE): ',x_rmse)

print_header('Part D',2)
print('Decode under Gaussian noise proportional to highest firing rate')

noise_stddev = 0.2*np.amax(A)
gauss_noise = np.random.normal(scale=noise_stddev,size=np.shape(A))
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

x_rmse_noisy = np.sqrt(np.mean(np.power(x_error_noisy,2)))
print('Root Mean Squared Error under Gaussian Noise (RMSE): ', x_rmse_noisy)

print_header('Part E',2)
print('Recompute optimal decoders to take Gaussian noise into account')

gamma_noisy = np.dot(np.transpose(A),A)/S + noise_stddev*np.identity(A.shape[1])
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

x_rmse_nd = np.sqrt(np.mean(np.power(x_error_nd,2)))
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

x_rmse_noisy_nd = np.sqrt(np.mean(np.power(x_error_noisy_nd,2)))
print('Root Mean Squared Error under Gaussian Noise with Noise-Optimized Decoders (RMSE): ', x_rmse_noisy_nd)

print_header('Part F')
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

# The end
print_header('End of script')
# Plot all figures
show()
