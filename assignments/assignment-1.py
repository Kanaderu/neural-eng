import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, show, draw

def print_header(header, level=0):
    encl_len = len(header) + 4
    enclosure = ''
    if level <= 0:
        enclosure = '='
    elif level == 1:
        enclosure = '-'
    elif level >= 2:
        enclosure = ' '
    print(enclosure*encl_len)
    print(' '.join([enclosure, header, enclosure]))
    print(enclosure*encl_len)

print_header("Section 1")
print_header("Section 1.1",1)

class RectifiedLinearNeuron:
    e_vals = [-1,1]

    def __init__(self):
        self.x_int = np.random.uniform(-0.95,0.95)
        self.max_fire_rate = np.random.uniform(100,200)
        self.encoder = np.random.choice(self.e_vals)
        self.alpha = (self.max_fire_rate)/abs(self.encoder - self.x_int)
        self.j_bias = -(self.alpha)*(self.x_int)*(self.encoder)

    def print_vars(self):
        vars = 'x-int: {}, max_fire_rate: {}, encoder: {}, gain: {}, bias: {}'.format(self.x_int, self.max_fire_rate, self.encoder, self.alpha, self.j_bias)
        print(vars)

    def rates(self, x):
        rates = []
        for pos in x:
            rate = max(self.alpha*pos*self.encoder+self.j_bias,0) # rectified
            rates.append(rate)
        return rates

print_header("Part A",2)

print("Generate set of random rectified linear neurons and plot them")
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

xlabel('x (stimuli)')
ylabel('$a$ (Hz)')
draw()

print_header("Part B",2)
print("Find the optimal decoders for the neurons")
A = rates # matrix of neuron activities
#gamma = np.dot(np.transpose(A),A)/S
#print(np.size(gamma))
#upsilon = np.dot(A,x)/S
#print(np.size(upsilon))
#decoders = np.dot(np.linalg.inv(gamma),upsilon)
#print(decoders)


print_header("End of script")

# Plot all figures
show()
