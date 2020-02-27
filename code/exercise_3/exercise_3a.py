import numpy as np
import matplotlib.pyplot as plt

# constants
K = 100 # number of input neurons
w = 1 # synaptic weight
rate = 10 # rate of input poisson neurons
sec = 2 # second interval
delta_t = 10**(-4) # length of timestep 
steps = int((1/delta_t)*sec) # number of steps in interval
tau = 20*10**(-3) # time constant

# initial conditions for voltage
voltage = np.zeros(steps)
# initialise input_spike_train to all 0's
input_spike_train = np.zeros([K, steps])
# populate the spike train matrix with spike trains
# these are modelled by sampling 20000 times (2s) from a
# bernoulli distribution with p = r_x*delta_t = 0.001
for i in range(K):
    input_spike_train[i] = np.random.binomial(1, rate*delta_t, steps)
    np.random.seed()

# implement 20000 timesteps (2s) of the LIF neuron with
# K= 100 poisson neurons as the input
for i in range(1, steps):
    voltage[i] = voltage[i-1] + delta_t*(-voltage[i-1]/tau) + (w/K)*np.sum(input_spike_train[:,i-1], axis = 0)

# plot the resulting voltage
plt.plot(voltage, color = 'k')
# indicate the end of the 100ms transient
plt.axvline(x =1000,linestyle = '--', color = 'g', label = 'End of Transient')
# plot mean and standard deviation lines in blue and red respectively
plt.plot(range(1000, len(voltage)), [np.mean(voltage[1000:])]*len(range(1000, len(voltage))), color = 'b', label='mean')
plt.plot(range(1000, len(voltage)),[np.mean(voltage[1000:]) + 2*np.std(voltage[1000:])]*len(range(1000, len(voltage))), color = 'r', label='+/- 2std')
plt.plot(range(1000, len(voltage)),[np.mean(voltage[1000:]) - 2*np.std(voltage[1000:])]*len(range(1000, len(voltage))), color = 'r')
# labelling y-axis
plt.ylabel('Voltage (a.u.)', size = 15, labelpad=10)
# labelling x axis and defining x-ticks 
plt.xlabel('Time (ms)', size = 15, labelpad = 10)
plt.xticks(np.linspace(0,steps, 11),np.linspace(0,steps/10, 11).astype(int), size = 10)
# displaying legend
plt.legend(prop={'size': 10})
plt.show()
