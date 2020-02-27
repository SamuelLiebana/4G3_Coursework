import numpy as np
import matplotlib.pyplot as plt

# constants
K = 100 # total number of inputs
w = 5 # synaptic weight giving mean = V_th
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
# K = 100 poisson neurons as the input
for i in range(1, steps):
    voltage[i] = voltage[i-1] + delta_t*(-voltage[i-1]/tau) + (w/K)*np.sum(input_spike_train[:,i-1], axis = 0)

# plot the membrane potential along with a horizontal line for
# the mean and V_th
plt.plot(voltage, color = 'k')
plt.axvline(x =1000,linestyle = '--', color = 'g', label = 'End of Transient')
plt.plot(range(1000, len(voltage)), [1]*len(voltage[1000:]), color = 'b', label ='Vth')
plt.plot(range(1000, len(voltage)), [np.mean(voltage[1000:])]*len(voltage[1000:]),'--',  color = 'orange', label ='Empirical Mean')

# labelling y-axis
plt.ylabel('Voltage (a.u.)', size = 20, labelpad=10)
plt.yticks(size = 15)

# labelling x axis and defining x-ticks
plt.xlabel('Time (ms)', size = 20, labelpad = 10)
plt.xticks(np.linspace(0,steps, 11),np.linspace(0,steps/10, 11).astype(int), size = 15)
plt.legend(prop={'size': 15})

plt.show()
