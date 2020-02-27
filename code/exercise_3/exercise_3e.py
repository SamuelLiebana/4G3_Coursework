import numpy as np
import matplotlib.pyplot as plt

# constants
K = 100 # number of input neurons
w = 4.275 # synaptic weight giving output spiking rate = 10
rate = 10 # rate of input neurons
sec = 100 # second interval
secs_to_plot = 5
delta_t = 10**(-4) # length of timestep 
steps = int((1/delta_t)*sec) # number of steps in interval
steps_to_plot = int((1/delta_t)*secs_to_plot)
tau = 20*10**(-3) # time constant
v_threshold = 1 # threshold voltage

# initial conditions for voltage
voltage = np.zeros(steps)
# initialise input_spike_train to all 0's
input_spike_train = np.zeros([K, steps])
# initialise output spike train to all 0's
output_spike_train = np.zeros(steps)

# populate the spike train matrix with spike trains
# these are modelled by sampling 1000000 times (10s) from a
# bernoulli distribution with p = r_x*delta_t = 0.001
for i in range(K):
    input_spike_train[i] = np.random.binomial(1, rate*delta_t, steps)
    np.random.seed()

# implement 1000000 timesteps (10s) of the LIF neuron with
# K= 100 poisson neurons as the input
for i in range(1, steps):
    voltage[i] = voltage[i-1] + delta_t*(-voltage[i-1]/tau) + (w/K)*np.sum(input_spike_train[:,i-1], axis = 0)
    if voltage[i] > v_threshold:
        output_spike_train[i] = 1
        voltage[i] = 0

# find the positions of the output spikes
output_position_train = np.nonzero(output_spike_train)
output_position_train_to_plot = np.nonzero(output_spike_train[:steps_to_plot])

# create three subplots - top: input, middle: voltage, bottom: output
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

# simple plot of voltage with axis label on the right
ax1.plot(voltage[:steps_to_plot], color = 'k')
ax1.set_ylabel('Voltage (a.u.)', size = 15, labelpad = 10)
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
ax1.set_title('Membrane Potential', size = 14, rotation='horizontal', fontweight="bold", x=-0.08, y=0.4)
ax1.grid(True, which= 'both')

# ouput spike train using eventplot with axis label and tick disabled
ax2.eventplot(output_position_train_to_plot, color = 'k')
ax2.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks on the left disabled
    labelleft= False,
    labelsize = 15)  # labels on the left disabled
ax2.set_title('Output Spike Train', size = 14, rotation='horizontal', fontweight="bold", x=-0.08, y=0.4)
ax2.grid(True, which='both')

# labelling x axis and defining x-ticks
plt.xlabel('Time (ms)', size = 20, labelpad = 10)
plt.xticks(np.linspace(0,steps_to_plot, 11),np.linspace(0,steps_to_plot/10, 11).astype(int), size = 15)

plt.show()

print('Mean Output Firing Rate: ', np.count_nonzero(output_position_train)/sec)

# calculating the Fano factor
window = int(0.1/delta_t)
windowed_spike_counts = np.convolve(output_spike_train, np.ones((window,)), mode='valid')
fano_factor = np.var(windowed_spike_counts)/np.mean(windowed_spike_counts)
print('Fano Factor: ', fano_factor)
print('Mean Fires in Window', np.mean(windowed_spike_counts))
print('Variance of Fires in Window', np.var(windowed_spike_counts))

