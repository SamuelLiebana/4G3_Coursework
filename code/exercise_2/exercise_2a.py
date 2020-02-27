import numpy as np
import matplotlib.pyplot as plt

# constants
N = 1000 # total number of neurons
rate = 10 # rate of input poisson neuron
sec = 2 # second interval
delta_t = 10**(-4) # length of timestep 
steps = int((1/delta_t)*sec) # number of steps in interval
w = 0.9 # synaptic weight
v_threshold = 1 # threshold voltage
tau = 20*10**(-3) # time constsnt

# initial conditions for voltage
voltage = np.zeros(steps)
# input spike train comes from a Poisson neuron
input_spike_train = np.random.binomial(1, rate*delta_t, steps)
# output train placeholder
output_spike_train = np.zeros(steps)

# implement 20000 timesteps (2s) of the LIF neuron with the
# poisson neuron as its input and the spike-and-reset mechanism activated
for i in range(1, steps):
    voltage[i] = voltage[i-1] + delta_t*(-voltage[i-1]/tau) + w*input_spike_train[i-1]
    if voltage[i] > v_threshold:
        output_spike_train[i] = 1
        voltage[i] = 0

# find positions of spikes in the input and output
input_position_train = np.nonzero(input_spike_train)
output_position_train = np.nonzero(output_spike_train)

# create three subplots - top: input, middle: voltage, bottom: output
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# input spike train using eventplot with axis label and tick disabled
ax1.eventplot(input_position_train, color = 'k')
ax1.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks on the left disabled
    labelleft= False)  # labels on the left disabled
ax1.set_title('Input Spike Train', size = 14, rotation='horizontal', fontweight="bold",x=-0.07,y=0.4)

# simple plot of voltage with axis label on the right
ax2.plot(voltage, color = 'k')
ax2.plot([1]*len(voltage), '--', color = 'r')
ax2.set_ylabel('Voltage (a.u.)', size = 15, labelpad = 10)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title('Membrane Potential', size = 14, rotation='horizontal', fontweight="bold",x=-0.08,y=0.4)

# ouput spike train using eventplot with axis label and tick disabled
ax3.eventplot(output_position_train, color = 'k')
ax3.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks on the left disabled
    labelleft= False)  # labels on the left disabled
ax3.set_title('Output Spike Train', size = 14, rotation='horizontal', fontweight="bold",x=-0.08,y=0.4)

# labelling x axis and defining x-ticks
plt.xlabel('Time (ms)', size = 20, labelpad = 10)
plt.xticks(np.linspace(0,steps, 11),np.linspace(0,steps/10, 11).astype(int), size = 15)

plt.show()