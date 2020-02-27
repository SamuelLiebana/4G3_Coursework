import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# constants
K = 100 # number of inputs from each population (E, I, X)
N = 100 # total number of neurons in each population
rate = 10 # rate of poisson neurons in X population
sec = 2 # second interval
delta_t = 10**(-4) # length of timestep 
steps = int((1/delta_t)*sec) # number of steps in interval
tau = 20*10**(-3) # time constant
v_threshold = 1 # threshold voltage
J= np.array([[1,-2, 1], [1, -1.8, 0.8]]) # connectivity matrix

# initial conditions for voltage
V_E = np.zeros([N, steps]) # E will be denoted as group 1
V_I = np.zeros([N, steps]) # I will be denoted as group 2 

# random connectivity initialisation
C = np.zeros([2, 3, N], dtype = object)

# populating the connections for E and I neurons
for alpha in range(2):
    for beta in range(3):
        for i in range(N):
            #if alpha != beta:
                # sample K neurons from each group of N neurons (E, I, X)
            C[alpha, beta, i] = random.sample(range(N), K)
                # might have to be careful with self connections here (commented code)
            #if alpha == beta:
            #  C[alpha, beta, i] = random.sample([x for x in range(N) if x != i], K)

# Note that the neurons in X only connect into the other
# groups of neurons and do not receive any inputs. They
# are modelled as independent Poisson neurons.

# initial conditions for spike trains
S_E = np.zeros([N, steps])
S_I = np.zeros([N, steps])
S_X = np.zeros([N, steps])

# simulating neurons in X as artificial Poisson neurons
for i in range(N):
    S_X[i] = np.random.binomial(1, rate*delta_t, steps)

# simulating the neurons in E and I as LIF neurons receiving
# inputs from K neurons from all three populations (E, I, X)
for t in tqdm(range(1, steps)):
    for i in range(N):
        V_E[i, t] = V_E[i, t-1] + delta_t*(-V_E[i, t-1]/tau) + (1/np.sqrt(K))*(J[0, 0]*np.sum(S_E[C[0,0,i], t-1]) + J[0,1]*np.sum(S_I[C[0,1,i], t-1]) + J[0,2]*np.sum(S_X[C[0,2,i], t-1]))
        V_I[i, t] = V_I[i, t-1] + delta_t*(-V_I[i, t-1]/tau) + (1/np.sqrt(K))*(J[1, 0]*np.sum(S_E[C[1,0,i] ,t-1]) + J[1,1]*np.sum(S_I[C[1,1,i] ,t-1]) + J[1,2]*np.sum(S_X[C[1,2,i], t-1]))

        if V_E[i, t] > v_threshold:
            S_E[i, t] = 1
            V_E[i, t] = 0

        if V_I[i, t] > v_threshold:
            S_I[i, t] = 1
            V_I[i, t] = 0

# find posistion of output spikes for
# neuron number 5 in population
output_spikes = np.nonzero(S_E[5])

# create 2 subplots - 1: voltage, 2: output
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

# simple plot of voltage with axis label on the right
ax1.plot(V_E[5], color = 'k')
ax1.set_ylabel('Voltage (a.u.)', size = 15, labelpad = 10)
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
ax1.set_title('Membrane Potential', size = 14, rotation='horizontal', fontweight="bold", x=-0.08, y=0.4)

# output spike train using eventplot with axis label and tick disabled
ax2.eventplot(output_spikes, color = 'k')
ax2.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks on the left disabled
    labelleft= False,
    labelsize = 15)  # labels on the left disabled
ax2.set_title('Output Spike Train', size = 14, rotation='horizontal', fontweight="bold", x=-0.08, y=0.4)

# labelling x axis and defining x-ticks
plt.xlabel('Time (ms)', size = 20, labelpad = 10)
plt.xticks(np.linspace(0,steps, 11),np.linspace(0,steps/10, 11).astype(int), size = 15)

plt.show()

# calculating firing rates
print('Mean Output Excitatory Firing Rate: ', np.mean(np.count_nonzero(S_E, axis= 1))/sec)
print('Mean Output Inhibitory Firing Rate: ', np.mean(np.count_nonzero(S_I, axis= 1))/sec)
