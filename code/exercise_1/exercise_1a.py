import numpy as np
import matplotlib.pyplot as plt

# constants
N = 1000 # total number of neurons
rate = 10
sec = 2 # second interval
delta_t = 10**(-4) # length of timestep 
steps = int((1/delta_t)*sec) # number of steps in interval

spike_train_matrix = np.zeros([N, steps]) # rows are neurons and 
                                          # columns are timestamps

# populate the spike train matrix with spike trains
# these are modelled by sampling 20000 times (2s) from a
# bernoulli distribution with p = r_x * delta_t = 0.001
for i in range(N):
    spike_train_matrix[i] = np.random.binomial(1, rate*delta_t, steps)
    np.random.seed()

plt.figure(1) # plot first 10 spike trains in a raster plot

# find the positions of the entries of the matrix which are nonzero
position_matrix = np.argwhere(np.transpose(spike_train_matrix[0:10]) > 0)
# unzip the points into arrays for x and y coordinates and scatter plot
plt.scatter(*zip(*position_matrix), s = 500, marker = "|", color = 'k')

# give x axis label for the spike raster plot
plt.xlabel('Time (ms)', size = 15)
# give y axis label for the spike raster plot
plt.ylabel('Neuron', size = 15)
# give x and y axes the correct tick marks
plt.yticks(np.linspace(0,9).astype(int), np.linspace(1,10).astype(int))
plt.xticks(np.linspace(0,steps, 5), np.linspace(0,steps/10, 5).astype(int))

plt.figure(2) # create new figure with all 1000 spike trains

# find the positions of the entries of the matrix which are nonzero
position_matrix = np.argwhere(np.transpose(spike_train_matrix) > 0)
# unzip the points into arrays for x and y coordinates and scatter plot
plt.scatter(*zip(*position_matrix), color = 'k', marker = ".", s= 10)

# give x axis label and ticks for the spike raster plot
plt.xlabel('Time (ms)', size = 15)
plt.xticks(np.linspace(0,steps, 5),np.linspace(0,steps/10, 5).astype(int))
# Give y axis label for the spike raster plot
plt.ylabel('Neuron', size = 15)
plt.show()

# find the empirical mean number of spikes over the 1000 neurons
print(np.mean(spike_train_matrix.sum(axis=1)))