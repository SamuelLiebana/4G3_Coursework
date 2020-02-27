import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# useful plotting functions
def lin_interp(x, y, i, half):
    '''Linear interpolation to find half-maximum crossing coordinate.'''
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    '''Returns the x-coordinate of the two half-maximum crossings.'''
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

# constants 
max_K = 1000 # maximum K for plot of theoretical mean and variance
K_th = np.linspace(1, max_K, max_K) # K's used to plot theoretical mean and variance
w = 1 # synaptic weight
rate = 10 # rate of input poisson neurons
delta_t = 10**(-4) # length of timestep 
tau = 20*10**(-3) # time constant
sec = 15 # run for over 10 seconds (to compute mean and variance)
steps = int((1/delta_t)*sec) # number of steps in interval

# THEORETICAL EXPRESSIONS
# define theoretical expressions for mean and variance
mean_th = tau*w*rate
variance_th = (w**2 * rate * tau**2)*(1 - delta_t*rate)/((2*tau - delta_t)*K_th)

# EMPIRICAL OBSERVATIONS
# range of K's to evaluate mean and variance experimentally
K = np.array([1, 10, 100, 1000])
# evaluate variance at these values of K
variance_th_plot = (w**2 * rate * tau**2)*(1 - delta_t*rate)/((2*tau - delta_t)*K)
# initial conditions for voltage
voltage = np.zeros([len(K), steps])
# placeholder for the several runs of the computation
# of the mean and variance for each value of K
mean1 = []
mean10 = []
mean100 = []
mean1000 = []
var1= []
var10 = []
var100 = []
var1000 = []

# perform 50 simulation trials 
for p in range(50):
    n=0
    for k in K:
        # initialise input_spike_train to all 0's
        input_spike_train = np.zeros([k, steps])
        # populate the spike train matrix with k spike trains,
        # these are modelled by sampling 150000 times (15s) from a
        # bernoulli distribution with p = r_x*delta_t = 0.001
        for i in range(k):
            input_spike_train[i] = np.random.binomial(1, rate*delta_t, steps)
            np.random.seed()
        # implement 15s of the LIF neuron with k poisson inputs
        for i in range(1, steps):
            voltage[n, i] = voltage[n, i-1] + delta_t*(-voltage[n, i-1]/tau) + (w/k)*np.sum(input_spike_train[:,i-1], axis = 0)
        # increase counting variable
        n += 1

    # append the means and variances of the voltage
    # (removing the 100ms transient) for the different values of K 
    mean1.append(np.mean(voltage[0,1000:]))
    mean10.append(np.mean(voltage[1,1000:]))
    mean100.append(np.mean(voltage[2,1000:]))
    mean1000.append(np.mean(voltage[3,1000:]))
    var1.append(np.var(voltage[0,1000:]))
    var10.append(np.var(voltage[1,1000:]))
    var100.append(np.var(voltage[2,1000:]))
    var1000.append(np.var(voltage[3,1000:]))

# plot the theoretical variance vs. the empirical variance as a function of K
plt.figure(1)
plt.loglog(K_th, variance_th, label='Theoretical Prediction')
plt.errorbar(1, np.mean(var1), yerr = 2*np.std(var1), solid_capstyle='projecting', capsize=5, fmt='.g', label = 'K=1')
plt.errorbar(10, np.mean(var10), yerr = 2*np.std(var10), solid_capstyle='projecting', capsize=5, fmt='.b', label = 'K=10')
plt.errorbar(100, np.mean(var100), yerr = 2*np.std(var100), solid_capstyle='projecting', capsize=5, fmt='.m', label = 'K=100')
plt.errorbar(1000, np.mean(var1000), yerr = 2*np.std(var1000), solid_capstyle='projecting', capsize=5, fmt='.r', label = 'K=1000')
plt.xlabel('K (number of input neurons)', size = 15, labelpad = 10)
plt.ylabel('Variance (a.u.)', size = 15, labelpad = 10)
plt.legend()

# plot the theoretical mean vs. the empirical mean as a function of K
plt.figure(2)
plt.semilogx(K_th, [mean_th]*max_K, label='Theoretical Prediction')
plt.errorbar(1, np.mean(mean1), yerr = 2*np.std(mean1), solid_capstyle='projecting', capsize=5, fmt='.g', label = 'K=1')
plt.errorbar(10, np.mean(mean10), yerr = 2*np.std(mean10), solid_capstyle='projecting', capsize=5, fmt='.b', label = 'K=10')
plt.errorbar(100, np.mean(mean100), yerr = 2*np.std(mean100), solid_capstyle='projecting', capsize=5, fmt='.m', label = 'K=100')
plt.errorbar(1000, np.mean(mean1000), yerr = 2*np.std(mean1000), solid_capstyle='projecting', capsize=5, fmt='.r', label = 'K=1000')
plt.xlabel('K (number of input neurons)', size = 15, labelpad = 10)
plt.ylabel('Mean (a.u.)', size = 15, labelpad = 10)
plt.legend()

# plot the voltage dynamics with the theoretical and empirical mean 
# and standard deviation lines
fig, axs = plt.subplots(2, 2)

# K=1
# voltage plot
axs[0, 0].plot(voltage[0, :], color = 'k')
# theoretical mean and variance lines
axs[0, 0].plot([mean_th]*len(voltage[0, :]), color = 'g')
axs[0, 0].plot([mean_th + 2*np.sqrt(variance_th_plot[0])]*len(voltage[0, :]), color = 'm')
# experimental mean and variance lines
axs[0, 0].plot([mean1[-1]]*len(voltage[0, :]), '--', color = 'g')
axs[0, 0].plot([mean1[-1] + 2*np.sqrt(var1[-1])]*len(voltage[0, :]), '--', color = 'm')
axs[0, 0].set_title('K = 1', size=15)

# K = 10
# voltage plot
axs[0, 1].plot(voltage[1, :], color = 'k')
# theoretical mean and variance lines
axs[0, 1].plot([mean_th]*len(voltage[1, :]), color = 'g')
axs[0, 1].plot([mean_th + 2*np.sqrt(variance_th_plot[1])]*len(voltage[1, :]), color = 'm')
axs[0, 1].plot([mean_th - 2*np.sqrt(variance_th_plot[1])]*len(voltage[1, :]), color = 'm')
# experimental mean and variance lines
axs[0, 1].plot([mean10[-1]]*len(voltage[1, :]), '--', color = 'g')
axs[0, 1].plot([mean10[-1] + 2*np.sqrt(var10[-1])]*len(voltage[1, :]),'--', color = 'm')
axs[0, 1].plot([mean10[-1] - 2*np.sqrt(var10[-1])]*len(voltage[1, :]), '--', color = 'm')
axs[0, 1].set_title('K = 10', size=15)

# K = 100
# voltage plot
axs[1, 0].plot(voltage[2, :], color = 'k')
# theoretical mean and variance lines
axs[1, 0].plot([mean_th]*len(voltage[2, :]), color = 'g')
axs[1, 0].plot([mean_th + 2*np.sqrt(variance_th_plot[2])]*len(voltage[2, :]), color = 'm')
axs[1, 0].plot([mean_th - 2*np.sqrt(variance_th_plot[2])]*len(voltage[2, :]), color = 'm')
# experimental mean and variance lines
axs[1, 0].plot([mean100[-1]]*len(voltage[2, :]), '--', color = 'g')
axs[1, 0].plot([mean100[-1] + 2*np.sqrt(var100[-1])]*len(voltage[2, :]), '--', color = 'm')
axs[1, 0].plot([mean100[-1] - 2*np.sqrt(var100[-1])]*len(voltage[2, :]), '--', color = 'm')
axs[1, 0].set_title('K = 100', size = 15)
axs[1, 0].set_xticks(np.linspace(0,steps, 16), minor = False)
axs[1, 0].set_xticklabels(np.linspace(0,steps/10000, 16).astype(int), fontdict=None, minor=False)

# K = 1000
# voltage plot
axs[1, 1].plot(voltage[3, :], 'k')
# theoretical mean and variance plots
axs[1, 1].plot([mean_th]*len(voltage[3, :]), color = 'g', label='Theoretical Mean')
axs[1, 1].plot([mean_th + 2*np.sqrt(variance_th_plot[3])]*len(voltage[3, :]), color = 'm', label='Theoretical +/- 2*std')
axs[1, 1].plot([mean_th - 2*np.sqrt(variance_th_plot[3])]*len(voltage[3, :]), color = 'm')
# experimental mean and variance lines
axs[1, 1].plot([mean1000[-1]]*len(voltage[3, :]), '--' ,color = 'g', label ='Simulation Mean')
axs[1, 1].plot([mean1000[-1] + 2*np.sqrt(var1000[-1])]*len(voltage[3, :]),'--', color = 'm', label= 'Simulation +/- 2*std')
axs[1, 1].plot([mean1000[-1] - 2*np.sqrt(var1000[-1])]*len(voltage[3, :]),'--', color = 'm')
axs[1, 1].set_title('K = 1000', size = 15)
axs[1, 1].set_xticks(np.linspace(0,steps, 16), minor = False)
axs[1, 1].set_xticklabels(np.linspace(0,steps/10000, 16).astype(int), fontdict=None, minor=False)
axs[1, 1].legend()

# set the y-label
for ax in [axs.flat[0], axs.flat[2]]:
    ax.set_ylabel('Voltage (a.u.)', size = 12, labelpad =10)
# set the x-ticks and x-label
for ax in [axs.flat[0], axs.flat[1]]:
    ax.set_xticks([])
    ax.set_xticklabels([])
for ax in [axs.flat[2], axs.flat[3]]:
    ax.set_xlabel('Time (s)', size =12, labelpad = 10)

# plot theoretical vs. empirical voltage distributions
fig, axs = plt.subplots(2, 2)

# theoretical values of sigma
sigma_th = np.sqrt(variance_th_plot)

# K = 1
# theoretical mean and variance
theor_gauss = np.linspace(mean_th - 3*sigma_th[0], mean_th + 3*sigma_th[0], 100)
theor_gauss_pdf = stats.norm.pdf(theor_gauss, mean_th, sigma_th[0])
axs[0, 0].plot(theor_gauss, theor_gauss_pdf, label = 'Theoretical Prediction')
# experimental normalised histogram
axs[0, 0].hist(voltage[0, :], density = True, bins= 100, label = 'Simulation Values')
# plot mean position
axs[0, 0].stem([mean_th], [theor_gauss_pdf[50]], markerfmt = '')
# plot full width at half maximum
theor_hmx = half_max_x(theor_gauss,theor_gauss_pdf)
axs[0, 0].hlines(theor_gauss_pdf[50]/2, theor_hmx[0], theor_hmx[1], colors = 'tab:blue')
axs[0, 0].set_title('K = 1', size = 15)
axs[0, 0].legend()

# K = 10
# theoretical mean and variance
theor_gauss = np.linspace(mean_th - 3*sigma_th[1], mean_th + 3*sigma_th[1], 100)
theor_gauss_pdf = stats.norm.pdf(theor_gauss, mean_th, sigma_th[1])
axs[0, 1].plot(theor_gauss, theor_gauss_pdf, label = 'Theoretical Prediction')
# experimental normalised histogram
axs[0, 1].hist(voltage[1, :], density = True, bins= 100, label = 'Simulation Values')
# plot mean position
axs[0, 1].stem([mean_th], [theor_gauss_pdf[50]], markerfmt = '')
# plot full width at half maximum
theor_hmx = half_max_x(theor_gauss,theor_gauss_pdf)
axs[0, 1].hlines(theor_gauss_pdf[50]/2, theor_hmx[0], theor_hmx[1], colors = 'tab:blue')
axs[0, 1].set_title('K = 10', size = 15)


# K = 100
# theoretical mean and variance
theor_gauss = np.linspace(mean_th - 3*sigma_th[2], mean_th + 3*sigma_th[2], 100)
theor_gauss_pdf = stats.norm.pdf(theor_gauss, mean_th, sigma_th[2])
axs[1, 0].plot(theor_gauss, theor_gauss_pdf, label = 'Theoretical Prediction')
# experimental normalised histogram
axs[1, 0].hist(voltage[2, :], density = True, bins= 100, label = 'Simulation Values')
# plot mean position
axs[1, 0].stem([mean_th], [theor_gauss_pdf[50]], markerfmt = '')
# plot full width at half maximum
theor_hmx = half_max_x(theor_gauss,theor_gauss_pdf)
axs[1, 0].hlines(theor_gauss_pdf[50]/2, theor_hmx[0], theor_hmx[1], colors = 'tab:blue')
axs[1, 0].set_title('K = 100', size = 15)


# K = 1000
# theoretical mean and variance
theor_gauss = np.linspace(mean_th - 3*sigma_th[3], mean_th + 3*sigma_th[3], 100)
theor_gauss_pdf = stats.norm.pdf(theor_gauss, mean_th, sigma_th[3])
axs[1, 1].plot(theor_gauss, theor_gauss_pdf, label = 'Theoretical Prediction')
# experimental normalised histogram
axs[1, 1].hist(voltage[3, :], density = True, bins= 100, label = 'Simulation Values')
# plot mean position
axs[1, 1].stem([mean_th], [theor_gauss_pdf[50]], markerfmt = '')
# plot full width at half maximum
theor_hmx = half_max_x(theor_gauss,theor_gauss_pdf)
axs[1, 1].hlines(theor_gauss_pdf[50]/2, theor_hmx[0], theor_hmx[1], colors = 'tab:blue')
axs[1, 1].set_title('K = 1000', size = 15)


plt.show()


