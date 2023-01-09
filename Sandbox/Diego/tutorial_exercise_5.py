import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# hyper-parameters
maxT = 1500           # time units
no_trials = 3500    # number of trials
lr = 0.005            # learning rate of nodes
alpha = 0.005        # learning rate of network parameters
mean = 5            # mean of normal distribution of this layer
std = math.sqrt(2)        # variance of normal distribution of this layer

# variables:
sigma = np.zeros(no_trials)  # variance to learn
sigma[0] = 1    # initialize with 1
eps = 0     # initialize node eps
e = 0       # initialize node e
g_of_upper_phi = mean   # activations from upper layer match mean of this layer

# start simulation of trials
for i in tqdm(range(1, no_trials)):
    phi = np.random.normal(mean, std, 1)  # generate features from this layer, normally distributed
    for j in range(0,maxT):
        # compute activity change
        eps_new = phi - g_of_upper_phi - e
        e_new = sigma[i-1] * eps - e

        # update neural activities
        eps = eps + lr * eps_new
        e = e + lr * e_new

    # compute weight change
    d_sigma = alpha * (eps * e - 1)
    # update the network weights
    sigma[i] = sigma[i-1] + d_sigma

plt.plot(range(0, no_trials), sigma, label="sigma")
plt.xlabel("Trial")
plt.ylabel("Sigma")
plt.legend()
plt.show()
print(np.mean(sigma))


