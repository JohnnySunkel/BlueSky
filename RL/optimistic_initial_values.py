# Optimistic Initial Values
import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_experiment_eps

# represents one bandit arm
class Bandit:
    def __init__(self, m):
        self.m = m # true mean
        self.mean = 10 # mean estimate = 10 instead of 0
        self.N = 0
        
    # simulates pulling the bandit's arm
    # is a Gaussian with unit variance
    def pull(self):
        return np.random.randn() + self.m

    # mean update equation
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

# N = number of times we play
# m1, m2, m3 = mean rewards for the 3 arms
# eps = epsilon for the epsilon greedy strategy
def run_experiment_oiv(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    
    data = np.empty(N)
    
    for i in range(N):
        # optimistic initial values
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    
    # plot moving average reward
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()
    
    for b in bandits:
        print(b.mean)
        
    return cumulative_average
    
if __name__ == '__main__':
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    oiv = run_experiment_oiv(1.0, 2.0, 3.0, 100000)
    
    # log scale plot
    plt.plot(c_1, label = 'eps = 0.1')
    plt.plot(oiv, label = 'optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear plot
    plt.plot(c_1, label = 'eps = 0.1')
    plt.plot(oiv, label = 'optimistic')
    plt.legend()
    plt.show()
