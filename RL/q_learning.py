# Q-Learning

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from td0_prediction import random_action

# Define constants
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# Define the main function
if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.1)
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
    # initialize Q(s, a)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0

    # keep track of how many times Q[s] has been updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    # repeat until convergence
    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3
        if it % 2000 == 0:
            print("it:", it)
            
        # instead of 'generating' an episode we will play an episode
        # within this loop
        s = (2, 0)  # start state
        grid.set_state(s)
        
        # The first (s, r) tuple is the starting state and 0
        # (we don't get a reward for simply starting the game). 
        # The last (s, r) tuple is the terminal state and the 
        # final reward. The value for the terminal state is 0 
        # by definition, so we don't care about updating it.
        a, _ = max_dict(Q[s])
        biggest_change = 0
        while not grid.game_over():
            a = random_action(a, eps = 0.5 / t)  # epsilon greedy
            # Random action also works, but slower since you can
            # bump into walls.
            # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            r = grid.move(a)
            s2 = grid.current_state()
            
            # adaptive learning rate
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            # we will update Q(s, a) as we experience the episode
            old_qsa = Q[s][a]
            # The difference between SARSA and Q-Learning is with
            # Q-Learning we will use this max[a']{ Q(s', a')} in our
            # update even if we do not end up taking this action
            # in the next step.
            a2, max_q_s2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change,
                                 np.abs(old_qsa - Q[s][a]))
            
            # we would like to know how often Q(s) has been updated too
            update_counts[s] = update_counts.get(s, 0) + 1

            # next state becomes the current state
            s = s2
            a = a2
            
        deltas.append(biggest_change)
        
    plt.plot(deltas)
    plt.show()
    
    # determine the policy from Q*
    # find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what proportion of time do we spend updating each part of Q?
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)
    
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
