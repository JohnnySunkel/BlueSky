# import packages and functions
import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

# define constants
SMALL_ENOUGH = 10e-4 # threshold for finding V(s)
GAMMA = 0.9 # discount factor
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all p(s', r | s, a) = 1 or 0


# main function
if __name__ == '__main__':
    grid = negative_grid()
    # the negative grid gives the agent a -0.1 reward for every
    # non-terminal state
    # we want to see if this will encourage the agent to find
    # a shorter path to the goal

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        
    # initial policy
    print("initial policy:")
    print_policy(policy, grid)
    
    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = 0
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    # repeat until converagence - will break out of the loop
    # when the policy does not change
    while True:
        
        # policy evaluation
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                # V(s) only has value if it's not a terminal state
                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                    
            if biggest_change < SMALL_ENOUGH:
                break
            
        # policy improvement
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                # loop through all possible actions to find the best
                # current action
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False
                    
        if is_policy_converged:
            break
        
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)