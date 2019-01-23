# Semi-Gradient SARSA

# Import packages and functions
import numpy as np
import matplotlib.pyplot as plt

from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict
from sarsa import random_action, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS

SA2IDX = {}
IDX = 0


# Create the model class
class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)
        
    def sa2x(self, s, a):
        # Note: using just (r, c, r * c, u, d, l, r, 1) is not
        # expressive enough
        return np.array([
            s[0] - 1                  if a == 'U' else 0,
            s[1] - 1.5                if a == 'U' else 0,
            (s[0] * s[1] - 3) / 3     if a == 'U' else 0,
            (s[0] * s[0] - 2) / 2     if a == 'U' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0,
            1                         if a == 'U' else 0,
            s[0] - 1                  if a == 'D' else 0,
            s[1] - 1.5                if a == 'D' else 0,
            (s[0] * s[1] - 3) / 3     if a == 'D' else 0,
            (s[0] * s[0] - 2) / 2     if a == 'D' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0,
            1                         if a == 'D' else 0,
            s[0] - 1                  if a == 'L' else 0,
            s[1] - 1.5                if a == 'L' else 0,
            (s[0] * s[1] - 3) / 3     if a == 'L' else 0,
            (s[0] * s[0] - 2) / 2     if a == 'L' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0,
            1                         if a == 'L' else 0,
            s[0] - 1                  if a == 'R' else 0,
            s[1] - 1.5                if a == 'R' else 0,
            (s[0] * s[1] - 3) / 3     if a == 'R' else 0,
            (s[0] * s[0] - 2) / 2     if a == 'R' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0,
            1                         if a == 'R' else 0,
            1
            ])
            
            # If we use SA2IDX, a one-hot encoding for every (s, a) pair:
            # In reality we would not want to do this because we have
            # just as many parameters as before.
            # x = np.zeros(len(self.theta))
            # idx = SA2IDX[s][a]
            # x[idx] = 1
            # return x
        
    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)
        
    def grad(self, s, a):
        return self.sa2x(s, a)
        
        
def getQs(model, s):
    # We need Q(s, a) to choose an action
    # i.e. a = argmax[a]{ Q(s, a) }
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs
    
    
# Define the main function
if __name__ == '__main__':
    # grid = standard_grid()
    grid = negative_grid(step_cost = -0.1)
    
    # print rewards
    print('rewards:')
    print_values(grid.rewards, grid)
    
    # No policy initialization, we will derive our policy
    # from most recent Q.
    # Enumerate all (s, a) pairs, each will have its own 
    # weight in our "dumb" model.
    # Essentially each weight will be a measure of Q(s, a) itself.
    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1
            
    # Initialize the model
    model = Model()
    
    # Repeat until convergence
    t = 1.0
    t2 = 1.0
    deltas = []
    for it in range(20000):
        if it % 100 == 0:
            t += 0.01
            t2 += 0.01
        if it % 1000 == 0:
            print('it:', it)
        alpha = ALPHA / t2
        
        # Instead of 'generating' an episode we will play 
        # an episode within this loop.
        s = (2, 0)  # start state
        grid.set_state(s)
        
        # Get Q(s) so we can choose the first action
        Qs = getQs(model, s)
        
        a = max_dict(Qs)[0]
        a = random_action(a, eps = 0.5 / t)  # epsilon greedy
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()
            
            # We need the next action as well since Q(s, a) depends
            # on Q(s', a').
            # If s2 not in policy then it's a terminal state, all
            # Q are 0.
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha * (r - model.predict(s, 
                    a) * model.grad(s, a))
            else:
                # Not terminal
                Qs2 = getQs(model, s2)
                a2 = max_dict(Qs2)[0]
                a2 = random_action(a2, eps = 0.5 / t)  # epsilon greedy
                
                # We will update Q(s, a) as we experience the episode
                model.theta += alpha * (r + GAMMA * model.predict(s2, 
                    a2) - model.predict(s, a)) * model.grad(s, a)
                
                # Next state becomes the current state
                s = s2
                a = a2
                
            biggest_change = max(biggest_change,
                                 np.abs(model.theta - old_theta).sum())
        deltas.append(biggest_change)
            
    plt.plot(deltas)
    plt.show()
    
    # Determine the policy from Q*
    # Find V* from Q*
    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q

    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)
