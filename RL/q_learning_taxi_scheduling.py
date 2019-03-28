# Step 1: Import dependencies
import numpy as np
import gym
import random

# Step 2: Create the environment
env = gym.make('Taxi-v2')
env.render()

# Step 3: Create the Q-table 
action_size = env.action_space.n
print('Action size ', action_size)

# Step 4: Initialize the Q-table 
state_size = env.observation_space.n
print('State size ', state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

# Step 5: Set hyperparameters
# Total episodes
total_episodes = 50000
# Total test episodes
total_test_episodes = 100
# Max steps per episode
max_steps = 99

# Learning rate
learning_rate = 0.7
# Discounting rate
gamma = 0.618

# Exploration parameters
# Exploration rate
epsilon = 1.0
# Exploration probability at start
max_epsilon = 1.0
# Min exploration probability
min_epsilon = 0.01
# Exponential decay rate for exploration probability
decay_rate = 0.01

# Step 6: Implement the Q-learning algorithm
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    for step in range(max_steps):
        # Choose an action (a) in the current world state (s)
        # Generate a random number
        exp_exp_tradeoff = random.uniform(0, 1)
        # If the  number > epsilon: exploit
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            # Random choice: explore
            action = env.action_space.sample()
        # Take the action (a) and observe outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * \
            (reward + gamma * np.max(qtable[new_state, :]) - \
             qtable[state, action])
        state = new_state
        if done == True:
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-decay_rate * episode)
            
# Step 7: Use the Q-table to play Taxi
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state, :])
        
        new_state, reward, done, info = env.step(action)
        
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            break
        state = new_state
        
env.close()
print('Score over time: ' + str(sum(rewards) / total_test_episodes))
