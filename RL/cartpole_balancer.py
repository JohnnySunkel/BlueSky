# Build a learning agent in OpenAI Gym

# import packages
import argparse
import gym

# define a function to parse the input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Run an environment')
    parser.add_argument('--input-env',
                        dest = 'input_env',
                        required = True,
                        choices = ['cartpole',
                                   'mountaincar',
                                   'pendulum'],
                        help = 'Specify the name of the environment')
    return parser
    
    
# define the main function and parse the input arguments
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_env = args.input_env
    
    # build a mapping from the input arguments to the names
    # of the environments in the OpenAI Gym package
    name_map = {'cartpole': 'CartPole-v0',
                'mountaincar': 'MountainCar-v0',
                'pendulum': 'Pendulum-v0'}
                
    # create the environment based on the input argument
    env = gym.make(name_map[input_env])
    
    # start iterating
    for _ in range(20):
        # reset the environment
        observation = env.reset()
        
        # for each reset iterate 100 times
        for i in range(100):
            # render the environment
            env.render()
            
            # print the current observation
            print(observation)
            
            # take action
            action = env.action_space.sample()
            
            # extract the observation, reward, status, and 
            # other information based on the action taken
            observation, reward, done, info = env.step(action)
            
            # check if the agent achieved the goal
            if done:
                print('Episode finished after {} timesteps'.format(i + 1))
                break
