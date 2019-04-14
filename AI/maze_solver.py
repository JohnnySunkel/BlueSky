# import the packages
import math
from simpleai.search import SearchProblem, astar

# create a class that contains the methods needed to solve the problem
class MazeSolver(SearchProblem):
    # initialize the class
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        
        # extract the initial and final positions
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)
                    
        super(MazeSolver, self).__init__(initial_state = self.initial)
        
    # define the method that takes actions
    # to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)
                
        return actions
        
    # update the state based on the action
    def result(self, state, action):
        x, y = state
        
        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1
        
        new_state = (x, y)
        
        return new_state
        
    # check if we have reached the goal
    def is_goal(self, state):
        return state == self.goal
        
    # compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # define the heuristic that we will use to arrive at the solution
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        
if __name__ == '__main__':
    # define the map
    MAP = """
    ##############################
    #         #              #   #
    # ####    ########       #   #
    #  o #    #              #   #
    #    ###     #####  ######   #
    #      #   ###   #           #
    #      #     #   #  #  #   ###
    #     #####    #    #  # x   #
    #              #       #     #
    ##############################
    """

    # convert the map to a list
    print(MAP)
    MAP = [list(x) for x in MAP.split("\n") if x]
    
    # define the cost of moving around the map
    cost_regular = 1.0
    cost_diagonal = 1.7
    
    # create the cost dictionary
    COSTS = {
             "up": cost_regular,
             "down": cost_regular,
             "left": cost_regular,
             "right": cost_regular,
             "up left": cost_diagonal,
             "up right": cost_diagonal,
             "down left": cost_diagonal,
             "down right": cost_diagonal,
    }
    
    # create the maze solver object
    problem = MazeSolver(MAP)
    
    # run the solver and extract the result
    result = astar(problem, graph_search = True)
    
    # extract the path from the result
    path = [x[1] for x in result.path()]
            
    # print the result
    print()
    for y in range(len(MAP)):
        for x in range(len(MAP[y])):
            if (x, y) == problem.initial:
                print('o', end = '')
            elif (x, y) == problem.goal:
                print('x', end = '')
            elif (x, y) in path:
                print('.', end = '')
            else:
                print(MAP[y][x], end = '')
        
        print()
