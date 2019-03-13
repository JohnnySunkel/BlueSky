# 8-puzzle solver

# import packages
from simpleai.search import astar, SearchProblem

# define a class that contains the methods to solve the puzzle
class PuzzleSolver(SearchProblem):
    # action method to get the list of possible
    # numbers that can be moved into the empty space
    def actions(self, cur_state):
        rows = string_to_list(cur_state)
        row_empty, col_empty = get_location(rows, 'e')
        
        # check the location of the empty space and 
        # create the new action
        actions = []
        if row_empty > 0:
            actions.append(rows[row_empty - 1][col_empty])
        if row_empty < 2:
            actions.append(rows[row_empty + 1][col_empty])
        if col_empty > 0:
            actions.append(rows[row_empty][col_empty - 1])
        if col_empty < 2:
            actions.append(rows[row_empty][col_empty + 1])
        
        return actions
        
    # return the resulting state after moving a piece to the
    # empty space
    def result(self, state, action):
        rows = string_to_list(state)
        row_empty, col_empty = get_location(rows, 'e')
        row_new, col_new = get_location(rows, action)
        
        rows[row_empty][col_empty], rows[row_new][col_new] = \
            rows[row_new][col_new], rows[row_empty][col_empty]

        return list_to_string(rows)
        
    # returns True if a state is the goal state
    def is_goal(self, state):
        return state == GOAL
        
    # returns an estimate of the distance from a state 
    # to the goal using manhattan distance
    def heuristic(self, state):
        rows = string_to_list(state)
        
        distance = 0
        
        for number in '12345678e':
            row_new, col_new = get_location(rows, number)
            row_new_goal, col_new_goal = goal_positions[number]

            distance += abs(row_new - row_new_goal) + abs(col_new - 
                col_new_goal)
            
        return distance
        
# define a function to convert a list to a string
def list_to_string(input_list):
    return '\n'.join(['-'.join(x) for x in input_list])
    
# define a function to convert a string to a list
def string_to_list(input_string):
    return [x.split('-') for x in input_string.split('\n')]
            
# define a function to find the 2D location of the input element
def get_location(rows, input_element):
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == input_element:
                return i, j
                
# final result that we want to achieve
GOAL = '''e-1-2
3-4-5
6-7-8'''

# starting point
INITIAL = '''6-1-2
3-4-5
e-7-8'''

# create a cache for the goal position of each piece
goal_positions = {}
rows_goal = string_to_list(GOAL)
for number in '12345678e':
    goal_positions[number] = get_location(rows_goal, number)

# create the solver object
result = astar(PuzzleSolver(INITIAL))

# print the results
for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('After moving', action, 'into the empty space. Goal achieved!')
    else:
        print('After moving', action, 'into the empty space')
        
    print(state)
