import random

def roll(n):
    count = [0] * 6
    for i in range(n):
        roll = random.randint(1, 6)
        if roll == 1:
            count[0] += 1
        elif roll == 2:
            count[1] += 1
        elif roll == 3:
            count[2] += 1
        elif roll == 4:
            count[3] += 1
        elif roll == 5:
            count[4] += 1
        else:
            count[5] += 1
    return count
