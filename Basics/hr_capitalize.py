import string

def solve(s):
    cap = ' '.join((i.capitalize() for i in s.strip().split(' ')))
    return cap

if __name__ == '__main__':
    s = input()
    print(solve(s))
