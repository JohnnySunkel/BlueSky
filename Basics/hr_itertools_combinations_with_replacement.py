from itertools import combinations_with_replacement
a, b = input().split()
[print(''.join(i)) for i in combinations_with_replacement(sorted(a), int(b))]
