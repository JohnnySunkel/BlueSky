import re
s, k = input(), input()
if not re.search(k, s):
    print('(-1, -1)')
else:
    i = 0
    while re.search(k, s[i:]):
        i += re.search(k, s[i:]).start() + 1
        print('(', i - 1, ', ', i + len(k) - 2, ')', sep = '')
