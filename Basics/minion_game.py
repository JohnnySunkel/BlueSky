def minion_game(string):
    v = frozenset('AEIOU')
    l = len(s)
    k_score = sum(q for c, q in zip(string, range(l, 0, -1)) if c in v)
    s_score = l * (l + 1) // 2 - k_score
    if k_score > s_score:
        print('Kevin {:d}'.format(k_score))
    elif k_score < s_score:
        print('Stuart {:d}'.format(s_score))
    else:
        print('Draw')

if __name__ == '__main__':
    s = input()
    minion_game(s)
