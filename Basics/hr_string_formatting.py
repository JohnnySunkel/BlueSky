def print_formatted(number):
    w = len(format(n, 'b'))
    for i in range(1, n + 1):
        d = str(i).rjust(w)
        o = format(i, 'o').rjust(w)
        h = format(i, 'x').rjust(w).upper()
        b = format(i, 'b').rjust(w)
        print(d, o, h, b)

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
