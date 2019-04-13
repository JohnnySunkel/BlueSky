if __name__ == '__main__':
    N = int(input())
    alist = []
    for i in range(N):
        x = input().split(' ')
        command = x[0]
        if command == 'append':
            alist.append(int(x[1]))
        if command == 'print':
            print(alist)
        if command == 'insert':
            alist.insert(int(x[1]), int(x[2]))
        if command == 'reverse':
            alist.reverse()
        if command == 'pop':
            alist.pop()
        if command == 'sort':
            alist.sort()
        if command == 'remove':
            alist.remove(int(x[1]))
