if __name__ == '__main__':
    n = int(input())
    arr = [[input(), float(input())] for i in range(0, n)]
    arr.sort(key = lambda x: (x[1], x[0]))
    names = [i[0] for i in arr]
    grades = [i[1] for i in arr]
    min_val = min(grades)
    while grades[0] == min_val:
        grades.remove(grades[0])
        names.remove(names[0])
    for i in range(0, len(grades)):
        if grades[i] == min(grades):
            print(names[i])
