if __name__ == '__main__':
    n = int(input())
    student_grades = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_grades[name] = scores
    query_name = input()
    avg = sum(student_grades[query_name]) / 3
    print('%.2f' % avg)
