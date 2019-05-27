n, x = map(int, input().split())
mark_sheet = []
for _ in range(x):
    mark_sheet.append(map(float, input().split()))
for i in zip(*mark_sheet):
    print(sum(i) / len(i))
