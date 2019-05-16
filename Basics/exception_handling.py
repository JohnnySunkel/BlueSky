for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a // b)
    except BaseException as err:
        print("Error Code:", err)
