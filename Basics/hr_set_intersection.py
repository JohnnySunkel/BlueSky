if __name__ == '__main__':
    n1, s1 = input(), set(input().split())
    n2, s2 = input(), set(input().split())
    print(len(s1.intersection(s2)))
