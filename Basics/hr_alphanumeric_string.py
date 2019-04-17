if __name__ == '__main__':
    s = input()
    print(bool(len([i for i in s if i.isalnum()])))
    print(bool(len([i for i in s if i.isalpha()])))
    print(bool(len([i for i in s if i.isdigit()])))
    print(bool(len([i for i in s if i.islower()])))
    print(bool(len([i for i in s if i.isupper()])))
