def encrypt(m, k):
    s = 'abcdefghijklmnopqrstuvwxyz'
    n = ''
    for i in m:
        j = (s.find(i) + 26 + k) % 26
        n = n + s[j]
    return n

def decrypt(m, k):
    s = 'abcdefghijklmnopqrstuvwxyz'
    n = ''
    for i in m:
        j = (s.find(i) + 26 - k) % 26
        n = n + s[j]
    return n