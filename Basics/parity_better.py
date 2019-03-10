def parity(x):
    result = 0
    while x:
        result ^= 1
        # drop the lowest set bit
        x &= x - 1
    return result
