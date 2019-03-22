# Given n, return all prime numbers up to and including n.

def generate_primes(n):
    primes = []
    # is_prime[p] represents if p is prime or not. Initiallly, set
    # each p to True, excepting 0 and 1. Then use sieving to
    # eliminate non-primes.
    is_prime = [False, False] + [True] * (n - 1)
    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            # Sieve p's multiples.
            for i in range(p, n + 1, p):
                is_prime[i] = False
    return primes
