import numpy as np
# import matplotlib.pylab as plt
# import time

# The idea of the document is to do number theory and groups and more


def linear_combination(a, b):
    """
    returns: np.ndarray [x, y] such that xa+yb = gcd(a, b)
    Doesn't work for np.ndarray"""
    sign_a, sign_b = np.sign(a), np.sign(b)
    a, b = np.abs(a), np.abs(b)
    combination = np.array([[1, 0], [0, 1]])        # stores how to get the current values of a and b
    while a != 0 and b != 0:
        if a > b:
            q, r = a//b, a % b
            a, combination[0] = r, combination[0]-q*combination[1]
        else:
            q, r = b//a, b % a
            b, combination[1] = r, combination[1]-q*combination[0]
    if a == 0:
        return np.array([sign_a*combination[1, 0], sign_b*combination[1, 1]])
    else:
        return np.array([sign_a*combination[0, 0], sign_b*combination[0, 1]])

def first_primes(n):
    """
    input: positive integer
    output: returns np.ndarray containing the primes smaller than n
    """
    P, Q = np.ones(n), []; P[0], P[1] = 0, 0
    for k in range(1, n):
        if P[k] == 1:
            P[2*k::k] = 0
            Q.append(k)
    return np.array(Q)

def mult_poly(P, Q):
    """
    input: two np.ndarray that represent polynomials
    The array [a0, a1, ...] represents a0+a1X+...
    output: returns np.ndarray of the product of polnomials
    converts ints to floats
    """
    # n, m = len(P), len(Q); r = n+m-1
    # X, Y = np.zeros(r), np.zeros(r); X[:n], Y[:m] = P, Q
    # return [np.add.reduce(X[:k+1]*Y[k::-1]) for k in range(r)]
    return np.flip(np.polymul(np.flip(P), np.flip(Q)))

def evaluate_poly(P, A):
    """
    input: an np.ndarray representing a polynomial and an np.ndarray of dimension 1
    output: np.ndarray with the evaluations of P at each entry
    """
    # return np.matmul(np.vander(A, len(P)), np.flip(P))
    return np.polyval(np.flip(P), A)

def char_poly(A):
    """
    returns: characteristic polynomial of the square matrix A
    Finds it by evaluating det(A-rI) for n+1 values of r, and then solving for the coefficients of the polynomial
    """
    # assert np.shape(A)[0] == np.shape(A)[1], f"The matrix {A} is not square"
    # n = len(A)+1
    # w = np.exp(2j*np.pi/n)
    # k = np.vander([w], n)[0]
    # B = np.vander(k, n)
    # y = np.linalg.det(np.array([A]*n)-np.multiply(np.array([np.eye(n-1)]*n),
    #                                 np.expand_dims(k, axis=[1, 2])))
    # return np.flip(np.linalg.solve(B, y))
    eig = np.linalg.eig(A)[0]
    return np.flip(np.poly(eig))

def prime_divisors(n):
    """
    input: positive integer
    output: returns dictionary mapping a prime divisor of n to its multiplicity
    time: O(k*sqrt(n))
    """
    assert type(n) == int and n >= 1, "Invalid input in prime_divisors"
    if n == 1:
        return np.array([])
    P = first_primes(int(np.floor(np.sqrt(n)))+2)
    m, A = n, {}
    while m > 1:
        k = 0
        while k < len(P) and m % P[k] != 0:
            k += 1
        if k == len(P):
            p, r = m, 1
            m = 1
        else:
            p, r = P[k], 0
            while m % p == 0:
                m = m//p
                r += 1
        A[p] = r
    return np.array(A)

def factor_int_poly(P):
    """returns: factorization in irreducible integer polynomials of P"""
    # print(f"P is {P}")
    if len(P) == 2:
        return [P]
    P = np.flip(P)
    roots = np.roots(P)
    coeff = np.gcd.reduce(P)
    degree = 1
    while degree <= len(P)//2+1:
        chosen = np.arange(degree)
        while True:
            # print(f"P is {P}, roots are {roots}, chosen is {chosen}")
            D = coeff*np.poly([roots[i] for i in chosen])
            i = 0
            while i < degree+1 and np.abs(D[i]-np.rint(np.real(D[i]))) < 0.00001:
                i += 1
            if i == degree+1:
                # print(D)
                D = np.round(D).astype(int)
                Q = np.polydiv(P, D)[0]
                Q = np.round(Q).astype(int)
                sol = factor_int_poly(D)
                sol.extend(factor_int_poly(Q))
                return sol
            j = degree - 1
            while chosen[j] == len(P)-2 + j - (degree - 1):
                j -= 1
            if j == -1:
                break
            chosen[j] += 1
            chosen[j+1:] = np.arange(chosen[j]+1, degree + chosen[j]-j)
        degree += 1
    return [np.flip(P)]

p = mult_poly(np.arange(2, 7), np.arange(1, 5))
print(p)
print(factor_int_poly(p))

def power_set(S):
    P = [[]]
    for s in S:
        for i in range(len(P)):
            P.append(P[i]+[s])
    return P
