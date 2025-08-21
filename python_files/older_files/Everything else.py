import matplotlib.pyplot as plt
import numpy as np
import time
# from Complex_functions import cfunction
# remember: bulgarian solitaire


# you can do this
# sofia, I believe it


# I didn't remember it was this bad
# I'm sorry girl, but you feel better now
# and it will only get better in the future


# It's so much better now. I'm so proud of you girl. You made it.
# I would've never imagined it would be this nice, but it happened.




# FRACTION AND RUBIK 2X2 CLASSES


def mcd(a, b):
    """
    input: to ints
    output: returns the mcd of them"""
    a, b = abs(a), abs(b)
    if a == b:
        return a
    while a != 0 and b != 0:
        if a > b:
            a = a % b
        else:
            b = b % a
    return a+b
class fraction:
    def __init__(self, a, b):
        assert b != 0, "Dont divide by zero yet"
        d = mcd(a, b)
        if b < 0:
            a, b = -a, -b
        self.n = a//d      #numerator
        self.d = b//d      #denominator
    def __eq__(self, other):
        if isinstance(other, fraction):
            return self.n*other.d == other.n*self.d
        elif type(other) == int:
            return self.n == other*self.d
    def __add__(self, other):
        if isinstance(other, fraction):
            return fraction(self.n*other.d+self.d*other.n, self.d*other.d)
        elif type(other) == int:
            return fraction(self.n+self.d*other, self.d)
    def __sub__(self, other):
        if isinstance(other, fraction):
            return fraction(self.n*other.d-self.d*other.n, self.d*other.d)
        elif type(other) == int:
            return fraction(self.n-other*self.d, self.d)
    def __mul__(self, other):
        if isinstance(other, fraction):
            return fraction(self.n*other.n, self.d*other.d)
        elif type(other) == int:
            return fraction(self.n*other, self.d)
    def __truediv__(self, other):
        if isinstance(other, fraction):
            assert other.n != 0, "Don't divide by zero yet"
            return fraction(self.n*other.d, self.d*other.n)
        elif type(other) == int:
            assert other != 0, "Don't divide by zero yet"
            return fraction(self.n, self.d*other)
    def __rtruediv__(self, other):
        if isinstance(other, fraction):
            assert other.n != 0, "Don't divide by zero yet"
            return fraction(self.d*other.n, self.n*other.d)
        elif type(other) == int:
            assert other != 0, "Don't divide by zero yet"
            return fraction(self.d*other, self.n*other)
    def __lt__(self, other):
        if isinstance(other, fraction):
            return self.n*other.d < other.n*self.d
        elif type(other) == int:
            return self.n < other*self.d
    def __rt__(self, other):
        if isinstance(other, fraction):
            return self.n*other.d > other.n*self.d
        elif type(other) == int:
            return self.n > other*self.d
    def __str__(self):
        if self.d == 1:
            return f"{self.n}"
        else:
            return f"{self.n}/{self.d}"

class Rubik2():
    """instances of the class are the elements of the group"""

    turns = ["R", "U", "F", "R'", "U'", "F'", "R2", "U2", "F2"]

    def __init__(self, perms = [i for i in range(7)], rots = [0 for _ in range(7)]):
        """the vertex 0 is ulf, 1 is urf, 2 is urb, 3 is ulb, 4 is dlf, 5 is drf, 6 is drb
           orientation 0 means white/yellow is on top/bottom, 1 means white/yellow in front/back, 2 in right/left
           perms is a list s.t. the vertex i is in the position of the vertex perms[i]
           rots is a list s.t. rots[i] is either 0, 1, 2"""
        assert type(perms) == list and len(perms) == 7, "invalid set of permutations for new cube state"
        assert type(rots) == list and len(rots) == 7, "invalid set of rotations for new cube state"
        self.p = perms.copy()
        self.r = rots.copy()

    def move(self, movement):
        """movements is a list of ints in [0, 8]
           or a string with the scramble in the form: R U' F2 R' ..."""
        if type(movement) == str:
            movement = movement.split()
            for mov in movement:
                if mov == "R":
                    self.move(0)
                elif mov == "R2":
                    self.move(6)
                elif mov == "R'":
                    self.move(3)
                elif mov == "U":
                    self.move(1)
                elif mov == "U2":
                    self.move(7)
                elif mov == "U'":
                    self.move(4)
                elif mov == "F":
                    self.move(2)
                elif mov == "F2":
                    self.move(8)
                elif mov == "F'":
                    self.move(5)
                else:
                    print(f"{mov} is not valid in the scramble")
                    raise NameError
            return None
        # 1 -> 2 -> 6 -> 5
        if movement == 0:   #R
            change = {1:2, 2:6, 6:5, 5:1, 3:3, 0:0, 4:4}
            change_or = [(1, 2, 6, 5), {0:1, 1:0, 2:2}]
        elif movement == 6: #R2
            change = {1:6, 6:1, 2:5, 5:2, 3:3, 0:0, 4:4}
            change_or = [(), {}]
        elif movement == 3: #R'
            change = {1:5, 5:6, 6:2, 2:1, 3:3, 0:0, 4:4}
            change_or = [(1, 2, 6, 5), {0:1, 1:0, 2:2}]
        # 0 -> 1 -> 2 -> 3
        elif movement == 1: #U
            change = {0:3, 3:2, 2:1, 1:0, 4:4, 5:5, 6:6}
            change_or = [(0, 1, 2, 3), {0:0, 1:2, 2:1}]
        elif movement == 7:
            change = {0:2, 2:0, 1:3, 3:1, 4:4, 5:5, 6:6}
            change_or = [(), {}]
        elif movement == 4:
            change = {0:1, 1:2, 2:3, 3:0, 4:4, 5:5, 6:6}
            change_or = [(0, 1, 2, 3), {0:0, 1:2, 2:1}]
        # 0 -> 1 -> 5 -> 4
        elif movement == 2: #F
            change = {0:1, 1:5, 5:4, 4:0, 2:2, 3:3, 6:6}
            change_or = [(0, 1, 5, 4), {0:2, 1:1, 2:0}]
        elif movement == 8:
            change = {0:5, 5:0, 1:4, 4:1, 2:2, 3:3, 6:6}
            change_or = [(), {}]
        elif movement == 5:
            change = {0:4, 4:5, 5:1, 1:0, 2:2, 3:3, 6:6}
            change_or = [(0, 1, 5, 4), {0:2, 1:1, 2:0}]
        for i in change_or[0]:
            j = self.p.index(i)
            self.r[j] = change_or[1][self.r[j]]
        self.p = [change[self.p[i]] for i in range(7)]

    def __eq__(self, other):
        return self.p == other.p and self.r == other.r
    def __str__(self):
        return str(self.p)+", "+str(self.r)
    def __hash__(self):
        return hash(str(self))


def solve(cube):
    """cube is an instance of Rubik2
       returns a sequence of moves that solves it"""
    def inv_turns(movements):
        """maps R to R', R' to R, R2 to R2 and similarly for the other letters in the format of ints from 0 to 8"""
        if movements < 6:
            return (movements + 3) % 6
        else:
            return movements

    previous_moves = []
    current_cube = Rubik2()
    cube_moves_from_solved = {current_cube.__hash__(): []}
    def dfs(i):
        if i == 5:
            return None
        for j in [k for k in range(9) if i == 0 or (k-previous_moves[-1]) % 3 != 0]:
            previous_moves.append(j)
            current_cube.move(j)
            cube_moves_from_solved[current_cube.__hash__()] = previous_moves.copy()
            dfs(i+1)
            current_cube.move(inv_turns(j))
            previous_moves.pop()
    dfs(0)

    if cube.__hash__() in cube_moves_from_solved:
        solution = ""
        solved_to_mid = reversed(cube_moves_from_solved[cube.__hash__()])
        for r, k in enumerate(solved_to_mid):
            if r != 0:
                solution += ' '
            solution += Rubik2.turns[inv_turns(k)]
        print(f"A solution is {solution}")
        return None
    done = False
    def second_dfs(i):
        nonlocal done
        if i == 6 or done:
            return None
        for j in [k for k in range(9) if i == 0 or (k-previous_moves[-1]) % 3 != 0]:
            previous_moves.append(j)
            cube.move(j)
            if cube.__hash__() in cube_moves_from_solved:
                # print(f"From unsolved to mid is: {[Rubik2.turns[k] for k in previous_moves]}")
                # print(f"From solved to mid is: {[Rubik2.turns[k] for k in cube_moves_from_solved[cube.__hash__()]]}")
                solution = ''
                for r, k in enumerate(previous_moves):
                    if r != 0:
                        solution += " "
                    solution += Rubik2.turns[k]
                solved_to_mid = reversed(cube_moves_from_solved[cube.__hash__()])
                for k in solved_to_mid:
                    solution += ' '+Rubik2.turns[inv_turns(k)]
                print(f"A solution is: {solution}")
                done = True
            second_dfs(i+1)
            cube.move(inv_turns(j))
            previous_moves.pop()
    second_dfs(0)

# import time

# cube = Rubik2()
# cube.move("R F' U R' U2 R' U R F' U F2")
# t0 = time.perf_counter()
# solve(cube)
# t1 = time.perf_counter()
# print(f"The solution was found in {t1-t0} seconds")





# problem 15 of number theory putnam seminar

def check(a, b, c, N = 100):
    """
    cheks if fa = fb * fc mod 2 up to coefficient N"""
    def mult_int_poly_mod_2(P, Q, N = 100):
        n, m = len(P), len(Q); r = n+m-1
        X, Y = np.zeros(r), np.zeros(r); X[:n], Y[:m] = P, Q
        return np.array([int(np.round(np.add.reduce(X[:k+1]*Y[k::-1]))) % 2 for k in range(N)])
    def q(a, N = 100):
        """
        sum of X^k for k geq 0 such that ak+1 is a square up to X^N"""
        ans = []
        x = 1
        for k in range(N):
            while x**2 < a*k+1:
                x += 1
            if x**2 == a*k+1:
                ans.append(1)
            else:
                ans.append(0)
        return ans
    poly = (q(a, N)-mult_int_poly_mod_2(q(b, N), q(c, N), N)) % 2
    if 1 in poly:
        return False
    else:
        return True

def print_poly_mod_2(P):
    string = ''
    first_is_done = False
    for n in range(len(P)):
        if P[n] == 1:
            if not first_is_done:
                string += f"X^{n}"
                first_is_done = True
            else:
                string += f" + X^{n}"
    print(string)
triples = [(4, 6, 12), (8, 12, 24), (16, 24, 48), (6, 8, 24),
           (10, 12, 60), (20, 24, 120), (21, 24, 168), (15, 24, 40)]        # the known solutions with b != c

# With Liza, we proved that 1/a = 1/b + 1/c, and that, defining b = b1 d, c = c1 d, the sum b1 + c1 is in {2, 3, 4, 6, 8, 12, 24}

possibles_b1_c1 = [(1, 2), (1, 3), (1, 5), (1, 7), (3, 5), (1, 11), (5, 7), (1, 23), (5, 19), (7, 17), (11, 13)]     #without (1, 1)

# Also, d = (b1 + c1) k, and a = b1 c1 k, b = b1 (b1 + c1) k, c = c1 (b1 + c1) k

# M = 100        # test up to a = M  (aproximately)
# t0 = time.perf_counter()
# possible_triples = {}
# group = []
# for pair in possibles_b1_c1:
#     possible_triples[pair] = []
# for pair in possibles_b1_c1:
#     b1, c1 = pair
#     for k in range(1, 22):
#         a, b, c = b1*c1*k, b1*(b1+c1)*k, c1*(b1+c1)*k
#         if check(a, b, c, 500):
#             # possible_triples[pair].append((a, b, c))
#             # print(b1+c1)
#             if pair not in group:
#                 group.append(pair)
# t1 = time.perf_counter()
# # print(possible_triples)
# print(group)

def is_square(x):
    if x < 0:
        return False
    y = int(np.round(np.sqrt(x)))
    return y**2 == x

def solutions_to_quadratic(x, a, b):
    """
    returns the positive solutions y,z to the equation x = ay^2 + bz^2"""
    z = 1
    solutions = []
    while b*z**2 < x:
        yy = (x-b*z**2)/a
        if is_square(yy):
            y = int(np.sqrt(yy))
            assert y**2 == yy, "Aproxmation error"

            if y % 2 == 0:
                z += 1              # ##
                continue

            solutions.append((y, z))
        z += 1
    return solutions

# from Algebra_number_theory import prime_divisors
# for q in range(1, 101):
#     if q % 3 == 0:
#         continue
#     x1 = solutions_to_quadratic(4*q, 3, 1)
#     # y1 = solutions_to_quadratic(4*q**2, 3, 1)
#     if len(x1) % 2 == 0 and len(x1) != 0:
#         print(f"x={q} gives {x1}")
    # if len(y1) == 1:
    #     print(f"x={q}^2 gives {y1}")

def check_if_k_works(k, N = 10000):
    assert k != 1, "k is not 1"
    x = 1
    # while (len(solutions_to_quadratic(4*x, 3, 1)) % 2) == (is_square(x) % 2):
    #     x += 6*k
    #     if x > N:
    #         return True
    while True:
        solutions = solutions_to_quadratic(4*x, 3, 1)
        actual_solutions = 0
        for sol in solutions:
            if (sol[1]**2 - 1) % (2*k) == 0:
                actual_solutions += 1
        if actual_solutions % 2 != is_square(x) % 2:
            break
        x += 6*k
        if x > N:
            return True
    return x

# for k in range(3, 100, 2):
#     print(f"k = {k} doesn't work for x = {check_if_k_works(k)}")

# print(check(6*3, 8*3, 24*3, 2))
def q(a, N = 100):
    """
    sum of X^k for k geq 0 such that ak+1 is a square up to X^N"""
    ans = []
    x = 1
    for k in range(N):
        while x**2 < a*k+1:
            x += 1
        if x**2 == a*k+1:
            ans.append(1)
        else:
            ans.append(0)
    return ans

def lowest_term(a, N = 100):
    fq = q(a, N)
    x = 1
    while fq[x] == 0 and x < N-1:
        x += 1
    return x

# for k in range(1, 201, 2):
#     # print(f"lowest term in 6*{k}: {lowest_term(6*k)}")
#     # print(f"lowest term in 8*{k}: {lowest_term(8*k)}")
#     # print(f"lowest term in 24*{k}: {lowest_term(24*k)}")
#     if check(6*k, 8*k, 24*k, 1+min(lowest_term(6*k), lowest_term(8*k), lowest_term(24*k))):
#         print(k)

# k = 15
# print(check(6*k, 8*k, 24*k, 1+min(lowest_term(6*k), lowest_term(8*k), lowest_term(24*k))))
# print_poly_mod_2(q(6*k, 30))
# print_poly_mod_2(q(8*k, 30))
# print_poly_mod_2(q(24*k, 30))

N = -2
U = 2+np.sqrt(3)
bound_x = np.sqrt(-N)*(np.sqrt(U)+1)/2
# print(bound_x)

from math import *
# newton root finding
def newton(f, df, x0, epsilon=0.001):
    while abs(f(x0)) > epsilon:
        x0 = x0 - f(x0) / df(x0)
    return x0

# define f and df
f = lambda x, R0, pii: x + (1 - pii) * (exp(-R0 * x) - 1)
df = lambda x, R0, pii: 1 - (1 - pii) * R0 * exp(-R0 * x)

# g calculates the solution to f(x; R0, pii) = 0
g = lambda R0, pii: newton(lambda x: f(x, R0, pii), lambda x: df(x, R0, pii), x0=1)

# start the plot
X = np.linspace(0.1, 10, 100)

# case 1: k = 0.75, pii = 0.4
Y1 = [g(0.75 * R0, pii=0.4) for R0 in X]

# case 2: k = 1, pii = 0.5
Y2 = [g(R0, pii=0.5) for R0 in X]

# find the intersection
i = 24 + np.argmin(np.abs(np.array(Y1) - np.array(Y2))[25:])
print(f"The curves intersect at R0 = {X[i]:.2f}")

plt.plot(X, Y1, label="Case 1: increase testing.")
plt.plot(X, Y2, label="Case 2: increase vaccination.")
plt.legend(loc='best')
plt.xlabel("R0")
plt.ylabel("R_inf - pi")
plt.show()