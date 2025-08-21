import time
import random
import numpy as np
import matplotlib.pylab as plt

def random_play(N = False, sort = False):
    """Returns a choice of non-negative integers for coins at random
    The distribution is that of normalizing 9 random variables
    If sort = True, then players prioritize 9 coins
    if N != False, then N is the number of trials. Returns array (9, N), X[0] is the first selection"""
    if N != False:
        X = np.random.rand(9, N)
        X = np.int64(X*1000/np.sum(X, axis=0))      # normalized random variables
        X[0] += 1000-np.sum(X, axis=0)              # make them add 1000
        if sort:
            X = np.sort(X, axis=0)
        X = X.T
    else:
        X = np.random.rand(9)
        X = np.int64(X*1000/np.sum(X))      # normalized random variables
        X[0] += 1000-np.sum(X)              # make them add 1000
        if sort:
            X = np.sort(X)
    return X

def random_play_other_distribution():
    play = np.zeros(9)
    for k in range(9):
        play[-k-1] = random.randint(0, 1000-np.sum(play))
    return np.array(play)

optimal_choice = 111*np.ones(9, dtype=int); optimal_choice[-1] += 1     # this is what I want to optimize

def random_game(L = optimal_choice):
    X = random_play()
    gold = 0
    for i in range(9):
        if L[i] > X[i]:
            gold += i + 1
    return gold

def game_two_players(X, Y):
    """returns [coins of X, coins of Y]"""
    gold = [0, 0]
    for i in range(9):
        if X[i] > Y[i]:
            gold[0] += i+1
        elif Y[i] > X[i]:
            gold[1] += i+1
    return gold

def game_against_many(L = optimal_choice, N = 1000, sort = False, info = False):
    """X is an arrays with the chosen numbers
    returns the total gold gained by playing against X (my choice is optimal_choice)"""
    if info:
        total_gold = []
    else:
        total_gold = 0
    Y = random_play(N, sort)
    for j in range(N):
        X = Y[j]
        gold = 0
        for i in range(9):
            if L[i] > X[i]:
                gold += i + 1
        # total_gold += gold
        if info:
            total_gold.append(gold)
        else:
            total_gold += gold
    if info:
        return total_gold
    else:
        return total_gold/N

# problem:
# given a1+a2+...+an = 1000    and ai are non negative integers
# goal:
# maximize the expected value of game_random()

# simplification:
# maximize sum i*P(xi < ai)

# NOTES ON SPEED: 100,000 plays takes 0.5/1 seconds


# this is a function f: R^9 -> [0, 45)      and the domain is a cube lattice

def prob_sum(a, n = 8, N = 1000):
    """prob that X1+X2+...+Xn > a if they are in (0, 1)"""
    num_times = 0
    for _ in range(N):
        X = np.random.rand(n)
        if np.sum(X) > a:
            num_times += 1
    return num_times/N

# prob sum is a poly of degree 7 in each interval [0, 1], [1, 2], ... , [7, 8]

def write_file():
    """makes the file polys_for_pokerbots_coins.txt"""
    set_of_polys = []
    for k in range(8):
        """poly of deg 7 in [k, k+1]"""
        data_x = np.linspace(k, k+1, 100)
        poly = np.polyfit(data_x, [prob_sum(i) for i in data_x], 7)
        poly = list(poly)                                                             # used to find
        # poly.reverse()                                                            # sum_of_variables
        poly = str(poly)[1:-1]
        set_of_polys.append(poly+"\n")
    with open("polys_for_pokerbots_coins.txt", "w") as file:
        file.writelines(set_of_polys)

filee = open("polys_for_pokerbots_coins.txt", "r")
file = filee.readlines()
for i in range(len(file)):
    array = list(file[i].split(", "))
    array = [float(a) for a in array]
    file[i] = array
roots_of_file = [-51.0630994, 0.50748467, 1.69384505, 4.2656371, 2.63125354, 6.03480525, 6.98702217]

def sum_of_variables(x):
    """probability that 8 random variables in (0, 1) add up to at least x
    it can take np.ndarrays"""
    x = np.float64(x)
    if type(x) == np.float64:
        if x <= 0.1:
            return 1
        elif x >= 7.9:
            return 0
        k = int(np.floor(x))
        return np.polyval(file[k], x)
    val = 0
    k_of_list = np.int16(x)
    for k in range(7):
        values_for_poly_k = np.where(k_of_list == k, x, roots_of_file[k])
        val += np.polyval(file[k], values_for_poly_k)
    return val

# remember: optimal_choice is a list of 9 variables to optimize

# in the random case, the expected value is aprox sum of k*int ( sum_of_variables((1000/ak - 1)t) dt )

def expected_value(L, N = 100):
    """from R^9 to R
    Gives the value of the function at L=[a1, a2, ... , a9]"""
    def f(r, t):
        return sum_of_variables((-1+1000/L[r])*t)
    expected_value = 0
    for r in range(9):
        if L[r] == 0:
            continue
        N = 2*N+1
        A = (np.arange(N) % 2) * 2 + 2
        A[0], A[-1] = 1, 1              # simson method
        A = A / 3
        I = np.linspace(0, 1, N)                # interval
        X = f(r, I)                             # values of f           #### fix this
        for i in range(N):
            if X[i] == None:
                print(i)
        expected_value += (r+1)*np.sum(A*X/N)
    return expected_value

basis = [np.where(np.arange(9) == i, np.ones(9), 0) for i in range(9)]

def gradient_expected_value(L, h = 1, N = 100):
    grad = []
    for k in range(9):
        grad_k = (expected_value(L+h*basis[k], N)-expected_value(L-h*basis[k], N))/(2*h)
        grad.append(grad_k)
    return np.array(grad)

def test_expected_value(grad = False):
    """test if expected value works and the time it takes
    grad == True tests the gradient"""
    if not grad:
        for _ in range(10):
            optimal_choice = random_play()
            print(optimal_choice)
            print(game_against_many())
            t0 = time.perf_counter()
            print(expected_value(optimal_choice))
            t1 = time.perf_counter()
            print(f"time is {t1-t0}")
            print("")
    else:
        for _ in range(10):
            optimal_choice = random_play()
            t0 = time.perf_counter()
            print(gradient_expected_value(optimal_choice))
            t1 = time.perf_counter()
            print(f"time is {t1-t0}")
            print("")

optimal_choice = np.array([1.30779518e-01, 4.09499251e-02, 1.35796136e+00, 4.31737757e+01,
 1.76752483e+02, 1.86575433e+02, 1.92577017e+02, 1.97025112e+02,
 2.02366488e+02])
sent_choice = np.array([  0,   4,  11,  31, 133, 163, 203, 206, 249])            # sent solution

def gradient_descent(l, N = 5, h = 10):
    """N steps of size h in the direction closests to the gradient, starting at l"""
    L = np.float64(l)
    for _ in range(N):
        grad = gradient_expected_value(L)
    # map this to the plane sum() = 1000
    # vector is L + h*grad
        u = np.ones(9)/3        # unit vector perp to plane
        step_not_normaized = grad - np.dot(grad, u)*u
        new_L = L + step_not_normaized*h/np.linalg.norm(step_not_normaized)
        for k in range(9):
            if new_L[k] < 0:
                step_not_normaized = step_not_normaized - np.dot(step_not_normaized, basis[k])*basis[k]
                new_L = L + step_not_normaized*h/np.linalg.norm(step_not_normaized)
        L = new_L*1000/np.sum(new_L)
        print(L, "\n", expected_value(L))
        print("")
    return np.int16(L)
# optimal_choice = np.int16(gradient_descent(optimal_choice, 10, h=30))
# print(np.sum(optimal_choice))
# expected value maps R^9 in the plane x1+...+x9=1000 to R, the goal is to find the maximum value


# optimal_choice = np.zeros(9); optimal_choice[-1] = 1000
optimal_choice = np.array([  0.   ,        0.,           0.,           0.,         188.38927484,
 194.66335974, 200.82532651, 206.21010889, 209.91193001])
# print(gradient_descent(optimal_choice, N = 10, h = 1))

optimal_choice = np.array([  0 ,  0,   0,   0, 188, 194, 200, 206, 209])

def monte_carlo_simulation(sort = False, n = 1000, other = False, all = False):
    """simulate n random people"""
    if other and not all:
        numbers_selected = [random_play_other_distribution() for _ in range(n)]
    elif not other and not all:
        numbers_selected = random_play(n, sort).T
    else:
        numbers_selected1 = [random_play_other_distribution() for _ in range(n)]
        numbers_selected2 = list(random_play(n, False).T)
        numbers_selected3 = list(random_play(n, True).T)
        numbers_selected = numbers_selected1+numbers_selected2+numbers_selected3
    coins_gained = np.zeros(n)
    pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
    for pair in pairs:
        a, b = pair
        coins_a, coins_b = game_two_players(numbers_selected[a], numbers_selected[b])
        coins_gained[a] += coins_a
        coins_gained[b] += coins_b
    coins_gained = coins_gained/(n-1)
    coins_with_index = [(coins_gained[i], numbers_selected[i]) for i in range(n)]
    coins_with_index = sorted(coins_with_index, reverse=True, key=lambda x: x[0])
    for i in range(5):
        print(f"Number {i+1} won {coins_with_index[i][0]} with selection {coins_with_index[i][1]}")

# monte_carlo_simulation(n = 1000, all = True)
best_other_distribution = np.array([  6. ,  8.,   6.,  24., 111., 106., 122., 500., 116.])
best_all_distribtutions = np.array([ 24. , 14.,   8.,  20., 183., 125., 149., 293., 177.])

# print(f"Solution won {game_against_many(1000, False)} with selection {np.int16(optimal_choice)} against not sorted")
# print(f"Solution won {game_against_many(1000, True)} with selection {np.int16(optimal_choice)} against sorted")
# print("")

def distribution(sorted = False, N = 1000):
    """To know how many good players to expect"""
    plays = random_play(N, sorted)
    coins_gained = []
    for i in range(N):
        coins_gained.append(game_against_many(plays[i], sort=sorted))
    plt.hist(coins_gained, bins=50)
    plt.show()

def stats_of_choice(L, sorted = False, N = 1000, other = False):
    """mean and std of choosing L against N sorted or non sorted oponents"""
    # data = np.array(game_against_many(L, N, sorted, info=True))
    # print(f"Mean of {N} plays is {np.mean(data)} and std is {np.std(data)}")
    if other:
        data = []
        for _ in range(N):
            other_player = random_play_other_distribution()
            data.append(game_two_players(L, other_player))
        print(f"{L} against {N} of other distribution gets {np.mean(data)}")
    else:
        data = [game_against_many(L, 100, sorted) for _ in range(N)]
        print(f"{L} against 100 people gets {np.mean(data)} coins with std {np.std(data)}")

stats_of_choice(best_all_distribtutions, False)
stats_of_choice(best_all_distribtutions, False)
stats_of_choice(best_all_distribtutions, other = True)

filee.close()