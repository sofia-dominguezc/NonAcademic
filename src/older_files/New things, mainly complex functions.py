import numpy as np
import matplotlib.pylab as plt

# -------------------------------------------------------------- Function g and importing from Complex_functions

# from Complex_functions import cfunction

def g(z):
    """Input: np.array
       Output: np.array of the same shape
       It calculates the function g componentwise.
    Properties of g:
    g(4z / (1+z)^2 ) = (1+z) g(z^2)            only inside the unit circle, this is the amg relation
    g(z)g'(1-z)+g(1-z)g'(z) = G / z(1-z)
    (z^2-z)g'' + (2z-1)g' + (1/4)g = 0
    g(z) = integral_0^1{ dt/sqrt((1-t^2)(1-zt^2)) }"""
    a, b = 1+0j, np.sqrt(1+0j-z)
    for _ in range(25):
        a, b = (a+b)/2, np.sqrt(a*b)
        b = b*np.sign(np.abs(a+b)-np.abs(a-b))
    return np.pi/(2*a)

# z = cfunction(lambda z: z, "z")
# g = cfunction(g, "g")
# dg = g.derivative()
# ddg = dg.derivative()

# w = (z**2 - z)*ddg + (2*z-1)*dg + 0.25*g

# ---------------------------------------------------- Plotting and debugging the NN for prediction of functions

def plot_connected_NN():
    N_plot = 50
    N_range_x = 2
    N_range_y = 2
    f_np = lambda z: z**3/(np.exp(z)-1)

    z_real = np.linspace(-N_range_x-1, N_range_x-1, N_plot).reshape(1, -1)
    z_imag = np.linspace(-N_range_y, N_range_y, N_plot).reshape(-1, 1)
    z = z_real + 1j*z_imag

    arrays_loaded = np.load("Numpy_arrays.npz")
    y_actual = arrays_loaded["y_actual"]
    y_pred = arrays_loaded["y_pred"]

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    ax.plot_wireframe(z_real, z_imag, y_actual[:,:,0], rstride=3, cstride=3)
    ax.plot_wireframe(z_real, z_imag, y_pred[:,:,0], rstride=3, cstride=3, color="C1")
    # ax.plot_wireframe(z_real, z_imag, np.real(2*f_np(z+0.5)-f_np(z)), rstride=3, cstride=3, color="C2")
    ax.set_title('Real part of the function')

    ax1.plot_wireframe(z_real, z_imag, y_actual[:,:,1], rstride=3, cstride=3)
    ax1.plot_wireframe(z_real, z_imag, y_pred[:,:,1], rstride=3, cstride=3, color="C1")
    # ax1.plot_wireframe(z_real, z_imag, np.imag(2*f_np(z+0.5)-f_np(z)), rstride=3, cstride=3, color="C2")
    ax1.set_title('Imaginary part of the function')

    plt.show()

# ------------------------------------------------------------------------------------ Plotting Riemman surfaces

def plot_surface1():
    N = 30
    z_real = np.linspace(0.5, 1.5, N).reshape((1, -1))
    z_imag = np.linspace(1, -1, N).reshape((-1, 1))
    z = z_real + 1j*z_imag
    f = lambda z: np.sqrt(z*(1-z))
    w1, w2 = f(z), -f(z)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    # triples_r = [(z_real[i], z_imag[j], w1[j, i]) for i in range(100) for j in range(100)] + [
    #                 (z_real[i], z_imag[j], w2[j, i]) for i in range(100) for j in range(100)]

    x_r = [z_real[0, i] for i in range(N) for j in range(N)] * 2
    y_r = [z_imag[j, 0] for i in range(N) for j in range(N)] * 2
    w_r = [np.real(w1[j, i]) for i in range(N) for j in range(N)] + [np.real(w2[j, i]) for i in range(N) for j in range(N)]

    ax.scatter(x_r, y_r, w_r)
    # ax.plot_surface(z_real, z_imag, np.real(w1), rstride=3, cstride=3)
    # ax.plot_surface(z_real, z_imag, np.real(w2), rstride=3, cstride=3, color='C0')
    ax.set_title('Real part')

    w_i = [np.imag(w1[j, i]) for i in range(N) for j in range(N)] + [np.imag(w2[j, i]) for i in range(N) for j in range(N)]


    # ax1.plot_surface(z_real, z_imag, np.imag(w1), rstride=3, cstride=3)
    # ax1.plot_surface(z_real, z_imag, np.imag(w2), rstride=3, cstride=3, color='C0')
    ax1.scatter(x_r, y_r, w_i)
    ax.set_title('Imaginary part')
    plt.show()

# ------------------------------------------------------------------------------------------- Leetcode debugging

class Solution(object):
    def getLengthOfOptimalCompression(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        def run_length_encoding(st):
            new_st = ''
            counter = 0
            for i, a in enumerate(st):
                if i == 0 or st[i-1] != a:
                    if counter > 1: new_st += str(counter)
                    new_st += a
                    counter = 1
                else:
                    counter += 1
            if counter > 1: new_st += str(counter)  # last step
            return new_st
        
        def is_one_or_power_ten_minus_one(N):
            if N == 1:
                return True
            N = N+1
            while N > 1:
                if N % 10 != 0:
                    return False
                N = N//10
            return True

        def recurse(i=0, k=k, d={}, t={}):
            """d maps (i, k) to the minimum length of s[i:] by 
               deleting at most k elements from the string
            t maps (i, k) to a list of possible starting strings"""
            if (i, k) in d:
                return d[(i, k)], t[(i, k)]
            elif k == 0:
                d[(i, k)] = len(run_length_encoding(s[i:]))
                j = i + 1
                while j < len(s) and s[j] == s[i]:
                    j += 1
                if j == i+1:
                    t[(i, k)] = [s[i]]
                else:
                    t[(i, k)] = [s[i] + str(j-i)]
                return d[(i, k)], t[(i, k)]
            elif i == len(s)-1:     # but k >=1, so delete it
                d[(i, k)] = 0
                t[(i, k)] = []
                return d[(i, k)], t[(i, k)]
            
            min1, start1 = recurse(i+1, k-1, d, t)     # delete s[i]
            min2, start2 = recurse(i+1, k, d, t)    # don't delete s[i]; use it

            new_start = {}  # start -> longer or not
            is_false = False    # true if at least one is not longer
            for p_start in start2:
                letter = p_start[0]
                if len(p_start) > 1:
                    number = int(p_start[1:])
                else:
                    number = 1
                if letter != s[i]:
                    new_start[s[i]] = True
                else:
                    new_start[s[i] + str(number+1)] = is_one_or_power_ten_minus_one(number)
                    if not is_one_or_power_ten_minus_one(number):
                        is_false = True
            if start2 == []:
                new_start[s[i]] = True

            start2 = [st for st in new_start if (new_start[st] != is_false)]
            if not is_false:
                # print(f"The minimum aumented from deleting in {s[i:]}")
                min2 += 1       # the minimum is one more
            
            # if i == 0: breakpoint()
            if min1 < min2:         # deleting is better
                # print(f"From {s[i:]} to {s[i+1:]} is better, k = {k-1}")
                d[(i, k)] = min1
                t[(i, k)] = start1
                return d[(i, k)], t[(i, k)]
            elif min1 > min2:       # not deleting is better
                # print(f"Staying in {s[i:]} is better, k = {k}")
                d[(i, k)] = min2
                t[(i, k)] = start2
                return d[(i, k)], t[(i, k)]
            else:
                # print("Edge case")
                new_start = set()
                for a in start1:
                    new_start.add(a)
                for a in start2:
                    new_start.add(a)
                d[(i, k)] = min1
                t[(i, k)] = list(new_start)
                return d[(i, k)], t[(i, k)]

        x = recurse()
        print(x)
        return x[0]
    
obj = Solution()
s = "babccacbbac"
k = 4
obj.getLengthOfOptimalCompression(s, k)