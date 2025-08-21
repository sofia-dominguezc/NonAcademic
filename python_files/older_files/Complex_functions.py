import matplotlib.pyplot as plt
import numpy as np

class cfunction:
    """
    The class of functions from C to C"""
    def __init__(self, function, representation = None):
        self.f = function
        if not representation:
            self.r = ""
        else:
            # code to remove extra parenthesis in operations
            left_p, right_p = [], {}            # the pair left[i], right[i] is the location of the parhentesis i
            for i in range(len(representation)):
                if representation[i] == "(":
                    left_p.append(i)
                elif representation[i] == ")":
                    j = len(left_p)-1
                    while j in right_p and j > 0:
                        j -= 1
                    right_p[j] = i
            locations_to_remove = []
            for k in range(len(left_p)-1):
                if left_p[k+1] == left_p[k] + 1 and right_p[k+1] == right_p[k] -1:
                    locations_to_remove.extend([left_p[k], right_p[k]])
            locations_to_remove.sort()
            new_representation = ""
            for i in range(len(locations_to_remove)):
                if i == 0:
                    a = 0
                else:
                    a = locations_to_remove[i-1]+1
                new_representation += representation[a : locations_to_remove[i]]
            self.r = new_representation

    def __str__(self):
        return self.r

    def __add__(self, other):                                               # addition and substraction of functions
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) + other, f"{self.r} + {other}")
        return cfunction(lambda z: self.f(z)+other.f(z), f"{self.r} + {other.r}")
    def __radd__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) + other, f"{other} + {self.r}")
        return cfunction(lambda z: self.f(z)+other.f(z), f"{other.r} + {self.r}")
    def __iadd__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) + other, f"{self.r} + {other}")
        return cfunction(lambda z: self.f(z)+other.f(z), f"{self.r} + {other.r}")
    def __sub__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) - other, f"{self.r} - {other}")
        return cfunction(lambda z: self.f(z)-other.f(z), f"{self.r} - {other.r}")
    def __rsub__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: other - self.f(z), f"{other} - {self.r}")
        return cfunction(lambda z: other.f(z)-self.f(z), f"{other.r} - {self.r}")
    def __isub__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) - other, f"{self.r} - {other}")
        return cfunction(lambda z: self.f(z)-other.f(z), f"{self.r} - {other.r}")
    def __neg__(self):
        return cfunction(lambda z: -self.f(z), f"-{self.r}")
    def __pos__(self):
        return self

    def __mul__(self, other):                                               # multiplication and division of functions
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) * other, f"({self.r})*{other}")
        return cfunction(lambda z: self.f(z)*other.f(z), f"({self.r}) * ({other.r})")
    def __rmul__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) * other, f"{other}*({self.r})")
        return cfunction(lambda z: self.f(z)*other.f(z), f"({other.r}) * ({self.r})")
    def __imul__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) * other, f"({self.r})*{other}")
        return cfunction(lambda z: self.f(z)*other.f(z), f"({self.r}) * ({other.r})")
    def __pow__(self, n):
        if n >= 0:
            if len(self.r) == 1:
                return cfunction(lambda z: self.f(z)**n, f"{self}^{n}")
            return cfunction(lambda z: self.f(z)**n, f"({self})^{n}")
        else:
            if len(self.r) == 1:
                return cfunction(lambda z: self.f(z)**n, f"1/{self}^{-n}")
            return cfunction(lambda z: self.f(z)**n, f"1/({self})^{-n}")
    def __truediv__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: self.f(z) / other, f"({self.r})/{other}")
        return cfunction(lambda z: self.f(z)/other.f(z), f"({self.r}) / ({other.r})")
    def __rtruediv__(self, other):
        if not isinstance(other, cfunction):
            return cfunction(lambda z: other / self.f(z), f"{other}/({self.r})")
        return cfunction(lambda z: other.f(z) / self.f(z), f"({other.r}) / ({self.r})")

    def __matmul__(self, other):                                            # composition and evaluation of functions
        if not isinstance(other, cfunction):
            return self.f(other)
        new_rep = self.r
        for i in range(len(self.r)-1, -1, -1):
            if new_rep[i] == "z":
                new_rep = new_rep[:i]+"("+other.r+")"+new_rep[i+1:]
        return cfunction(lambda z: self.f(other.f(z)), new_rep)
    def __rmatmul__(other, self):
        if not isinstance(other, cfunction):
            return self.f(other)
        new_rep = self.r
        for i in range(len(self.r)-1, -1, -1):
            if new_rep[i] == "z":
                new_rep = new_rep[:i]+"("+other.r+")"+new_rep[i+1:]
        return cfunction(lambda z: self.f(other.f(z)), new_rep)
    def __imatmul__(self, other):
        """ a @= b changes  a  to  a @ b  or  a(b)"""
        if not isinstance(other, cfunction):
            return other
        new_rep = self.r
        for i in range(len(self.r)-1, -1, -1):
            if new_rep[i] == "z":
                new_rep = new_rep[:i]+"("+other.r+")"+new_rep[i+1:]
        return cfunction(lambda z: other.f(self.f(z)), new_rep)
    def __call__(self, other):
        return self @ other
    def inverse(self, a = 0.5+0j):
        """ a function g such that, for every z, outputs a value g(z) such that f(g(z))=z
        it starts with the point (a, f(a)) and gets closer to the root with small steps, then uses the newton method"""
        df = self.derivative().f
        f = self.f
        def inv_func(z, a):
            y = a
            for _ in range(20):
                w = (z-f(y))/df(y)
                w = w/np.abs(w)
                y += 0.4*w
            for _ in range(15):
                y += (z-f(y))/df(y)
            return y
        return cfunction(lambda z: inv_func(z, a), f"inv({self.r})")

    def derivative(self, h = 0.001):                                        # derivatives, integrals, and graphs
        """
        returns the derivative of self"""
        return cfunction(lambda z: (self.f(z+h)-self.f(z-h)
                                       -1j*self.f(z+h*1j)+1j*self.f(z-h*1j))/(4*h), f"D({self.r})")
    def integral(self, a = 0, b = 1, N = 500):
        """
        input: function f:C -> np.ndarray
        Gives the same weight to all parts of the interval and doesn't consider the points a and b
        output: returns a number, the value of it's integral from a to b"""
        N = 2*N+1
        A = (np.arange(N) % 2) * 2 + 2
        A[0], A[-1] = 1, 1              # simson method
        A = A / 3 + 0j
        I = np.linspace(a+1/N, b-1/N, N, dtype=complex)
        try:
            x = self.f(I)
        except:
            x = np.array([self.f(i) for i in I])
        if x.ndim == 1:
            return np.add.reduce(A*x*(b-a)/N, 0)
        else:
            return np.add.reduce(A.reshape(N, 1)*x*(b-a)/N, 0)
    def coeff(self, n, o = 0j, s = 0, S = 1, N = 500):
        """
        input: f:C -> C
        output: returns a number the n-th coefficient in the laurent expansion of f around o, in the annulus of radius s < r < S"""
        r = (s+S)/2
        c = lambda t: o+r*np.exp(2j*np.pi*t)
        dc = lambda t: 2j*np.pi*r*np.exp(2j*np.pi*t)
        gg = lambda t: dc(t)*self.f(c(t))*(c(t)-o)**(-n-1)+0j
        return gg.integral(0, 1, N)/(2j*np.pi)
    def plot_c(self, a = -1, b = 1, c = -1, d = 1, bad = False, is_wareframe = True, N = 100):
        """
        plot the real and imaginary parts of self in 3d 
        from x=a to x=b and from y=c to y=d with N steps
        bad means that self doesn't work with np.ndarrays"""
        z1, z2 = np.linspace(a, b, N), np.linspace(c, d, N)
        z3 = z2[:, np.newaxis]
        Z = z1+1j*z3
        if not bad:
            x = self.f(Z)
        else:
            x = np.array([[self.f(i) for i in j] for j in Z])
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        if not is_wareframe:
            ax.plot_surface(z1, z3, np.real(x))
            ax.set_aspect('equal')
            ax1.plot_surface(z1, z3, np.imag(x))
            ax1.set_aspect('equal')
            plt.show()
        else:
            ax.plot_wireframe(z1, z3, np.real(x), rstride=3, cstride=3)
            ax1.plot_wireframe(z1, z3, np.imag(x), rstride=3, cstride=3)
            plt.show()
    def plot_r(self, a = -1, b = 1, bad = False, N = 100):
        """
        plots the real part of self in the interval [a, b] with N steps
        bad means that self doesn't work with np.ndarrays"""
        X = np.linspace(a, b, N)
        if not bad:
            Y = np.real(self @ X)
        else:
            Y = [np.real(self @ x) for x in X]
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        plt.show()
    def plot_param(self, a, b, bad = False, N = 100):
        """
        Plots the values of self in the complex plane in the interval [a, b] with N steps
        bad means that self doesn't work with np.ndarrays"""
        A = np.linspace(a, b, N, dtype=complex)
        if not bad:
            Z = self @ A
        else:
            Z = np.array([self @ z for z in Z])
        X, Y = np.real(Z), np.imag(Z)
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        plt.show()

x = cfunction(lambda z: z, "z")
cexp = cfunction(np.exp, "exp(z)")
pi = cfunction(lambda _: np.pi, "pi")

# This is NICE          just be careful with the order of operations
# f = x**2-3*x+1
# X = np.linspace(0, 10, 100)
# Y = f @ X
# fig, ax = plt.subplots()
# ax.plot(X, Y)
# plt.show()

G = 0.318309886185+0j  # 1/pi?
# g(4z / (1+z)^2) = (1+z) g(z^2)     in D(0, 1), this is the amg relation
# g(z)g'(1-z)+g(1-z)g'(z) = G / z(1-z)
# (z^2-z)g'' + (2z-1)g' + (1/4)g = 0
# g(z) = integral_0^1{ dt/sqrt((1-t^2)(1-zt^2)) }
# the expansion at 0 has coefficients (pi/2) * (2n choose n)^2 / 16^n
def function_g(z):
    a, b = 1+0j, np.sqrt(1+0j-z)
    for _ in range(25):
        a, b = (a+b)/2, np.sqrt(a*b)
        b = b*np.sign(np.abs(a+b)-np.abs(a-b))
    return np.pi/(2*a)
g = cfunction(function_g, "g(z)")
dg = g.derivative()

K1 = g @ x**2; K1.r = "K1(z)"
K2 = g @ (1-x**2); K2.r = "K2(z)"
q = cexp @ (-pi*K2/K1)
tau = 1j*K2/K1
modular_lambda = tau.inverse(0.7)**2
# it can evaluate modular_lambda about 20 times per second
# the j-invariant can be found in terms of this


# Differential equation methods

def Runge_Kutta(F, t0, x0, tf, N = 100):
    """
    Solves x'=F(t, x) with condition x(t0)=x0,
    outputs x(tf), works with vectors too"""
    h = (tf-t0)/N
    #X = [x0]
    x = x0
    for n in range(N):
        t = t0 + h*n
        #x = X[-1]
        k1 = F(t, x)
        k2 = F(t+h/2, x+h*k1/2)
        k3 = F(t+h/2, x+h*k2/2)
        k4 = F(t+h, x+h*k3)
        #X.append(x+h*(k1+2*k2+2*k3+k4)/6)
        x = x+h*(k1+2*k2+2*k3+k4)/6
    #T = np.linspace(t0, tf, N+1)
    return x

def graphc_c_from_diff_eq(F, z0, w0, a = -1, b = 1, c = -1, d = 1, N = 30, except_points = [], m = 4, distance = 0.1):
    """
    graphs the function of f'=F(z, f) in the rectangle [a, b]x[c, d] with N steps per unit
    it avoids the points specified, and it does m steps of Runge Kutta per point
    In the case of second order equations, w0 = [f(z0)  f'(z0)] and f'' = F(z, f, f')"""
    w0 = np.array(w0)
    assert np.real(z0) > a and np.real(z0) < b and np.imag(z0) > c and np.imag(z0) < d, f"{z0} is not in the specified range"
    if type(except_points) == list:
        for p in except_points:
            assert np.abs(z0-p) > distance, f"{z0} is too close to an exception point"
    z1, z2 = np.linspace(a, b-1/N, N), np.linspace(c, d-1/N, N)
    z3 = z2[:, np.newaxis]
    initial = (int((np.real(z0)-a)*N/(b-a)), int((np.imag(z0)-c)*N/(d-c)))
    Solved, Border = {}, [[initial], [w0]]        # solved is a dict of coordinate: value

    i = 0
    while Border != [[], []]:
        coord_z, w = Border[0][0], Border[1][0]
        z = a+coord_z[0]*(b-a)/N + 1j*(c+coord_z[1]*(d-c)/N)

        new_coord = None
        if i == 0 and coord_z[0] > 0:
            new_coord = (coord_z[0]-1, coord_z[1])
        elif i == 1 and coord_z[1] > 0:
            new_coord = (coord_z[0], coord_z[1]-1)      # choose the new point
        elif i == 2 and coord_z[0] < N-1:
            new_coord = (coord_z[0]+1, coord_z[1])
        elif i == 3 and coord_z[1] < N-1:
            new_coord = (coord_z[0], coord_z[1]+1)
        elif i == 4:
            Solved[coord_z] = w
            del Border[0][0]
            del Border[1][0]
            i = 0
            continue
        if new_coord == None or new_coord in Solved or new_coord in Border[0]:
            i += 1
            continue

        zf = a+new_coord[0]*(b-a)/N + 1j*(c+new_coord[1]*(d-c)/N)
        close_to_a_pole = False
        for p in except_points:                          # exceptions
            if np.abs(zf-p) < distance:
                close_to_a_pole = True
                break
        if close_to_a_pole:
            i += 1
            continue
        
        if len(w0) != 2:
            wf = Runge_Kutta(lambda t, f: (zf-z)*F(t*(zf-z)+z, f), 0, w, 1, m)  # the value at the new point
        else:
            def G(z, y):
                """[f'  f''] = G(z, [f  f'])"""
                return np.array([y[1], F(z, y[0], y[1])])                       # second order case
            wf = Runge_Kutta(lambda t, f: (zf-z)*G(t*(zf-z)+z, f), 0, w, 1, m)  # w, wf store [f(z)  f'(z)]
        Border[0].append(new_coord)
        Border[1].append(wf)

    Values = np.zeros((N, N), dtype=complex)
    for coord in Solved.keys():
        if len(w0) != 2:
            Values[coord[1], coord[0]] = Solved[coord]     # the pairs are (x, y), so reverse them
        else:
            Values[coord[1], coord[0]] = Solved[coord][0]   # change the last 0 to 1 to plot f' instead
    return Values
    #fig = plt.figure()
    #ax = fig.add_subplot(121, projection='3d')
    #ax1 = fig.add_subplot(122, projection='3d')
    #ax.plot_surface(z1, z3, np.real(Values))
    #ax.set_aspect('equal')
    #ax1.plot_surface(z1, z3, np.imag(Values))
    #ax1.set_aspect('equal')
    #plt.show()

def integrate_diff_eq(F, P, w0, N = 100):      #integrates f'' = F(z, f, f') along the path P=[z0, z1, ... ]
    k = len(P)-1
    w = np.array(w0)
    for n in range(k):
        def G(t, y):    # [f'  f''] = G(t, [f  f'])
            return (P[n+1]-P[n])*np.array([y[1], F((P[n+1]-P[n])*t+P[n], y[0], y[1])])
        w = Runge_Kutta(G, 0, w, 1, N)
    return w

F = lambda z, f, df: f/(4*z*(1-z))+df*(2*z-1)/(z*(1-z))     # diff eq. of g
w = np.array([1.31102878+1.31102878j, -0.47752472-0.17798967j])     # the value of g, dg at 2, going above 1
w1 = np.array([1.31102878-1.31102878j, -0.47752472+0.17798966j])    # the same but going under 1

# a, b, c, d, N = -3, 3, -3, 3, 20
# main_sheet = graphc_c_from_diff_eq(F, -1, [g @ -1, dg @ -1], a, b, c, d, N, [0, 1], 10, 0.05)
# over_zero = graphc_c_from_diff_eq(F, 2, w, a, b, c, d, N, [0, 1], 10, 0.05)
# under_zero = graphc_c_from_diff_eq(F, 2, w1, a, b, c, d, N, [0, 1], 10, 0.05)

# z1, z2 = np.linspace(a, b-1/N, N), np.linspace(c, d-1/N, N)
# z3 = z2[:, np.newaxis]
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax1 = fig.add_subplot(122, projection='3d')
# ax.plot_surface(z1, z3, np.real(main_sheet), label="Main sheet")
# ax.plot_surface(z1, z3, np.real(under_zero), label="Under zero")
# ax.plot_surface(z1, z3, np.real(over_zero), label="Over zero")
# ax.set_aspect('equal')
# ax1.plot_surface(z1, z3, np.imag(main_sheet), label="Main sheet")
# ax1.plot_surface(z1, z3, np.imag(under_zero), label="Under zero")
# ax1.plot_surface(z1, z3, np.imag(over_zero), label="Over zero")
# ax1.set_aspect('equal')
# ax.legend()
# ax1.legend()
# plt.show()

# plot the transformation z^2 -> 4z / (1+z)^2
num_points = 50
radius = np.random.random(num_points)
angles = 0.5 * np.pi * (np.random.random(num_points) - 0.5)
points = radius * np.exp(1j * angles)

def intermediate_points(points, t):
    """
    Returns the values of the points at time t
    t = 0: z^2, t = 1: 4z/(1+z)^2
    points: (nun_points, 1)
    t: (1, num_t)
    out: (num_points, num_t)
    """
    return (1 - t) * points**2 + t * 4 * points / (1 + points)**2

def plot_points_progression(points, int_points_fun, num_steps=100):
    """
    Makes graphs showing the points while they move from t = 0 to t = 1
    int_poits_fun: function that takes (points, t) and returns points
    """
    times = np.linspace(0, 1, num_steps)
    colors = np.arange(len(points) + 2)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Transformation of points')
    int_points = int_points_fun(  # (num_points, num_times)
        points.reshape((-1, 1)),
        times.reshape((1, -1)),
    )
    # plot the paths
    for p, c in zip(int_points, colors):  # p: (num_times)
        ax.plot(np.real(p), np.imag(p), times, c=f'C{c}')
    # plot circles for reference
    ax.plot(np.cos(2*np.pi*times), np.sin(2*np.pi*times), 0, c=f'C{colors[-2]}')
    ax.plot(np.cos(2*np.pi*times), np.sin(2*np.pi*times), 1, c=f'C{colors[-1]}')

    plt.show()

plot_points_progression(points, intermediate_points)