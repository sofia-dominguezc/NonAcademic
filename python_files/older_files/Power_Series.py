import numpy as np
import scipy

def _mul(array1, array2):
    """Implements multiplication of arrays of the same length
    The return also has the same length (higher order terms are deleted)"""
    n = len(array1)
    assert len(array2) == n, "Multiplying arrays of different lengths"
    return np.flip(
        np.convolve(np.flip(array1), np.flip(array2))  # O(n^2)
    )[:n]

def _pow(array, exponent):
    """Implements the exponentiation of arrays by using base 2"""
    assert isinstance(exponent, int), "The exponent must be an int"
    result = np.zeros(len(array)); result[0] = 1  # initialize the power
    pure_power = array
    while exponent > 0:  # O(log n) steps
        if exponent % 2 == 1:
            result = _mul(result, pure_power)  # O(n^2)
        pure_power = _mul(pure_power, pure_power)  # O(n^2)
        exponent //= 2
    return result

def _div(array1, array2=None):
    """Implements division of arrays of the same length. Runtime is O(n^2)
    Returns: array2/array1
             If array2 is not provided, returns 1/array1"""
    assert array1[0] != 0, "Inverting a Series with non-zero constant term."
    n = len(array1)
    e1 = np.zeros(n); e1[0] = 1  # to use later
    topeliz_matrix = scipy.linalg.toeplitz(array1, e1)
    # Idea: solve the matrix equation  toepliz_matrix @ v = array2
    if array2 is not None:
        assert len(array2) == n, "Dividing arrays of different lengths"
        return scipy.linalg.solve_triangular(topeliz_matrix, array2, lower=True)
    else:
        return scipy.linalg.solve_triangular(topeliz_matrix, e1, lower=True)

class Series():
    """
    Stores an 'infinite' series a0 + a1*X + a2*X^2 + ...
    It stores some coefficients and has a function to calculate the next.
    It supports all field operations, differentiation, and substitution.
    """
    num_digits = 8  # number of digits to check if two numbers are equal

    def __init__(self, vec=None, get_coeff=None, var="z"):
        """
        Initializes a Series: stores a few values and the generator function.
           If no data provided, initializes a zero Series or pads with zeros

        var: name of the variable, by default it's z
        vec: coefficients already known, list or np.array, in increasing order
        get_coeff: function that takes a non-negative int n and returns the
                   coefficient of X^n in the Series
        
        self.vec: values already calculated
        self.gen: iterator that recursively returns the next coefficient of 
                  the series, starting from the first that vec doesn't have.
                  self.gen = None means that it's a polynomial.

        NOTE: the generators may call the _extend() method on other instances
              the generators use yield and can use self.vec
              the generators DO NOT update self.vec, only find the next coeff
        """
        self.var = var
        # store some values
        if vec is not None:
            self.vec = np.array(vec)  # use the already known values
        else:
            assert get_coeff is not None,\
                  "Neither coefficients nor a generator were provided"
            self.vec = np.array(
                [get_coeff(n) for n in range(8)]  # start with 8 values
                )
        # define the generator
        if get_coeff is not None:  # this is only for instances created from outside
            def generator():
                """Returns the first not known coefficient"""
                idx = len(self.vec)  # start in the first not known index
                while True:
                    yield get_coeff(idx)
                    idx += 1
            self.gen = generator()
        else:  # this is for internal use or to pad with zeros (see _extend)
            self.gen = None  # it'll be overwritten later

    def _extend(self, num_terms=1):
        """Advances the generator and updates self.vec."""
        for _ in range(num_terms):
            # find new value and initialize the new vector
            if self.gen is None:
                new_val = 0
            else:
                new_val = next(self.gen)
            new_vec = np.zeros(len(self.vec) + 1)
            # create the new vector and update
            new_vec[:-1] = self.vec
            new_vec[-1] = new_val
            self.vec = new_vec

    def _make_equal(self, other):
        """Extend the shorter Series to the length of the longest"""
        assert isinstance(other, Series), "Error in _make_equal"
        n, m = len(self.vec), len(other.vec)
        if n < m:  # self is shorter
            self._extend(num_terms=m-n)
        else:  # other is shorter
            other._extend(num_terms=n-m)

    def __eq__(self, other, num_terms=16):
        """Compares equality up to num_terms with num_digits precision"""
        num_digits = Series.num_digits
        if not isinstance(other, Series):
            return False
        # calculate enough coefficients
        if len(self.vec) < num_terms:
            self._extend(num_terms=num_terms-len(self.vec))
        if len(other.vec) < num_terms:
            other._extend(num_terms=num_terms-len(other.vec))
        self._make_equal(other)  # so both have the same length

        are_equal = True  # change to False if they are different
        for coeff1, coeff2 in zip(self.vec, other.vec):
            if np.round(coeff1, num_digits) != np.round(coeff2, num_digits):
                are_equal = False
        return are_equal

    def __add__(self, other):
        if isinstance(other, Series):
            # make sure both Series have the same number of coefficients
            self._make_equal(other)
            # store some values
            new_vec = self.vec + other.vec
            # create instance
            result = Series(new_vec, var=self.var)
            # create the generator
            def generator():
                idx = len(new_vec)  # idx to get in this step
                while True:
                    # calculate new values if necessary
                    if len(self.vec) <= idx:  # (extend at most once)
                        self._extend()
                    if len(other.vec) <= idx:
                        other._extend()
                    # return the correct value
                    yield self.vec[idx] + other.vec[idx]
                    idx += 1
            result.gen = generator()

        else:  # if 'other' is a scalar
            new_vec = self.vec.copy()
            new_vec[0] += other
            # create instance
            result = Series(new_vec, var=self.var)
            # create the generator (basically return the same coeff)
            def generator():
                idx = len(new_vec)  # initialize index
                while True:
                    # calculate new values if necessary
                    if len(self.vec) <= idx:
                        self._extend()
                    # return the same value
                    yield self.vec[idx]
                    idx += 1
            result.gen = generator()

        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        """Returns the negative of a series"""
        new_vec = -self.vec
        result = Series(new_vec, var=self.var)
        def generator():
            idx = len(new_vec)
            while True:
                # calculate enough values
                if len(self.vec) <= idx:
                    self._extend()
                # return the value
                yield -self.vec[idx]
                idx += 1
        result.gen = generator()
        return result

    def __sub__(self, other):
        if isinstance(other, Series):
            # make sure both Series have the same number of coefficients
            self._make_equal(other)
            # store some values
            new_vec = self.vec - other.vec
            # create instance
            result = Series(new_vec, var=self.var)
            # create the generator
            def generator():
                idx = len(new_vec)  # idx to get in this step
                while True:
                    # calculate new values if necessary
                    if len(self.vec) <= idx:  # (extend at most once)
                        self._extend()
                    if len(other.vec) <= idx:
                        other._extend()
                    # return the correct value
                    yield self.vec[idx] - other.vec[idx]
                    idx += 1
            result.gen = generator()
            return result
        else:
            return self + (-other)
    
    def __rsub__(self, other):
        """It's called in the form other - self when self is a scalar"""
        new_vec = -self.vec
        new_vec[0] += other
        # create instance
        result = Series(new_vec, var=self.var)
        # create the generator (the same as for __neg__)
        def generator():
            idx = len(new_vec)  # initialize index
            while True:
                # calculate new values if necessary
                if len(self.vec) <= idx:
                    self._extend()
                # return the negative
                yield -self.vec[idx]
                idx += 1
        result.gen = generator()
        return result

    def __mul__(self, other):
        if isinstance(other, Series):
            # make sure the loaded arrays have the same size
            self._make_equal(other)
            # calculate the product of known coefficients
            new_vec = _mul(self.vec, other.vec)
            result = Series(new_vec, var=self.var)
            # define the generator
            def generator():
                idx = len(new_vec)
                while True:
                    # extend parents if needed
                    if len(self.vec) <= idx:
                        self._extend()
                    if len(other.vec) <= idx:
                        other._extend()
                    yield np.dot(
                        self.vec[:idx+1], np.flip(other.vec[:idx+1])
                    )
                    idx += 1
            result.gen = generator()
        else:
            new_vec = other * self.vec
            result = Series(new_vec, var=self.var)
            def generator():
                idx = len(new_vec)
                while True:
                    # extend parent if needed
                    if len(self.vec) <= idx:
                        self._extend()
                    yield other * self.vec[idx]
                    idx += 1
            result.gen = generator()
        return result

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        """It uses base 2 exponentiation for the already loaded values,
        and it uses the formula based on g=f^n -> g'f = n*gf
        for the generator, so the next coeff is found in O(n).
        It supports fractional exponents."""
        m = exponent
        if np.round(self.vec[0], Series.num_digits) != 0:  # use formula
            a0 = self.vec[0]
            if isinstance(m, int):  # integer power
                new_vec = _pow(self.vec, m)
            else:  # fractional power
                assert np.round(a0, Series.num_digits) == 1, \
                "To raise a Series to a non-integer power, it must have " +\
                "constant term 1."
                new_vec = np.array([1])  # first term
            result = Series(new_vec, var=self.var)
            # generator
            def generator():
                idx = len(new_vec)
                while True:
                    if len(self.vec) <= idx:
                        self._extend()
                    yield (1/(a0*idx)) * np.sum(
                        (np.arange(1, idx + 1)*(m+1) - idx) *\
                        self.vec[1: idx + 1] *\
                        np.flip(result.vec)
                    )
                    idx += 1
            result.gen = generator()
        else:  # use shifted formula
            assert isinstance(m, int), "Trying to raise Series with zero " +\
            "constant term to a non-integer power."
            if m < 0:
                return (1/self)**(-m)
            new_vec = _pow(self.vec, m)
            result = Series(new_vec, var=self.var)
            # define the generator:
            #   all terms idx < ord*m are zero
            #   the term ord*m is a0**m
            #   after that use the recursive formula
            def generator():
                # first try to find the order of self
                ord = 0  # self.vec[ord] is the first non-zero
                order_found = False  # True if ord is assigned correctly
                while ord < len(self.vec):
                    if np.round(self.vec[ord],  # first non-zero term
                                Series.num_digits) != 0:
                        order_found = True
                        a0 = self.vec[ord]
                        break
                    ord += 1
                # now start the generation
                idx = len(new_vec)
                while True:
                    if len(self.vec) <= idx:
                        self._extend()
                    if not order_found:  # try to update it
                        if np.round(self.vec[ord]) != 0:  # first non-zero term
                            order_found = True
                            a0 = self.vec[ord]
                        else:
                            ord += 1
                    if not order_found:  # all terms up to now are 0
                        yield 0
                    elif idx < ord * m:
                        yield 0
                    elif idx == ord * m:
                        yield a0**m
                    else:  # use the formula above but slightly shifted
                        yield (1/(a0*(idx - ord*m))) * np.sum(
                            (np.arange(1, idx - ord*m + 1)*(m+1) - idx + ord*m) *\
                            self.vec[ord + 1: idx - ord*(m-1) + 1] *\
                            np.flip(result.vec[ord*m: idx])
                        )
                    idx += 1
            result.gen = generator()
        return result

    def __truediv__(self, other):
        """Division takes O(n^2), the same as multiplication"""
        if isinstance(other, Series):
            # make sure both series have the same num of coeffs
            self._make_equal(other)
            # calculate known coefficients
            new_vec = _div(other.vec, self.vec)
            result = Series(new_vec, var=self.var)
            # define the generator
            def generator():
                idx = len(new_vec)
                a0 = other.vec[0]
                while True:
                    if len(self.vec) <= idx:
                        self._extend()
                    if len(other.vec) <= idx:
                        other._extend()
                    yield (1/a0) * (self.vec[idx] - np.sum(
                        other.vec[1: idx+1] *\
                        np.flip(result.vec)  # has length idx
                    ))
                    idx += 1
            result.gen = generator()
            return result
        else:
            return self * (1/other)

    def __rtruediv__(self, other):
        """Only called for other/self when 'other' is a scalar"""
        # calculate known coefficients
        new_vec = other * _div(self.vec)
        result = Series(new_vec, var=self.var)
        # define the generator
        def generator():
            idx = len(new_vec)
            a0 = self.vec[0]
            while True:
                if len(self.vec) <= idx:
                    self._extend()
                yield (-1/a0) * np.sum(
                    self.vec[1: idx+1] *\
                    np.flip(result.vec)  # has length n
                )
                idx += 1
        result.gen = generator()
        return result

    def deriv(self):
        """Returns the derivative"""
        # calculate known coefficients
        new_vec = self.vec[1:] * np.arange(1, len(self.vec))
        result = Series(new_vec, var=self.var)
        # define the generator
        def generator():
            idx = len(new_vec)
            while True:
                if len(self.vec) <= idx + 1:
                    self._extend()
                yield (idx+1) * self.vec[idx+1]
                idx += 1
        result.gen = generator()
        return result

    def inv(self):
        """Returns the compositional inverse, finds a new coefficient
        in O(n^2 log n). It could be optimized by using fourier transform for
        multiplication but it seems to be slower and numerically unstable."""
        assert np.round(self.vec[0], Series.num_digits) == 0, \
        "The series must have zero constant term to invert it."
        if len(self.vec) == 1:
            self._extend()
        assert np.round(self.vec[1], Series.num_digits) != 0, \
        f"The coefficient of {self.var}^1 must be non-zero to invert it."
        # initialize the result
        new_vec = np.array([0, 1/self.vec[1]])  # first two terms
        result = Series(new_vec, var=self.var)
        # store the Series z/f(z) to speed up computation
        new_vec_2 = self.vec[1:].copy()
        self_shifted = Series(new_vec_2, var=self.var)
        def generator_2():
            idx = len(new_vec_2)
            while True:
                if len(self.vec) <= idx+1:
                    self._extend()
                yield self.vec[idx+1]
                idx += 1
        self_shifted.gen = generator_2()
        self_inv = 1/self_shifted
        # make the generator, it will use self_inv
        # The strategy is to use the lagrange inversion formula,
        #   so I will calculate the n-1 coeff in (z/f(z))^n over n
        def generator():
            idx = len(new_vec)  # want the n-th coefficient; n=idx
            while True:
                if len(self_inv.vec) <= idx-1:
                    self_inv._extend()
                self_inv_pow = self_inv**idx  # O(n log^2 n)
                yield self_inv_pow.vec[idx-1]/idx
                idx += 1
        result.gen = generator()
        return result

    def __call__(self, other, return_error=False, num_terms=16):
        """If other is a Series, computes the series composition.
        If not, then other can be a scalar or a 1d list, it uses num_terms
        terms to calculate the expansion and can also return the errors"""
        if isinstance(other, Series):
            raise NotImplementedError
        else:
            # calculate enough coefficients
            N = num_terms
            if len(self.vec) <= N:
                self._extend(num_terms = N + 1 - len(self.vec))
            # compute the powers of the "other" scalar or list
            other = np.array(other)
            if other.ndim == 0:
                van_matrix = np.vander(np.array([other]), N, increasing=True)
            elif other.ndim == 1:  # columns are other**0, other**1, ...
                van_matrix = np.vander(other, N, increasing=True)  # (., N)
            else:
                assert False, "A Series can only be evaluated in a 0d or 1d array."
            # compute the result and change the shape in the 0d case
            result = van_matrix @ self.vec[:N]
            if other.ndim == 0:
                result = result[0]
            # compute the error(s) in the expansion and check if it's small enough
            error = np.abs(self.vec[N] * other**N)
            if np.any(error > 10**-Series.num_digits):  # if not enough precision
                if num_terms < 512:  # try with more terms
                    return self.__call__(
                        other, return_error=return_error, num_terms=2*num_terms
                        )
                else:
                    assert False, f"The series evaluation of {self} at {other} "+\
                    f"did not converge. The last iteration had output {result} "+\
                    f"with last term equal to {error}."
            # return the answer
            if return_error:
                return result, error
            else:
                return result

    def __str__(self, num_terms=8, num_digits=4):
        """Creates a str representation of the series with >= 8 terms"""
        if len(self.vec) < num_terms:  # calculate enough coefficients
            self._extend(num_terms=num_terms-len(self.vec))
        str_repr = ""
        first_coeff = True
        for n, coeff in enumerate(self.vec[:num_terms]):
            int_term = np.int64(np.round(coeff))
            rounded_term = np.round(coeff, num_digits)
            if rounded_term == 0:
                continue
            # make the sign
            if first_coeff:
                sign = "" if coeff > 0 else "- "
                first_coeff = False
            else:
                sign = " + " if coeff > 0 else " - "
            # make the X^n term
            if n == 0:
                exponent = ""
            elif n == 1:
                exponent = self.var
            else:
                exponent = self.var + "^" + str(n)
            # make the coefficient, round and keep ints as ints
            if n > 0 and (rounded_term == 1 or rounded_term == -1):
                term = ""
            else:
                if rounded_term == int_term:  # use int_term
                    term = str(np.abs(int_term))
                else:  # use rounded_term
                    term = str(np.abs(rounded_term))
            # update the str
            str_repr += sign + term + exponent
        if first_coeff:  # if all asked coefficients are zero
            str_repr += "0"
        str_repr += f" + O({self.var}^{num_terms})"
        return str_repr

class MultipleSeries():
    """
    It represents a complex differentiable function by storing its taylor
    expansion around a set of points, along with (in some cases) the
    radius of convergence.
    It will be able to automatically find the coefficients around another
    point in the domain by complex integration.
    """
    def __init__(self):
        raise NotImplementedError

x = Series([0, 1], var="x")
def inv_factorial(n):
    prod = 1
    for k in range(1, n+1):
        prod *= k
    return 1/prod
exp = Series(get_coeff=inv_factorial, var="x")  # e^x
log = (exp - 1).inv()  # log(1+x)

print((x - x**2).inv())

def g(z):
    a, b = 1+0j, np.sqrt(1+0j-z)
    for _ in range(25):
        a, b = (a+b)/2, np.sqrt(a*b)
        b = b*np.sign(np.abs(a+b)-np.abs(a-b))
    return np.pi/(2*a)

# (pi/2) * (2n choose n)^2 / 16^n
def coeff_g(n):
    twon_choose_n = 1  # (2n choose n)/4^n
    for a, b in zip(range(1, n+1), range(n+1, 2*n+1)):
        twon_choose_n *= b/(4*a)
    return (np.pi/2) * twon_choose_n**2

g_pow = Series(get_coeff=coeff_g, var="z")  # equal to g in the unit circle

print(g_pow(1))
values = (np.random.rand(12) + 1j*np.random.rand(12))/2
print(np.abs(g_pow(values) - g(values)))
