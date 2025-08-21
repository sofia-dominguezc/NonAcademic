#PLEASE SHE ME BECAUSE I FEEL BAD

print("sofia you are a beautiful girl")

#OMGG I JUST READ THIS AND AAAAAA
#I LOVED BEING SHE'D THOUGH

#this is sofia from the future and can confirm
#it's great



# This is the document with all matplotlib notes (numpy below)

print('hi girl')

import matplotlib.pyplot as plt
import numpy as np

fig1, ax1 = plt.subplots()  # Create a figure containing a single axes.
ax1.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Add data to the axes.
ax1.plot(np.array([1, 2, 3, 4]), np.array([4, 1, 3, 1]))    # It also works with numpy

#fig = plt.figure()  # an empty figure with no Axes
#fig, axs1 = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
fig, axs2 = plt.subplot_mosaic([['left', 'right_top'],  # a figure with one axes on the left, and two on the right:
                                ['left', 'right_bottom']])
#fig, axs3 = plt.subplot_mosaic([['one', 'right'], ['one', 'right']])    # the name doesn't matter

ax1.set_title('First figure')
ax1.set_xlabel('This is the x axis')

plt.show()




# ALL NUMPY NOTES

a1, b1, c1 = np.array([1, 2, 3]), np.array([-2, 1, 3]), np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
d1, e1 = a1+b1, a1*b1

a2, b2, c2 = np.arange(2, 5, dtype=int), np.arange(2, 5, 0.1), np.linspace(2, 5, 10)    # the step, the amount

a3, b3, c3 = np.eye(4, 3, dtype=int), np.diag([2, -1, 3, 4]), np.diag([[-1, 2, 5], [3, 4, 6], [-7, 4, 8]]) #id

a4, b4, c4 = np.vander((1, 2, 3, 4), 5), np.zeros([2, 3], dtype=int), np.ones([3, 6])

a5, b5 = c1[1], c1[1].copy()
a5 += 1     #this updates c1 too
b5 += 1     #this doesn't

a6, b6, c6, d6 = np.array([[1, 2], [3, 4]]), np.diag([-5, 7]), np.zeros((2, 2)), np.eye(2, dtype=int)
e6 = np.block([[a6, c6], [b6, d6]])

a8, b8, c8 = a6[1, 1], a6[1], np.diag(np.arange(10, dtype=int))
d8 = c8[-1:3:-2]      # start : end (without itself) : step
e8 = c8[3:4]          # assumes the maximum
f8 = c8[5:]           # assumes step 1
g8 = c8[[2, 3, 5]]    # choose specific values

a9 = np.arange(1, 10).reshape(3, 3)
b9 = a9[:, 1]           # slice any dimension
c9 = a9[0::2, [0, 1]]   # slice specific rows and columns

a10 = np.arange(1, 17).reshape(4, 4)
b10 = np.add.reduce(a10, 0)         # adds every column         #doesn't update the input
c10 = np.add.reduce(a10, 1)         # adds every row
d10 = np.multiply.reduce(a10, 1)    # multiplies column / row

a11, b11 = np.array([2, 3, -1]), np.array([[1, -1, 0], [4, 3, 2]])
c11 = a11*b11

a12, b12 = np.arange(5)+1j*np.ones([5]), np.vander([1j], 5)[0]
c12 = np.dot(a12, b12).imag         # the standard dot product, no conjugation

a13, b13 = np.vander([2, 1-3j], 4), np.arange(12).reshape(4, 3)
c13, d13, q13 = np.dot(a13, b13), np.matmul(a13, b13), a13 @ b13        # matrix product
e13 = np.arange(9).reshape(3, 3)+1j*np.arange(9, 0, -1).reshape(3, 3)+np.eye(3)
f13, g13, h13 = np.linalg.matrix_power(e13, -2), np.linalg.matrix_rank(e13), np.linalg.det(e13)
i13, j13, k13 = np.linalg.eig(e13), np.transpose(e13), np.trace(e13)#eigenvalues (not ordered) and eigenvectors
l13, p13 = np.linalg.solve(e13, np.arange(3)+1j*np.ones(3)), np.linalg.inv(e13)   #input (a, b), solves ax = b

a14, b14 = np.arange(4).reshape(2, 2), np.array([3, -1])
c14 = np.matmul(a14, b14)   #for evaluation of vectors, the dimensions can be 1 x n, this doesn't work with @

def char_poly(A):       # characteristic polynomial of A
    n = len(A)+1
    w = np.exp(2j*np.pi/n)
    k = np.vander([w], n)[0]
    B = np.vander(k, n)
    y = np.linalg.det(np.array([A]*n)-np.multiply(np.array([np.eye(n-1)]*n),
                                    np.expand_dims(k, axis=[1, 2])))
    return np.flip(np.linalg.solve(B, y))

x1 = np.linspace(-np.pi, np.pi, 100)
x = x1 + 1j*x1[:, np.newaxis]
y = np.exp(x)
#plt.subplot(121)
#plt.imshow(np.abs(y), extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi], cmap='gray')
#plt.subplot(122)
#plt.imshow(np.angle(y), extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi], cmap='hsv')
#plt.show()

a15, b15 = np.matrix([[0, -1], [1, 0]]), np.matrix([[1, -1], [1, 1]])   # doesn't work in matplotlib
c15 = a15*b15       # matrix multiplication

data = {'a': np.arange(10),
        'b': np.array([4,6,1,2]),
        'c': np.linspace(-1, 1, 10)}
b = data['a']   # is the list [0, 1, 2, ..., 10]
