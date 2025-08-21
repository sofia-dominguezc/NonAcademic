# notes on opening/writting in files            pytorch later

#file = open("text_for_tests.txt", "a")        # a for appending at the end. w for overwritting
#L = ["\n", "I've been feeling better lately.\n"]        # \n for a new line
#file.writelines(L)
#file.write("I like looking like a girl. \n")
#file.write('Yess')
#file.close()
#file = open("text_for_tests.txt", "r")
#print(file.read())
    # file.readline() reads a line (paragraph), calling it twice reads two lines
    # it is possible to iterate through the lines of a file with:            for x in file
#file.close()

#file = open("new_file", "a")                # a and w create a file if it doesn't exist. x only creates it
#L = ["This file should be completetely new\n", 
#        "And this is the only text that it should contain\n", "I reaaaally like to look like a girl"]
#file.writelines(L)
#file.close()
#file = open("new_file", "r")
#print(file.read())
#file.close()                               # package "os" for removing files and checking if files exist

# with open("text_for_tests.txt", "a") as file:
#    file.write("\n")
#    file.write("'with' always closes the file and makes sure no errors happen")
# with open("text_for_tests.txt", "r") as file:
#    print(file.read())
# with open("new_file.txt", "w") as file:
#    file.write("okkk lets see if this works")





# import torch                                # see pytorch documentation
# import matplotlib.pyplot as plt

# # almost everything from numpy works here
# a1, b1 = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3), torch.ones((3, 3))+2*torch.eye(3)
# c1, d1 = b1 @ a1, torch.linspace(0, 1, 10)        # or matmul

# a2 = torch.rand((3, 2))             # vs np.random.rand(3, 2)
# b2 = torch.transpose(a2, 0, 1)      # specify axis, not in numpy

# # example: find a cuadratic that minimizes the mean of squares error

# # data_x = torch.linspace(-5, 5, 100)
# # data_y = 0.8131*data_x**2-0.6597*data_x+1.3956 + torch.rand((100, ))

# #fig, ax = plt.subplots()
# #ax.plot(data_x, data_y, 'o')
# #plt.show()

# params = {"a": torch.rand((), requires_grad=True),      # no CUDA suported by this tablet :(
#           "b": torch.rand((), requires_grad=True),
#           "c": torch.rand((), requires_grad=True)}
# def predict(x):
#     return params["a"]*x**2+params["b"]*x+params["c"]

# f = lambda x: x**2-5*x

# x1 = torch.tensor(10., requires_grad=True)
# y1 = f(x1)
# y1.backward()
# print(x1.grad)





# I'm gonna try to code a neural network :)


import torch
# from torch import utils
# from torchvision import datasets, transforms

# # Transform PIL image into a tensor. The values are in the range [0, 1]
# t = transforms.ToTensor()

# # Load datasets for training and apply the given transformation.
# mnist = datasets.MNIST(root='data', train=True, download=True, transform=t)

# # Specify a data loader which returns 500 examples in each iteration.
# n = 500
# loader = utils.data.DataLoader(mnist, batch_size=n, shuffle=True)

# count = 0
# for imgs, labels in loader:     #imgs[i, 0] is a 28x28 array
#     if count == 5:
#         break
#     im = [[torch.round(imgs[0, 0, i, j], decimals=1) for j in range(28)] for i in range(28)]
#     print(f"Label is {labels[0]}")
#     count += 1


torch.random.manual_seed(0)
first_weigths = torch.rand((16, 28*28))
first_bias = torch.rand((16, ))
second_weigths = torch.rand(16, 16)
second_bias = torch.rand((16, ))
third_weigths = torch.rand((10, 16))
third_bias = torch.rand((10, ))

def classify(I, n):
    """I is a 28x28 torch array, 0 <= n <= 9 is an int
    returns: distance of the answer of the neural network to n (sum of squares)"""
    zero_layer = torch.reshape(I, (-1, ))
    first_layer = torch.sigmoid(first_bias+torch.matmul(first_weigths, zero_layer))
    second_layer = torch.sigmoid(second_bias+torch.matmul(second_weigths, first_layer))
    third_layer = torch.sigmoid(third_bias+torch.matmul(third_weigths, second_layer))
    goal = torch.zeros(10); goal[n] = 1
    return torch.sum((third_layer-goal)**2)

def objective_function(data):
    """data is an iterable (img, result)
    returns: the average of the output for each
    Represents how well the network classifies the images of data"""
    mean = 0
    for I, n in data:
        mean += classify(I, n)
    return mean/len(data)

# assuming a fixed training data,
# the function has inputs (weights, bias) and outputs a number
# to find the gradient, it's enough to differentiate with respect to the weights and bias

# then, subtract the gradient to the current weights and bias

def gradient(data):
    """finds the gradient of the objective function when I feed 'data' to it
    returns: a tuple of six elements, the first is a 16x28*28 where each entry is the value
    of the derivative of the cost with respect to that specific weight, then it's the first bias, and so on"""
    