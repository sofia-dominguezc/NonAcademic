# import random
# random.seed(0)

# HEAPS -------------------------------------------------
# import heapq
# A tree such that L[i] -> L[2i+1],L[2i+2] and the parent is smaller than the daughters
# Advantage: construct it in O(n) time and insert and element/pop the smallest in O(log n) time

# L = [random.randint(0, 100) for _ in range(10)]
# print(L)
# heapq.heapify(L)
# print(L)
# heapq.heappushpop(L, 6)     # this one adds the element before poping the smallest
# print(L)
# heapq.heapreplace(L, 2)     # This one pops before adding the element
# print(L)
# heapq.heappop(L)
# print(L)
# heapq.heappush(L, 3)
# print(L)


# DOUBLY LINKED LIST -------------------------------------
# A list that takes O(1) time to add an element/pop an element on both sides
from collections import deque

# a = deque(L)
# print(a)
# a.appendleft(2)             # there is append and appendleft, pop and popleft
# print(a)
# a.pop()
# print(a)
# a.popleft()                 # there is copy, reverse, insert, extend, and extendleft
# print(a)



class Rubik2():
    """instances of the class have the position of all pieces and their orientation."""

    def __init__(self, perms = [i for i in range(7)], rots = [0 for _ in range(7)]):
        """the vertex 0 is originally in ulf, 1 in urf, 2 in urb, 3 in ulb, 4 in dlf, 5 in drf, 6 in drb
           (however, the vertex 0 may be in another position after some turns)
           orientation 0 means white/yellow is on top/bottom, 1 means white/yellow in front/back, 2 in right/left
           perms is a list s.t. the vertex i is in the initial position of the vertex perms[i]
           rots is a list s.t. rots[i] is the orientation of vertex i, either 0, 1, 2"""
        assert isinstance(perms, list) and len(perms) == 7, "invalid set of permutations for new cube state"
        assert isinstance(rots, list) and len(rots) == 7, "invalid set of rotations for new cube state"
        self.p = perms.copy()
        self.r = rots.copy()
    def compress(self):
        """Compressed and hashable instance of a cube"""
        return perms_rots_to_compr(self.p, self.r)

    def move(self, movement):
        """Returns a new Rubik2 instance according to the movement specified."""
        assert movement in possible_moves, "Invalid movement."
        new_compr = move_compr(self.compress(), movement)  # compress first to make the movement
        return Rubik2(
            list(compr_to_perms(new_compr)),
            list(compr_to_rots(new_compr))
        )
    def moves(self, movements):
        """Movements is a list: ["R'", "U", "F2", ...]"""
        new_cube = self.copy()  # iterate in this new variable
        for movement in movements:
            new_cube = new_cube.move(movement)
        return new_cube

    def __eq__(self, other):
        return self.p == other.p and self.r == other.r
    def __str__(self):
        return "p: " + str(self.p) + ", r: " + str(self.r)
    def __hash__(self):
        return hash(str(self))
    def copy(self):
        """Returns a copy of the cube instance"""
        return Rubik2(self.p, self.r)

def perms_rots_to_compr(perms, rots):
    """Given the iterables of perms and rots, returns a compressed cube"""
    # compress the rotations as a 7-digit number in base 3
    rots_compr = 0  # number between 0 and 2186
    for i, r in enumerate(rots):
        rots_compr += r * 3**i
    # compress the permutation as a 7-digit number in base 7
    perms_compr = 0  # number between 0 and 823542
    for i, p in enumerate(perms):
        perms_compr += p * 7**i
    return (perms_compr, rots_compr)

def compr_to_perms(cube_compr):
    """Returns a tuple with the permutations of the compressed cube"""
    perms_compr = cube_compr[0]
    perms = []
    while perms_compr > 0 or len(perms) < 7:
        perms.append(perms_compr % 7)
        perms_compr //= 7
    return tuple(perms)

def compr_to_rots(cube_compr):
    """Returns a tuple with the rotations of the compressed cube"""
    rots_compr = cube_compr[-1]
    rots = []
    while rots_compr > 0 or len(rots) < 7:
        rots.append(rots_compr % 3)
        rots_compr //= 3
    return tuple(rots)

def move_compr(cube_compr, movement):
    """Moves a compressed version of a cube"""
    assert movement in possible_moves, "Invalid movement"

    perms_compr = cube_compr[0]
    rots_compr = cube_compr[-1]

    changes = changes_by_move[movement]
    affected_or = changes[0]
    change_or = changes[1]
    change_perm = changes[2]

    new_perm = 0
    for v in range(7):
        i = perms_compr % 7  # the index v (vertex) in the perms list has element i (position)
        new_perm += change_perm[i] * 7**v  # change the position of the vertex
        perms_compr //= 7
        if i in affected_or:  # if the position is affected, change the orientation
            old_idx = (rots_compr//3**v) % 3
            new_idx = change_or[old_idx]
            rots_compr += (new_idx - old_idx) * 3**v  # this won't interfere with other operations

    return (new_perm, rots_compr)

# Set-up 1: dictionary mapping moves to next moves that would be non-redundant
possible_moves = ["R", "R'", "R2", "U", "U'", "U2", "F", "F'", "F2"]

next_possible_moves = {move: set() for move in possible_moves}
for i, move1 in enumerate(possible_moves):
    for j, move2 in enumerate(possible_moves):
        if i//3 != j//3:  # if not the same letter
            next_possible_moves[move1].add(move2)
next_possible_moves[None] = set(possible_moves)  # add a None move to start

# Set-up 2: dictionary mapping moves to how to change the orientation and position of each vertex
changes_by_move = {
    "R": ((1, 2, 5, 6), (1, 0, 2), (0, 2, 6, 3, 4, 1, 5)),   # if an R or R', for the vertices 1, 2, 5, 6:
    "R'": ((1, 2, 5, 6), (1, 0, 2), (0, 5, 1, 3, 4, 6, 2)),  # change orientation 0 to 1, 1 to 0, and leave 2
    "U": ((0, 1, 2, 3), (0, 2, 1), (3, 0, 1, 2, 4, 5, 6)),
    "U'": ((0, 1, 2, 3), (0, 2, 1), (1, 2, 3, 0, 4, 5, 6)),  # if R, leave vertex 0, move vertex 1 to position 2,
    "F": ((0, 1, 4, 5), (2, 1, 0), (1, 5, 2, 3, 0, 4, 6)),   # move vertex 2 to position 6, etc
    "F'": ((0, 1, 4, 5), (2, 1, 0), (4, 0, 2, 3, 5, 1, 6)),
    "R2": (set(), set(), (0, 6, 5, 3, 4, 2, 1)),
    "U2": (set(), set(), (2, 3, 0, 1, 4, 5, 6)),
    "F2": (set(), set(), (5, 4, 2, 3, 1, 0, 6)),  # don't change orientation
}

def solve(cube1, cube2):
    """Returns the shortest list of moves to go from cube1 to cube2"""
    def inv_move(movement):
        """Invert a single movement"""
        if len(movement) == 1:  # if a single letter
            return movement + "'"
        elif movement[-1] == "2":  # if a double move
            return movement
        else:  # if a ' move
            return movement[0]
    def reverse_moves(movements):
        """Invert a list of movements"""
        # reverse each move and put them in the inversed order
        movements = [inv_move(movement) for movement in reversed(movements)]
        return movements
    def backtrack(node_compr, start_compr, parents):
        """Returns a list of moves to get from start to node given the parents dic"""
        solution = []
        while node_compr != start_compr:  # backtracking
            move = parents[node_compr]  # last move done
            solution.append(move)  # add it
            # go back one move
            reversed_move = inv_move(move)
            node_compr = move_compr(node_compr, reversed_move)
        solution.reverse()
        return solution

    def BFS(start_compr, end_condition, max_length):
        """Runs BFS, returns the path from start to the first node s.t. end_condition(node.compr()) == True
        Stops the recursion at move sequences of length max_length
        If end_condition is never True, then returns the parents dictionary"""
        parents = {start_compr: None}  # last move done to get there, is never rewritten
        queue = deque([(start_compr, 0)])  # BFS queue: stores (cube, # of moves to get there)

        while queue:  # while non-empty
            cube_compr, prev_depth = queue.popleft()  # pop and return first element
            if end_condition(cube_compr):  # we solved the cube, so return the solution
                return cube_compr, backtrack(cube_compr, start_compr, parents)

            if prev_depth <= max_length - 1:
                prev_move = parents[cube_compr]  # last move done
                for move in next_possible_moves[prev_move]:  # for all possible next moves
                    new_cube_compr = move_compr(cube_compr, move)
                    if new_cube_compr not in parents:
                        parents[new_cube_compr] = move  # record parent pointers
                        queue.append((new_cube_compr, prev_depth + 1))  # add to queue on the right
        return parents

    cube1_compressed = cube1.compress()
    cube2_compressed = cube2.compress()
    first_try = BFS(cube1_compressed, lambda node_compr: node_compr == cube2_compressed, 5)
    if isinstance(first_try, list):  # if a path was found, return it
        _, first_path = first_try
        return first_path
    # otherwise, first_try is the parents dictionary
    first_parents = first_try
    # get the path from cube2 to node
    try:
        node_compr, second_path = BFS(cube2_compressed, lambda node_compr: node_compr in first_parents, 6)
    except ValueError:
        return "<No solution was found>"
    first_path = backtrack(node_compr, cube1_compressed, first_parents)  # path from cube1 to node
    return first_path + reverse_moves(second_path)

# import time
# cube1 = Rubik2()
# cube2 = cube1.moves(["F", "R", "U'", "R'", "U'", "R", "U", "R'", "F'", "R", "U", "R'", "U'", "R'", "F", "R", "F'"])
# # cube2 = cube1.moves(["R", "U'", "F", "R", "U'", "R2", "U'", "F'"])
# t0 = time.perf_counter()
# solution = solve(cube1, cube2)
# t1 = time.perf_counter()
# print(f'A solution is {solution} and was found in {t1 - t0}s')
# # It's the same speed as DFS


# WEDGE PRODUCT CLASS ----------------------------------------------------------------------

def has_two_equal_elements(array):
    """True if a sorted array has two equal elements"""
    answer = False
    for i in range(len(array) - 1):
        if array[i] == array[i + 1]:
            answer = True
    return answer

def bubble_sort(array):
    """Scans the array and swaps if needed
    Returns the sorted array and the sign of the permutation"""
    array = array.copy()
    n = len(array)
    def swap(a, b):
        """Updates array"""
        array[a], array[b] = array[b], array[a]
        return None

    sign = 1
    for i in range(n - 1):  # sort from 0 to n - i - 1
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                swap(j, j + 1)
                sign *= -1

    return array, sign

class AltForm():
    """A tree, right will represent using a term and left not using it"""
    def __init__(self, val=None):
        """Initialize as a 0-form"""
        self.val = val
        self.left = None
        self.right = None
    def add_term(self, term):
        """indices is a sorted list of positive ints without repetitions
        adds the term corresponding to the indices
        edits the Tree instance"""
        val, indices = term.val, term.indices  # sorted indices
        if len(indices) == 0:
            if self.val == None: self.val = val
            else: self.val += val
            return self
        node = self  # iterator
        for i in range(1, indices[-1] + 1):  # all possible indices
            if i in indices:  # if index is there, go right
                if node.right is None:
                    node.right = AltForm()
                node = node.right  # go down the tree
            else:  # if index is not there, go left
                if node.left is None:
                    node.left = AltForm()
                node = node.left
        if node.val is None: node.val = val
        else: node.val += val
        return self
    def __add__(self, other):
        """other can be AltForm or int, float, complex, creates new instance"""
        if not isinstance(other, AltForm):
            other = AltForm(other)
        new_form = AltForm()
        for element in other:
            one_zeros, val = element
            term = AltTerm(val, [i + 1 for i, is_there in enumerate(one_zeros) if is_there == 1])
            new_form.add_term(term)
        for element in self:
            one_zeros, val = element
            term = AltTerm(val, [i + 1 for i, is_there in enumerate(one_zeros) if is_there == 1])
            new_form.add_term(term)
        return new_form
    def __radd__(self, other):
        """other can be AltForm or int, float, complex"""
        return self + other
    def __mul__(self, other):  # wedge product
        """other can be AltForm or int, float, complex, creates new instance"""
        new_form = AltForm()
        if not isinstance(other, AltForm):
            if self.val is not None: new_form.val = other * self.val
            if self.left is not None: new_form.left = self.left * other
            if self.right is not None: new_form.right = self.right * other
            return new_form
        else:
            raise NotImplementedError
    def __rmul__(self, other):
        if not isinstance(other, AltForm):
            return self * other
    def __iter__(self, prefix=[]):
        """Returns an iterator ([list of 0, 1 depending on left or right], value there)
        prefix is only used for recursion and represents the decisions to get here"""
        if self.val is not None: yield (prefix, self.val)
        if self.left is not None: yield from self.left.__iter__(prefix + [0])
        if self.right is not None: yield from self.right.__iter__(prefix + [1])
    def __str__(self):
        answer = ""
        if self.val is not None: answer += str(self.val)
        for element in self:
            one_zeros, val = element
            if len(one_zeros) == 0:
                continue
            term = AltTerm(val, [i + 1 for i, is_there in enumerate(one_zeros) if is_there == 1])
            if term.val < 0:
                sign = "-"
                term.val *= -1
            else:
                sign = "+"
            answer += f" {sign} {term}"
        return answer

class AltTerm():
    """Stores a term of the form: val dx_i1 ^ dx_i2 ^ ..."""
    def __init__(self, val, indices):
        """val: int, float, complex
        indices: list of indices (positive ints), can be in any order and there may be repetitions"""
        indices = list(indices)
        indices, sign = bubble_sort(indices)  # sort and get sign
        if has_two_equal_elements(indices):  # if repetition, return the 0 form
            self.val = 0
            self.indices = []
        else:
            self.val = sign * val
            self.indices = indices
    def __str__(self):
        answer = str(self.val)
        for idx in self.indices:
            answer += f" e{idx} ^"
        if self.indices == []:
            return answer
        else:
            return answer[:-2]

# w = AltForm(3)
# w.add_term(AltTerm(7, (3, 1)))
# w.add_term(AltTerm(2, (1, 3)))
# w.add_term(AltTerm(2, (1, 2)))
# k = AltForm()
# k.add_term(AltTerm(1, (1, )))
# k.add_term(AltTerm(-1, (2, )))
# print(w)
# print(k)
# print(2 * w + (-1)*k)

