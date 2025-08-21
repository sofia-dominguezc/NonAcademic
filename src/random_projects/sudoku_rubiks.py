from __future__ import annotations
import sys
from collections import defaultdict
from typing import Union, Optional, TypeVar, Generic, Iterator

T = TypeVar('T')
num_pieces = 6 * 1 + 12 * 2 + 8 * 3

# Here I solve the sudoku rubik's cube
# I implement a graph data structure for the corners and then
#   for the edges.
# The logic is to consider what positions each piece can go to
#   and what pieces each position can contain, up to orientation.
# I keep a bidict that keeps track of these "weak links"
# Then I do a BFS style algorithm to iteratively make a weak link
#   small until are all strong and we have a 1-1 mapping which
#   corresponds to a solution (for either the corners or edges)
# It's possible to introduce initial conditions, so the edges can be
#   solved for each corner solution (or viceversa)

def dict_print(dic):
    """Pretry print a dictionary"""
    return '\n'.join(f"{repr(k)}: {repr(v)}" for k, v in dic.items())

class Vertex:
    """Parent class to represent pieces (corners/edges) and positions"""

    def __init__(
        self,
        value: Union[str, int],
        group: Union[list[str], list[int]],
        vertex_id: Optional[int] = None,
        group_id: Optional[int] = None,
    ):
        """
        value: int for pieces, str for positions
        group: values of all elements of the same piece/position. Should
            be in anti-clockwise order and consistent among different calls.
        vertex_id: index of element in group
        group_id: unique integer group identifier
            - For positions, it's the position index
            - For pieces, it's the piece index
            - For fixed values, it's the face index
            - For faces, it's the fixed value index
        vertex_id, group_id, class_id, size identify the instance
        """
        self.value = value
        self.group = group
        self.vertex_id = vertex_id
        self.group_id = group_id
        self.size = len(group)

    def _get_rotations(self):
        """All rotations of the vertex in order"""
        out: list[Vertex] = []
        for i in range(self.size):
            j = (self.vertex_id + i) % self.size
            out.append(self.__class__(self.group[j], self.group, j, self.group_id))
        self._rotations = out

    def rotations(self):
        if not hasattr(self, "_rotations"):
            self._get_rotations()
        return self._rotations

    def neighbors(self) -> set[Vertex]:
        return self.graph[self]

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Vertex)
            and (self.vertex_id == other.vertex_id)
            and (self.group_id == other.group_id)
            and (self.class_id == other.class_id)
            and (self.size == other.size)
        )

    def __str__(self) -> str:
        return str([r.value for r in self.rotations()])

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({repr(self.value)}, {self.group}, {self.vertex_id}, {self.group_id})"

    def __hash__(self) -> int:
        return hash((self.class_id, self.vertex_id, self.group_id, self.size))

    @property
    def graph(self) -> dict[Vertex, set[Vertex]]:
        raise NotImplementedError

    @property
    def class_id(self) -> int:
        raise NotImplementedError

class PositionVertex(Vertex):
    """Class to represent positions of corners and edges"""
    obj_name = "Position"
    class_id = 0
    graph: dict[PositionVertex, set[PositionVertex]] = defaultdict(set)

class PieceVertex(Vertex):
    """Class to represent corners and edges"""
    obj_name = "Piece"
    class_id = 1
    graph: dict[PieceVertex, set[PieceVertex]] = defaultdict(set)


class WeakLinks:
    """
    Undirected graph class whose edges can be updated.
    graph[vertex] is its set of neighbors.
    """
    def __init__(self, adjacency: Optional[dict] = None):
        self._adjacency: dict[Vertex, set[Vertex]] = adjacency or defaultdict(set)

    def __setitem__(self, key, value):
        """graph[k] = v adds the edge (k, v)"""
        self._adjacency[key].add(value)
        self._adjacency[value].add(key)

    def __delitem__(self, key):
        """Removes all edges of a vertex if it exists"""
        neighbors = self._adjacency.get(key, set())
        for neighbor in tuple(neighbors):
            self.remove_edge(key, neighbor)
        if key in self._adjacency and not self._adjacency[key]:
            del self._adjacency[key]

    def remove_edge(self, v1, v2):
        """Removes the edge (v1, v2) if it exists"""
        if v1 in self._adjacency:
            self._adjacency[v1].discard(v2)
            if not self._adjacency[v1]:
                del self._adjacency[v1]
        if v2 in self._adjacency:
            self._adjacency[v2].discard(v1)
            if not self._adjacency[v2]:
                del self._adjacency[v2]

    def copy(self):
        copy = WeakLinks()
        copy._adjacency = {k: v.copy() for k, v in self._adjacency.items()}
        return copy

    def __getitem__(self, key):
        return self._adjacency.get(key, set())

    def __contains__(self, key):
        return key in self._adjacency

    def __iter__(self):
        return iter(self._adjacency)

    def items(self):
        return self._adjacency.items()

    def items(self):
        return self._adjacency.items()

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{dict_print(self._adjacency)}\n)"

    def __eq__(self, other):
        return isinstance(other, WeakLinks) and self._adjacency == other._adjacency


class StrongLinks:
    def __init__(self, fwd: Optional[dict] = None, bwd: Optional[dict] = None):
        self._fwd: dict[PieceVertex, PositionVertex] = fwd or {}
        self._bwd: dict[PositionVertex, PieceVertex] = bwd or {}

    def __setitem__(self, key, value):
        """key and value must not already be in the object"""
        assert (
            isinstance(key, Vertex)
            and isinstance(value, Vertex)
            and key.__class__ != value.__class__
        ), "StrongLinks should be used for links Piece <-> Position"
        if isinstance(key, PositionVertex):
            self.__setitem__(value, key)
            return
        self._fwd[key] = value
        self._bwd[value] = key

    def __getitem__(self, item):
        # Try forward, then backward mapping
        if isinstance(item, PieceVertex):
            return self._fwd[item]
        elif isinstance(item, PositionVertex):
            return self._bwd[item]
        else:
            raise ValueError(f"{item} isn't a Vertex object")

    def copy(self):
        return StrongLinks(self._fwd.copy(), self._bwd.copy())

    def __contains__(self, item):
        return item in self._fwd or item in self._bwd

    def __len__(self):
        return len(self._fwd)

    def keys(self):
        return self._fwd.keys()

    def values(self):
        return self._fwd.values()

    def items(self):
        return self._fwd.items()

    def __repr__(self):
        return f'{self.__class__.__name__}(\n{dict_print(self._fwd)}\n\n{dict_print(self._bwd)}\n)'

    def __eq__(self, other):
        return isinstance(other, StrongLinks) and (self._fwd == other._fwd)


def init_graph(
        subclass: type,
        vertex_groups: Union[list[list[str]], list[list[int]]],
) -> None:
    """
    Initializes graph in the subclass from the given vertex groups.
    Should only be called once for PositionVertex and PieceVertex.
    """
    subclass.graph = defaultdict(set)
    # initialize empty graph
    for group_id, vertex_group in enumerate(vertex_groups):
        for vertex_id, vertex_value in enumerate(vertex_group):
            vertex = subclass(vertex_value, vertex_group, vertex_id, group_id)
            subclass.graph[vertex] = set()
    # add edges
    for vertex1 in subclass.graph:
        for vertex2 in subclass.graph:
            if vertex2.value == vertex1.value and vertex2.group_id != vertex1.group_id:
                subclass.graph[vertex1].add(vertex2)


def initialize_links(
        fixed_values: dict[str, list[int]],
) -> tuple[WeakLinks, StrongLinks]:
    """
    Initializes graph of weak and strong links between pieces and positions and
        updates the piece and position graphs.
    A piece has a weak link to a position if putting it there would not
    cause a direct contradiction (two equal numbers in the same face).
    This is equivalent to:
        a piece has a weak link to a position if
        - neither has a strong link, and
        - no rotation has neighbors with a strong link to each other
    Args:
        fixed_values: name (e.g. 'U') -> values in that face (e.g. the center)
    """
    strong_links = StrongLinks()
    # Set initial conditions
    original_positions = tuple(PositionVertex.graph)
    original_pieces = tuple(PieceVertex.graph)
    for id_face, (face, face_fixed_vals) in enumerate(fixed_values.items()):
        for id_num, fixed_val in enumerate(face_fixed_vals):
            # Add face node to position graph
            face_vertex = PositionVertex(face, [face], vertex_id=id_num, group_id=id_face)
            for pos in original_positions:
                if pos.value == face:
                    pos.graph[pos].add(face_vertex)
                    pos.graph[face_vertex].add(pos)
            # Add fixed value node in pieces graph
            fixed_val_vertex = PieceVertex(fixed_val, [fixed_val], vertex_id=id_num, group_id=id_face)
            for piece in original_pieces:
                if piece.value == fixed_val:
                    piece.graph[piece].add(fixed_val_vertex)
                    piece.graph[fixed_val_vertex].add(piece)
            # Connect face and fixed value nodes
            strong_links[fixed_val_vertex] = face_vertex

    # Set initial weak links
    secondary_non_weak_links = WeakLinks()  # filter the neighbor condition
    for piece, pos in strong_links.items():
        for piece_neigh in piece.neighbors():
            for pos_neigh in pos.neighbors():
                for piece_neigh_rot, pos_neigh_rot in zip(piece_neigh.rotations(), pos_neigh.rotations()):
                    secondary_non_weak_links[piece_neigh_rot] = pos_neigh_rot
    weak_links = WeakLinks()
    for piece in PieceVertex.graph:
        if piece in strong_links:  # filter the strong link condition
            continue
        for pos in PositionVertex.graph:
            if pos in strong_links:
                continue
            if pos in secondary_non_weak_links[piece]:
                continue
            if piece.size != pos.size:
                continue
            weak_links[piece] = pos

    return weak_links, strong_links


def choose_next(weak_links: WeakLinks, strong_links: StrongLinks) -> Union[Vertex, bool]:
    """
    Get the position/piece that has the least number of weak links.
    If a position has no weak or strong links, it returns None.
    NOTE: a smarter data structure like a heap can optimize this a lot
    """
    min_links = float('inf')
    chosen_element = len(strong_links) == num_pieces
    for element, neighbors in weak_links.items():
        if element in strong_links:  # element has been matched
            continue
        if not neighbors:  # element can't be matched
            return False
        if len(neighbors) < min_links:
            min_links = len(neighbors)
            chosen_element = element
    return chosen_element


def strengthen_link(
    weak_links: WeakLinks,
    strong_links: StrongLinks,
    element: Vertex,
    other: Vertex,
) -> tuple[WeakLinks, StrongLinks]:
    """
    Chooses a weak link of element and makes it strong.
    Other orientations of the same group are also made into strong links.
    """
    weak_links = weak_links.copy()
    strong_links = strong_links.copy()
    for element_rot, other_rot in zip(element.rotations(), other.rotations()):
        # delete their weak links
        del weak_links[element_rot]
        del weak_links[other_rot]
        # add their strong links
        strong_links[element_rot] = other_rot
        # delete all other weak links
        for element_neigh in element_rot.neighbors():
            for other_neigh in other_rot.neighbors():
                for element_neigh_rot, other_neigh_rot in zip(element_neigh.rotations(), other_neigh.rotations()):
                    weak_links.remove_edge(element_neigh_rot, other_neigh_rot)
    return weak_links, strong_links


def show_pairing(strong_links: StrongLinks) -> None:
    print("\nPairing:")
    for piece, pos in strong_links.items():
        if piece.size == 1:
            continue
        if pos.vertex_id == 0:
            print(f"{pos}: {piece}")


def check_solution(strong_links: StrongLinks) -> bool:
    """Check that the total orientation of the corners sums to zero"""
    orientation_sum = 0
    for piece, pos in strong_links.items():
        if pos.vertex_id == 0 and piece.size == 3:
            orientation_sum += piece.vertex_id
    return orientation_sum % 3 == 0


def main(weak_links: WeakLinks, strong_links: StrongLinks):
    """
    Does a DFS style loop.
    Chooses element with least number of weak links and makes it strong.
    If no weak links remaining and strong_links is not full, undo previous move.
    Prints progress dynamically by line, showing branch numbers.
    """
    progress_lines = []

    def print_progress(depth, element, ramification_idx, num_ramifications):
        # Remove lines deeper than current depth (when backtracking)
        while len(progress_lines) > depth:
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            progress_lines.pop()
        indent = '    ' * depth
        # ramification_idx is 1-based, so we print 1/3, 2/3, etc
        line = f"{indent}{depth+1}: {element} -- ramifications: {ramification_idx}/{num_ramifications}"
        print(line)
        progress_lines.append(line)
        sys.stdout.flush()

    def recursive(weak_links: WeakLinks, strong_links: StrongLinks, depth=0):
        if __name__ == "__main__":
            global start_time
            if time() - start_time > 5:
                raise TimeoutError
        element = choose_next(weak_links, strong_links)
        if not isinstance(element, Vertex):  # stop recursion
            was_matched = element
            if was_matched and check_solution(strong_links):
                global counter
                counter += 1
            return None

        weak_connections = tuple(weak_links[element])
        num_ramifications = len(weak_connections)
        for idx, other in enumerate(weak_connections, 1):
            print_progress(depth, element, idx, num_ramifications)
            new_weak_links, new_strong_links = strengthen_link(
                weak_links, strong_links, element, other
            )
            recursive(new_weak_links, new_strong_links, depth + 1)

        # Backtrack print: when a branch ends, remove the last line
        while len(progress_lines) > depth:
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            progress_lines.pop()
        sys.stdout.flush()

    recursive(weak_links, strong_links)


if __name__ == "__main__":
    init_graph(
        PositionVertex, vertex_groups=[
            ["U", "F", "R"], ["U", "R", "B"], ["U", "B", "L"], ["U", "L", "F"],
            ["D", "R", "F"], ["D", "F", "L"], ["D", "L", "B"], ["D", "B", "R"],
            ["U", "F"], ["U", "R"], ["U", "B"], ["U", "L"], ["F", "R"], ["B", "R"],
            ["B", "L"], ["F", "L"], ["D", "F"], ["D", "R"], ["D", "B"], ["D", "L"],
        ]
    )
    init_graph(
        PieceVertex, vertex_groups=[
            [8, 6, 9], [5, 2, 4], [5, 8, 2], [3, 7, 1],
            [7, 1, 3], [7, 6, 4], [4, 6, 9], [9, 5, 1],
            [8, 6], [3, 6], [2, 1], [2, 8], [5, 8], [7, 9],
            [7, 3], [9, 8], [9, 4], [5, 2], [4, 2], [3, 3],
        ]
    )
    weak_links, strong_links = initialize_links({
        "U": [1], "F": [1], "R": [7],
        "L": [4], "D": [6], "B": [5],
    })

    from time import time
    counter = 0
    start_time = time()
    import cProfile
    with cProfile.Profile() as pr:
        try:
            main(weak_links, strong_links)
        except TimeoutError:
            pass
    pr.dump_stats(".prof")
    print(f"Solutions found: {counter}")  # ~500 to ~1000 in 5s
