import sudoku_rubiks as file
from collections import defaultdict

# first test:
#
#      0  0
#    a      b
#  1          1
#     1  c  0

def compare_objects(dic1, dic2):
    """Objects should have an .items() method"""
    for k, v1 in dic1.items():
        v2 = dic2[k]
        assert v1 == v2, f"Error in {k}"


def test_cube_1():
    file.init_graph(
        file.PositionVertex, vertex_groups=[
            ["a", "b"], ["b", "c"], ["c", "a"],
        ]
    )
    Position1 = file.PositionVertex("a", ["a", "b"], vertex_id=0, group_id=0)
    Position2 = file.PositionVertex("b", ["a", "b"], vertex_id=1, group_id=0)
    Position3 = file.PositionVertex("b", ["b", "c"], vertex_id=0, group_id=1)
    Position4 = file.PositionVertex("c", ["b", "c"], vertex_id=1, group_id=1)
    Position5 = file.PositionVertex("c", ["c", "a"], vertex_id=0, group_id=2)
    Position6 = file.PositionVertex("a", ["c", "a"], vertex_id=1, group_id=2)

    file.init_graph(
        file.PieceVertex, vertex_groups=[
            [0, 0], [0, 1], [1, 1],
        ]
    )
    Piece1 = file.PieceVertex(0, [0, 0], vertex_id=0, group_id=0)
    Piece2 = file.PieceVertex(0, [0, 0], vertex_id=1, group_id=0)
    Piece3 = file.PieceVertex(0, [0, 1], vertex_id=0, group_id=1)
    Piece4 = file.PieceVertex(1, [0, 1], vertex_id=1, group_id=1)
    Piece5 = file.PieceVertex(1, [1, 1], vertex_id=0, group_id=2)
    Piece6 = file.PieceVertex(1, [1, 1], vertex_id=1, group_id=2)

    weak_links, strong_links = file.initialize_links({})

    assert Position1.neighbors() == {Position6}
    assert Position2.neighbors() == {Position3}
    assert Position3.neighbors() == {Position2}
    assert Position4.neighbors() == {Position5}
    assert Position5.neighbors() == {Position4}
    assert Position6.neighbors() == {Position1}

    assert Piece1.neighbors() == {Piece3}
    assert Piece2.neighbors() == {Piece3}
    assert Piece3.neighbors() == {Piece1, Piece2}
    assert Piece4.neighbors() == {Piece5, Piece6}
    assert Piece5.neighbors() == {Piece4}
    assert Piece6.neighbors() == {Piece4}

    assert strong_links == file.StrongLinks()
    expected_weak_links = file.WeakLinks({
        Piece1: {Position1, Position2, Position3, Position4, Position5, Position6},
        Piece2: {Position1, Position2, Position3, Position4, Position5, Position6},
        Piece3: {Position1, Position2, Position3, Position4, Position5, Position6},
        Piece4: {Position1, Position2, Position3, Position4, Position5, Position6},
        Piece5: {Position1, Position2, Position3, Position4, Position5, Position6},
        Piece6: {Position1, Position2, Position3, Position4, Position5, Position6},
        Position1: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
        Position2: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
        Position3: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
        Position4: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
        Position5: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
        Position6: {Piece1, Piece2, Piece3, Piece4, Piece5, Piece6},
    })
    compare_objects(weak_links, expected_weak_links)

    assert list(Position1.rotations()) == [Position1, Position2]
    assert list(Position2.rotations()) == [Position2, Position1]
    assert list(Position3.rotations()) == [Position3, Position4]
    assert list(Position4.rotations()) == [Position4, Position3]
    assert list(Position5.rotations()) == [Position5, Position6]
    assert list(Position6.rotations()) == [Position6, Position5]
    assert list(Piece1.rotations()) == [Piece1, Piece2]
    assert list(Piece2.rotations()) == [Piece2, Piece1]
    assert list(Piece3.rotations()) == [Piece3, Piece4]
    assert list(Piece4.rotations()) == [Piece4, Piece3]
    assert list(Piece5.rotations()) == [Piece5, Piece6]
    assert list(Piece6.rotations()) == [Piece6, Piece5]

    weak_links_1, strong_links_1 = file.strengthen_link(
        weak_links, strong_links, Piece1, Position1
    )
    weak_links_2, strong_links_2 = file.strengthen_link(
        weak_links, strong_links, Position1, Piece1
    )
    assert weak_links_1 == weak_links_2
    assert strong_links_1 == strong_links_2
    assert strong_links_1 == file.StrongLinks(
        {Piece1: Position1, Piece2: Position2},
        {Position1: Piece1, Position2: Piece2},
    )
    expected_weak_links_1 = file.WeakLinks({
        Piece1: set(),
        Piece2: set(),
        Piece3: {Position4, Position5},
        Piece4: {Position3, Position6},
        Piece5: {Position3, Position4, Position5, Position6},
        Piece6: {Position3, Position4, Position5, Position6},
        Position1: set(),
        Position2: set(),
        Position3: {Piece4, Piece5, Piece6},
        Position4: {Piece3, Piece5, Piece6},
        Position5: {Piece3, Piece5, Piece6},
        Position6: {Piece4, Piece5, Piece6},
    })
    compare_objects(weak_links_1, expected_weak_links_1)

    weak_links_3, strong_links_3 = file.strengthen_link(
        weak_links_1, strong_links_1, Piece3, Position4,
    )
    assert strong_links_3 == file.StrongLinks(
        {Piece1: Position1, Piece2: Position2, Piece3: Position4, Piece4: Position3},
        {Position1: Piece1, Position2: Piece2, Position3: Piece4, Position4: Piece3},
    )
    print(weak_links_3)
    assert weak_links_3 == file.WeakLinks({
        Piece5: {Position5, Position6},
        Piece6: {Position5, Position6},
        Position5: {Piece5, Piece6},
        Position6: {Piece5, Piece6},
    })

    weak_links_4, strong_links_4 = file.strengthen_link(
        weak_links_3, strong_links_3, Piece5, Position5,
    )
    assert strong_links_4 == file.StrongLinks(
        {
            Piece1: Position1, Piece2: Position2, Piece3: Position4,
            Piece4: Position3, Piece5: Position5, Piece6: Position6
        },
        {
            Position1: Piece1, Position2: Piece2, Position3: Piece4,
            Position4: Piece3, Position5: Piece5, Position6: Piece6
        },
    )
    assert weak_links_4 == file.WeakLinks({})

    file.PositionVertex.graph = defaultdict(set)
    file.PieceVertex.graph = defaultdict(set)


# second test:
#
#        d: 0   
#                 
#  a: 0        c: 2
#   1             
#     2  b: 1   
#  
#  (1, 2), (2, 0), (0, 1), (0, 1)


def test_cube_2():
    file.init_graph(
        file.PositionVertex, vertex_groups=[
            ["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"],
        ]
    )
    Position1 = file.PositionVertex("a", ["a", "b"], vertex_id=0, group_id=0)
    Position2 = file.PositionVertex("b", ["a", "b"], vertex_id=1, group_id=0)
    Position3 = file.PositionVertex("b", ["b", "c"], vertex_id=0, group_id=1)
    Position4 = file.PositionVertex("c", ["b", "c"], vertex_id=1, group_id=1)
    Position5 = file.PositionVertex("c", ["c", "d"], vertex_id=0, group_id=2)
    Position6 = file.PositionVertex("d", ["c", "d"], vertex_id=1, group_id=2)
    Position7 = file.PositionVertex("d", ["d", "a"], vertex_id=0, group_id=3)
    Position8 = file.PositionVertex("a", ["d", "a"], vertex_id=1, group_id=3)
    Face1 = file.PositionVertex("a", ["a"], vertex_id=0, group_id=0)
    Face2 = file.PositionVertex("b", ["b"], vertex_id=0, group_id=1)
    Face3 = file.PositionVertex("c", ["c"], vertex_id=0, group_id=2)
    Face4 = file.PositionVertex("d", ["d"], vertex_id=0, group_id=3)

    file.init_graph(
        file.PieceVertex, vertex_groups=[
            [1, 2], [2, 0], [0, 1], [0, 1],
        ]
    )
    Piece1 = file.PieceVertex(1, [1, 2], vertex_id=0, group_id=0)
    Piece2 = file.PieceVertex(2, [1, 2], vertex_id=1, group_id=0)
    Piece3 = file.PieceVertex(2, [2, 0], vertex_id=0, group_id=1)
    Piece4 = file.PieceVertex(0, [2, 0], vertex_id=1, group_id=1)
    Piece5 = file.PieceVertex(0, [0, 1], vertex_id=0, group_id=2)
    Piece6 = file.PieceVertex(1, [0, 1], vertex_id=1, group_id=2)
    Piece7 = file.PieceVertex(0, [0, 1], vertex_id=0, group_id=3)
    Piece8 = file.PieceVertex(1, [0, 1], vertex_id=1, group_id=3)
    FixVal1 = file.PieceVertex(0, [0], vertex_id=0, group_id=0)
    FixVal2 = file.PieceVertex(1, [1], vertex_id=0, group_id=1)
    FixVal3 = file.PieceVertex(2, [2], vertex_id=0, group_id=2)
    FixVal4 = file.PieceVertex(0, [0], vertex_id=0, group_id=3)

    weak_links, strong_links = file.initialize_links({
        "a": [0], "b": [1], "c": [2], "d": [0],
    })

    assert strong_links == file.StrongLinks(
        {
            FixVal1: Face1, FixVal2: Face2, FixVal3: Face3, FixVal4: Face4,
        },
        {
            Face1: FixVal1, Face2: FixVal2, Face3: FixVal3, Face4: FixVal4,
        },
    )
    expected_weak_links = file.WeakLinks({
        Piece1: {Position1, Position4, Position5, Position7, Position8},
        Piece2: {Position2, Position3, Position6, Position8, Position7},
        Piece3: {Position1, Position3, Position6},
        Piece4: {Position2, Position4, Position5},
        Piece5: {Position2, Position3, Position5},
        Piece6: {Position1, Position4, Position6},
        Piece7: {Position2, Position3, Position5},
        Piece8: {Position1, Position4, Position6},
        Position1: {Piece1, Piece3, Piece6, Piece8},
        Position2: {Piece2, Piece4, Piece5, Piece7},
        Position3: {Piece2, Piece3, Piece5, Piece7},
        Position4: {Piece1, Piece4, Piece6, Piece8},
        Position5: {Piece1, Piece4, Piece5, Piece7},
        Position6: {Piece2, Piece3, Piece6, Piece8},
        Position7: {Piece1, Piece2},
        Position8: {Piece1, Piece2},
    })
    compare_objects(weak_links, expected_weak_links)

    weak_links_1, strong_links_1 = file.strengthen_link(
        weak_links, strong_links, Piece1, Position1
    )
    weak_links_2, strong_links_2 = file.strengthen_link(
        weak_links, strong_links, Position1, Piece1
    )
    assert weak_links_1 == weak_links_2
    assert strong_links_1 == strong_links_2
    assert strong_links_1 == file.StrongLinks(
        {
            FixVal1: Face1, FixVal2: Face2, FixVal3: Face3, FixVal4: Face4,
            Piece1: Position1, Piece2: Position2
        },
        {
            Face1: FixVal1, Face2: FixVal2, Face3: FixVal3, Face4: FixVal4,
            Position1: Piece1, Position2: Piece2
        },
    )
    expected_weak_links_1 = file.WeakLinks({
        Piece1: set(),
        Piece2: set(),
        Piece3: {Position6},
        Piece4: {Position5},
        Piece5: {Position3, Position5},
        Piece6: {Position4, Position6},
        Piece7: {Position3, Position5},
        Piece8: {Position4, Position6},
        Position1: set(),
        Position2: set(),
        Position3: {Piece5, Piece7},
        Position4: {Piece6, Piece8},
        Position5: {Piece4, Piece5, Piece7},
        Position6: {Piece3, Piece6, Piece8},
        Position7: set(),
        Position8: set(),
    })
    compare_objects(weak_links_1, expected_weak_links_1)

    file.main(weak_links, strong_links)
