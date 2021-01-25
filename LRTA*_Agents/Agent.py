
from World import World
from LRTAStarAlgorithm import LTRAStarAlgorithm

def print_delays_per_agent(path):
    print("{0} delay for Nick".format(path[1][0][2]*30))
    print("{0} delay for Ann".format(path[1][1][2]*30))
    print("{0} delay for Tasos".format(path[1][2][2]*30))
    print("{0} delay for Mary".format(path[1][3][2]*30))
    print("{0} delay for George".format(path[1][4][2]*30))

if __name__ == "__main__":


    delays = (30, 60, 90, 120)

    # (N, A, T, M, G)
    startingPoint = (0, 0, 0, 0, 0)
    targetPoint = (2, 2, 2, 2, 2)

    # instantiate world as an object to be used during value iteration
    w = World( start=startingPoint, end=targetPoint, delays=delays)

    # state = [(0, 1, 1, 1), (1, 4, 0, 2), (2, 1, 1, 1), (3, 1, 1, 1), (4, 2, 0, 1)]

    # a = w.nextMoves(state)

    path = LTRAStarAlgorithm(w)

    print_delays_per_agent(path)
