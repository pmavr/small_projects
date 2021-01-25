
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent, state):
        self.parent = parent

        # 0 <- home
        # 1 <- cafe
        # 2 <- cinema
        # state = [
        #           (Nick, home, delay, wait),
        #           (Ann, home, delay, wait),
        #           (Tasos, home, delay, wait),
        #           (Mary, home, delay, wait),
        #           (George, home, delay, wait)
        #           ]

        self.state = state

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.state == other.state

    def getSumDelays(self):
        sum = 0
        for person in self.state:
            sum += person[2]
        return sum

