from Node import Node
import random


def get_min_from(open_list):
    current_node = open_list[0]
    for index, item in enumerate(open_list):
        if item.f < current_node.f:
            current_node = item

    min_elements = []
    for index, item in enumerate(open_list):
        if current_node.f == item.f:
            min_elements.append((index, item))

    random_min = random.choice(min_elements)
    return random_min[1]

def LTRAStarAlgorithm(world):

    has_not_converged = True
    second_to_last_path = []

    # Create start and end node
    s = [
         (world.isNick,     world.is_at_home, 0, 0),
         (world.isAnn,      world.is_at_home, 0, 0),
         (world.isTasos,    world.is_at_home, 0, 0),
         (world.isMary,     world.is_at_home, 0, 0),
         (world.isGeorge,   world.is_at_home, 0, 0)
         ]

    start_node = Node(None, s)
    start_node.g = start_node.h = start_node.f = 0

    # Initialize both open and closed list
    open_list = []
    current_node = start_node

    while has_not_converged:

        while world.time <= 14:

            if world.is_end_state(current_node.state):
                path = []
                current = current_node
                while current is not None:
                    # print(current.state)
                    path.append(current.state)
                    current = current.parent
                last_path = path[::-1] # Return reversed path
                break

            # Generate children
            children = []
            for move in world.next_moves(current_node.state):  # get

                new_node = Node(current_node, move)
                children.append(new_node)

            # Loop through children
            for index, child in enumerate(children):
                # Create the f, g, and h values

                if world.graph_contains(child.state):
                    node = world.get_node_from_graph(child.state)
                    node.g = child.getSumDelays()
                    node.f = child.g + child.h
                    world.update_graph_node(node)
                    children[index] = node
                else:
                    child.g = child.getSumDelays()
                    child.f = child.g + child.h
                    world.append_node_to_graph(child)

            previous_node = current_node
            current_node = get_min_from(children)
            previous_node.h = max(previous_node.h, current_node.f)

            world.update_graph_node(previous_node)

            # world.update_graph(children)
            world.change_turn()
            # print(world.time)

        if second_to_last_path == last_path:
            has_not_converged = False
        else:
            second_to_last_path = last_path
            current_node = start_node
            world.time = 0

    return second_to_last_path

