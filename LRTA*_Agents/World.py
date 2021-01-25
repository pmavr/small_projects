
class World:

    is_at_home = 0
    is_waiting_at_home = 1
    is_going_to_cafe = 2
    is_waiting_at_cafe = 3
    is_at_cafe = 4
    is_waiting_at_cinema = 5
    is_at_cinema1 = 6
    is_at_cinema2 = 7

    is_ready_to_change_state = 1

    isNick = 0
    isAnn = 1
    isTasos = 2
    isMary = 3
    isGeorge = 4

    transTimeForNick = 1
    transTimeForAnn = 0
    transTimeForGeorge = 1
    transTimeForTasos = 2
    transTimeForMary = 2

    movieAirTime1 = 4
    movieAirTime2 = 6
    cinemaStay = 4
    cafeStay = 2



    def __init__(self, start, end, delays):

        # (
        # (Nick, home, delay, wait),
        # (Ann, home, delay),
        # (Tasos, home, delay),
        # (Mary, home, delay),
        # (George, home, delay)
        # )

        self.start = [
                     (self.isNick,     self.is_at_home, 0, 0),
                     (self.isAnn,      self.is_at_home, 0, 0),
                     (self.isTasos,    self.is_at_home, 0, 0),
                     (self.isMary,     self.is_at_home, 0, 0),
                     (self.isGeorge,   self.is_at_home, 0, 0)
                     ]

        # (2, 2, 2, 2, 2)   everyone is at the cinema
        self.end = end

        # delays = (30, 60, 90, 120, 180, 240)
        self.delays = delays

        self.graph = [[] for i in range(14)]

        self.time = 0

    def is_end_state(self, s):
        return s[0][1] >= self.is_at_cinema1 and \
               s[1][1] >= self.is_at_cinema1 and \
               s[2][1] >= self.is_at_cinema1 and \
               s[3][1] >= self.is_at_cinema1 and \
               s[4][1] >= self.is_at_cinema1

    def change_turn(self):
        self.time += 1

    def movie_doors_open(self):
        return self.time % 2 == 0

    def avail_seats_in_cafe(self, state):
        reservedSeats = 0
        for person in state:
            if person[1] == self.is_at_cafe:
                reservedSeats += 1
        return 2 - reservedSeats

    def avail_seats_in_cinema1(self, state):
        reservedSeats = 0
        for person in state:
            if person[1] == self.is_at_cinema1:
                reservedSeats += 1
        return 3 - reservedSeats

    def avail_seats_in_cinema2(self, state):
        reservedSeats = 0
        for person in state:
            if person[1] == self.is_at_cinema2:
                reservedSeats += 1
        return 3 - reservedSeats

    def get_start_state(self):
        return self.start

    def get_transition_states(self, person, state):

        # (agent, home, delay, wait)

        if person[1] == self.is_waiting_at_home:
            if person[3] == self.is_ready_to_change_state:
                if person[0] == self.isAnn:
                    if self.avail_seats_in_cafe(state) > 0:
                        tmp = (person[0], person[1] + 3, person[2], self.cafeStay)  # Ann gets to cafe
                    else:
                        tmp = (person[0], person[1] + 2, person[2] + 1, 1)  # Ann waits to enter cafe

                if person[0] == self.isNick:
                    tmp = (person[0], person[1] + 1, person[2], self.transTimeForNick)  # Nick departs for cafe

                if person[0] == self.isTasos:
                    tmp = (person[0], person[1] + 1, person[2], self.transTimeForTasos)  # Tasos departs for cafe

                if person[0] == self.isMary:
                    tmp = (person[0], person[1] + 1, person[2], self.transTimeForMary)  # Mary departs for cafe

                if person[0] == self.isGeorge:
                    tmp = (person[0], person[1] + 1, person[2], self.transTimeForGeorge)  # George departs for cafe

            else:  # still have to waiting at home
                tmp = (person[0], person[1], person[2], person[3] - 1)

        elif person[1] == self.is_going_to_cafe:
            if person[3] == self.is_ready_to_change_state:
                if self.avail_seats_in_cafe(state) > 0:
                    tmp = (person[0], person[1] + 2, person[2], self.cafeStay)
                else:  # will have to wait
                    tmp = (person[0], person[1] + 1, person[2] + 1, 1)
            else:  # still getting there
                tmp = (person[0], person[1], person[2], person[3] - 1)

        elif person[1] == self.is_waiting_at_cafe:
            if self.avail_seats_in_cafe(state) > 0:
                tmp = (person[0], person[1] + 1, person[2], self.cafeStay)
            else:  # will have to wait
                tmp = (person[0], person[1], person[2] + 1, 1)

        elif person[1] == self.is_at_cafe:
            if person[3] == self.is_ready_to_change_state:
                if self.time >= self.movieAirTime1 and self.movie_doors_open() and self.avail_seats_in_cinema1(state) > 0:
                    tmp = (person[0], person[1] + 2, person[2], self.cinemaStay)
                elif self.time >= self.movieAirTime2 and self.movie_doors_open() and self.avail_seats_in_cinema2(state) > 0:
                    tmp = (person[0], person[1] + 3, person[2], self.cinemaStay)
                else:  # will have to wait
                    tmp = (person[0], person[1] + 1, person[2] + 1, 1)
            else:  # still drinking coffee
                tmp = (person[0], person[1], person[2], person[3] - 1)

        elif person[1] == self.is_waiting_at_cinema:
            if self.time >= self.movieAirTime1 and self.movie_doors_open() and self.avail_seats_in_cinema1(state) > 0:
                tmp = (person[0], person[1] + 1, person[2], self.cinemaStay)
            elif self.time >= self.movieAirTime2 and self.movie_doors_open() and self.avail_seats_in_cinema2(state) > 0:
                tmp = (person[0], person[1] + 2, person[2], self.cinemaStay)
            else:  # will have to wait
                tmp = (person[0], person[1], person[2] + 1, 1)

        else:
            tmp = (person[0], person[1], person[2], person[3])

        return tmp

    def remove_duplicate(self, moves):
        tmp = list(set(map(tuple, moves)))
        return tmp

    def next_moves(self, state):

        # state = [('N', 2, 0, 1), ('A', 4, 0, 2), ('T', 1, 2, 1), ('M', 1, 2, 1), ('G', 1, 2, 1)]
        moves = []

        if state == self.get_start_state():

            for index1, agent1 in enumerate(state):
                for index2, agent2 in enumerate(state):
                    if index2 == index1:
                        continue
                    for index3, agent3 in enumerate(state):
                        if index3 == index1 or index3 == index2:
                            continue
                        for index4, agent4 in enumerate(state):
                            if index4 == index1 or index4 == index2 or index4 == index3:
                                continue
                            for index5, agent5 in enumerate(state):
                                if index5 == index1 or index5 == index2 or index5 == index3 or index5 == index4:
                                    continue

                                for i in self.delays:
                                    for j in self.delays:
                                        for k in self.delays:


                                            if agent1[0] == self.isNick:
                                                tmp1 = (agent1[0], 2, 0, self.transTimeForNick)
                                            if agent1[0] == self.isAnn:
                                                tmp1 = (agent1[0], 4, 0, self.cafeStay)
                                            if agent1[0] == self.isTasos:
                                                tmp1 = (agent1[0], 2, 0, self.transTimeForTasos)
                                            if agent1[0] == self.isMary:
                                                tmp1 = (agent1[0], 2, 0, self.transTimeForMary)
                                            if agent1[0] == self.isGeorge:
                                                tmp1 = (agent1[0], 2, 0, self.transTimeForGeorge)

                                            if agent2[0] == self.isNick:
                                                tmp2 = (agent2[0], 2, 0, self.transTimeForNick)
                                            if agent2[0] == self.isAnn:
                                                tmp2 = (agent2[0], 4, 0, self.cafeStay)
                                            if agent2[0] == self.isTasos:
                                                tmp2 = (agent2[0], 2, 0, self.transTimeForTasos)
                                            if agent2[0] == self.isMary:
                                                tmp2 = (agent2[0], 2, 0, self.transTimeForMary)
                                            if agent2[0] == self.isGeorge:
                                                tmp2 = (agent2[0], 2, 0, self.transTimeForGeorge)

                                            tmp3 = (agent3[0], 1, i//30, i//30)
                                            tmp4 = (agent4[0], 1, j//30, j//30)
                                            tmp5 = (agent5[0], 1, k//30, k//30)
                                            obj = []
                                            for h in range(5):
                                                for t in [tmp1, tmp2, tmp3, tmp4, tmp5]:
                                                    if h == t[0]:
                                                        obj.append(t)
                                            moves.append(obj)
                                            # print(obj)

            return self.remove_duplicate(moves)
        else:
            for person_index, person in enumerate(state):  # remove the person1 value from the person2 iteration
                moves.append(self.get_transition_states(person, state))

            return [moves]

    def update_graph_node(self, updated_node):
        for index, node in enumerate(self.graph[self.time]):
            if node.state == updated_node.state:
                self.graph[self.time][index] = updated_node

    def append_node_to_graph(self, new_node):
        self.graph[self.time].append(new_node)

    def get_node_from_graph(self, move):
        for node in self.graph[self.time]:
            if move == node.state:
                return node

    def graph_contains(self, move):
        for node in self.graph[self.time]:
            if move == node.state:
                return True
        return False
