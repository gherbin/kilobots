from mesa import Agent
from direction import Direction, Orientation
import utils
import numpy as np
import world
import parameters
from state import State


class Robot(Agent):
    def __init__(self, unique_id, model, world_pos=None, shape=None, intrinsic_speed_factor=1):
        super().__init__(unique_id, model)
        # attributes related to the world
        if world_pos is None:
            _x = self.random.randrange(self.model.space.width)
            _y = self.random.randrange(self.model.space.height)
            self.pos = (_x, _y)
        else:
            self.pos = world_pos
        self._old_y = self.pos[1]
        self.orn = Orientation.N

        # intrinsic variables
        self.intrinsic_speed_factor = intrinsic_speed_factor


        # attributes related to the swarm
        self.is_seed = False
        self.is_stationary = True
        self.is_localized = False
        self.neighbors = None  # refreshed at every step
        self.previous_neighbors = []
        ## kind of private -> more of an indication this should not be accessed nor modified
        self._s_pos = (0, 0)

        ## opt: could be a deque of tuple, or an array of array
        self._s_pos_memory_size = 2
        self._s_pos_memory_x = np.tile(np.array([np.float('inf'), 0]), self._s_pos_memory_size)
        self._s_pos_memory_y = np.tile(np.array([np.float('inf'), 0]), self._s_pos_memory_size)

        # indicates if the agent has met the seed
        self._has_met_coord = False
        # indicates stopping condition
        self.met_root_twice = False
        # helpers
        self.crossed = 0 # cross the X World axis (= external stopping condition)
        self.limit_crossing = self.model.space.height // 2

        self._s_gradient_value = parameters.GRADIENT_MAX
        self.shape = shape
        self.radius = parameters.NEIGHBOR_RADIUS  # radius to look for neighbours
        ##
        self.prev_ef = parameters.DISTANCE_MAX  # prev used in edge_follow procedure
        self.current_ef = parameters.DISTANCE_MAX  # current used in edge_follow procedure
        ##
        self.state = State.START
        self._next_state = State.START
        self.timer = 0

    def advance(self):
        self.state = self._next_state

    def step(self):
        self.timer += 1
        self.previous_neighbors = self.neighbors
        self.neighbors = self.get_neighbors()

        if (self.unique_id > 3) and (self.timer % 500 == 0):
            print(self)
        if self.state == State.START:
            self._step_start()
        elif self.state == State.WAIT_TO_MOVE:
            self._step_wait_to_move()
        elif self.state == State.MOVE_WHILE_OUTSIDE:
            self._step_move_while_outside()
        elif self.state == State.MOVE_WHILE_INSIDE:
            self._step_move_while_inside()
        elif self.state == State.JOINED_SHAPE:
            self._step_joined_shape()
        else:
            raise ValueError("state not recognized:" + str(self.state))

    def _step_start(self):
        self.is_stationary = True
        self.is_localized = False
        self.compute_gradient()
        self.localize()
        if self.timer >= parameters.STARTUP_TIME:
            self._next_state = State.WAIT_TO_MOVE
        return

    def _step_wait_to_move(self):
        self.is_stationary = True
        self.is_localized = False
        self.compute_gradient()
        self.localize()
        highest_gradient = 0
        # neighbors = self.get_neighbors()
        no_moving_neighbors = all([neighbor.is_stationary for neighbor in self.neighbors])  # check if all
        # neighbors are stationary

        if len(self.neighbors) == 0:
            raise RuntimeError("no neighbors")

        if no_moving_neighbors and len(self.neighbors) > 0:
            assert len(self.neighbors) > 0, "neighbors is empty"

            neighbors_not_in_JOINT_SHAPE = [neighbor.get_s_gradient_value() for neighbor in self.neighbors \
                                            if not (neighbor.state == State.JOINED_SHAPE)]
            if len(neighbors_not_in_JOINT_SHAPE):
                highest_gradient = max(neighbors_not_in_JOINT_SHAPE)
            # print("[DEBUG_WAIT_TO_MOVE] ("+ str((self.unique_id, self._s_gradient_value)) + \
            #       "): no_moving neighbors: " + str(no_moving_neighbors) +\
            #       " highest gradient => " + str(highest_gradient))
            if self._s_gradient_value > highest_gradient:
                self._next_state = State.MOVE_WHILE_OUTSIDE
            elif self._s_gradient_value == highest_gradient:
                list_of_same_gradient_values = [neighbor.unique_id \
                                                for neighbor in self.neighbors \
                                                if (neighbor.get_s_gradient_value() == self._s_gradient_value) \
                                                and not (neighbor.state == State.JOINED_SHAPE)]
                # print(list_of_same_gradient_values)
                highest_id = max(list_of_same_gradient_values)
                # print("[DEBUG_WAIT_TO_MOVE]("+ str(self.unique_id) + "): same gradient Highest_ID = ", highest_id)
                if self.unique_id > highest_id:
                    self._next_state = State.MOVE_WHILE_OUTSIDE
        else:
            pass
        return

    def _step_move_while_outside(self):
        self.is_stationary = False
        self.compute_gradient()
        self.localize()
        if self.is_in_shape():
            self._next_state = State.MOVE_WHILE_INSIDE
        # todo: if distance to front edge-following robot > yield_distance ? how to ? by gathering distances of
        # neighbors robot that are not stationnary ? if so, ok !
        self.edge_follow()
        return

    def _step_move_while_inside(self):
        self.is_stationary = False

        self.compute_gradient()
        self.localize()

        if not self.is_in_shape():
            print("[INSIDE SHAPE] Bot(" + str(self.unique_id) + ") stopped => almost leave shape : " + str(self._s_pos))
            print(self)
            self._next_state = State.JOINED_SHAPE
            return
        if self._s_gradient_value <= self.get_closest_neighbor().get_s_gradient_value():
            print("[INSIDE SHAPE] Bot(" + str(self.unique_id) + ") stopped => _s_gradient_value <= closest neighbor")
            print(self)
            self._next_state = State.JOINED_SHAPE
            return
        # todo distance to front ...
        self.edge_follow()
        return

    def _step_joined_shape(self):
        self.is_stationary = True

        pass

    def is_in_shape(self):
        if not (self._has_met_coord and self.is_localized):
            # print("is_in_shape => Not localized -> Abort")
            return False
        else:
            _x = (self._s_pos[0]) / self.shape.scale
            _y = (self._s_pos[1]) / self.shape.scale
            res = False
            if _x >= 0 and _y >= 0:
                try:
                    _x = int(np.round(_x * 1000)) // 1000
                    _y = int(np.round(_y * 1000)) // 1000
                    map_shape = self.shape.map.shape
                    if _x < map_shape[0] and _y < map_shape[1]:
                        res = self.shape.map[_x, _y] == 1
                    else:
                        res = False
                except IndexError:
                    print("[is_in_shape] Index Error: " + str((_x, _y)) + " but max is " + \
                          str((self.shape.map.shape[0] - 1, self.shape.map.shape[1] - 1)))
                    res = False

            # list_coord = ?
            # res = utils.point_inside_polygon(self._s_pos[0],self._s_pos[1],list_coord, False)
            return res

    def get_s_pos(self):
        return self._s_pos

    # pour test uniquement
    def set_s_pos(self, new_pos):
        self._s_pos = new_pos

    def get_s_gradient_value(self):
        return self._s_gradient_value

    def get_neighbors(self):
        """
        :return: a list of the neighbors
        """
        neighbors = self.model.space.get_neighbors(self.pos, radius=self.radius, include_center=False)
        # print("debug: get_neighbors")
        # print("self.unique_id = ", self.unique_id)
        # print("self.get_neighbors = ", [a.unique_id for a in neighbors])
        # print("debug end : get_neighbors")

        return neighbors

    def get_distance(self, agent):
        return np.linalg.norm(np.array(self.pos) - np.array(agent.pos), ord=2)
        #

    def get_s_distance(self, agent):
        '''
        only if agent is stationary
        :param agent:
        :return:
        '''
        if agent.is_stationary:
            a = np.array(self._s_pos)
            b = np.array(agent.get_s_pos())
            return np.linalg.norm(a - b, ord=2)
        else:
            raise RuntimeError("Agent is not stationary!! ")

    def get_closest_distance(self, from_stationary_only=False):
        d = self.get_neighbors_distances(from_stationary_only)
        if len(d) > 0:
            return np.min(d)  # self.get_neighbors_distances())
        else:
            return parameters.DISTANCE_MAX

    def get_closest_neighbor(self):
        if self.neighbors is None:
            self.neighbors = self.get_neighbors()
        assert len(self.neighbors) > 0, "neighbors is empty"
        sP = np.array([agent.pos for agent in self.neighbors])
        distances = np.linalg.norm(sP - np.array(self.pos), ord=2, axis=1)
        ind = np.argmin(distances)
        return self.neighbors[ind]

    def get_neighbors_distances(self, from_stationary_only=False):
        if self.neighbors is None:
            self.neighbors = self.get_neighbors()
        if from_stationary_only:
            stationary_neighbors = [neighbor for neighbor in self.neighbors if neighbor.is_stationary]

        if len(stationary_neighbors) > 0:
            # print("debug dist: get_neighbors_distances -- START")
            # print("debug dist: neighbors of self ("+ str(self.unique_id)+ ') = '+str([a.unique_id for a in neighbors]))
            sP = np.array([agent.pos for agent in stationary_neighbors])
            # print("debug dist: neighbors POS of : "+ str(self.pos))
            # print("debug dist: neighbors POS of self.neighbors: \n"+ str(sP))
            distances = np.linalg.norm(sP - np.array(self.pos), ord=2, axis=1)
        else:
            distances = []
        # print("debug dist: All neighbors distances: ", distances)
        return distances

    def edge_follow(self):
        '''
        Algorithm as implemented in page 3 of the supplementary material
        loop replaced by the fact Agent is called per steps
        :return:
        '''
        # print("[EF] old closest distance : ", self.current_ef)
        self.current_ef = self.get_closest_distance(from_stationary_only=True)
        # print("[EF] prev = " + str(self.prev_ef) + "; current = " + str(self.current_ef))

        if self.current_ef < parameters.DESIRED_DISTANCE:
            if self.prev_ef < self.current_ef:
                # print("[Debug EF] cur < desired; prev < cur => FWD")
                self.move_dir(Direction.FWD)
            else:
                # print("[Debug EF] cur < desired; prev >= cur => FWD_CCW")
                self.move_dir(Direction.FWD_CCW)
        else:
            if self.prev_ef > self.current_ef:
                # print("[Debug EF] cur > =  desired; prev > cur => FWD")
                self.move_dir(Direction.FWD)
            else:
                # print("[Debug EF] cur > =  desired; prev <= cur => FWD")
                self.move_dir(Direction.FWD_CW)

        self.prev_ef = self.current_ef

    def compute_gradient(self):
        self._s_gradient_value = parameters.GRADIENT_MAX
        if self.neighbors is None:
            self.neighbors = self.get_neighbors()
        # stationary_neighbors = [neighbor for neighbor in neighbors if neighbor.is_stationary]
        #
        # for neighbor in stationary_neighbors:
        #     if self.get_distance(neighbor) < parameters.G:
        #         v = neighbor.get_s_gradient_value()
        #         if v < self._s_gradient_value:
        #             self._s_gradient_value = v
        stationary_neighbors_s_gradient_value = [neighbor.get_s_gradient_value() for neighbor in self.neighbors \
                                                 if neighbor.is_stationary and (
                                                             self.get_distance(neighbor) < parameters.G)]
        if len(stationary_neighbors_s_gradient_value):
            self._s_gradient_value = min(stationary_neighbors_s_gradient_value)

        self._s_gradient_value += 1
        # transmit => done through update

    def localize(self):
        # n_list = []
        if self.neighbors is None:
            self.neighbors = self.get_neighbors()
        # for neighbor in neighbors:
        #     if neighbor.is_localized and neighbor.is_stationary:
        #         n_list.append(neighbor)
        n_list = [neighbor for neighbor in self.neighbors if (neighbor.is_stationary and neighbor.is_localized)]
        s_positions_neighbors = [neighbor.get_s_pos() for neighbor in n_list]
        # print(s_positions_neighbors)
        self.is_localized = False

        if utils.at_least_three_non_colinear(s_positions_neighbors):
            if (((-0.5, 0) in s_positions_neighbors) and  # met_root
                    ((0.5, 0) in s_positions_neighbors) and  # met_v1
                    ((0.0, -0.8660254037844387) in s_positions_neighbors) and  # met_v2
                    ((0.0, 0.8660254037844387) in s_positions_neighbors)):  # met_v3
                self._has_met_coord = True
            if self.crossed > 1:
                self.met_root_twice = True


            if parameters.TRILATERATION_TYPE == "opt":
                s_pos_n_list = s_positions_neighbors
                distances_n_list = [self.get_distance(neighbor) for neighbor in n_list]
                new_pos = utils.trilateration(self._s_pos,
                                              s_pos_n_list=s_pos_n_list,
                                              distances_n_list=distances_n_list)
                self._s_pos = (new_pos[0], new_pos[1])
            elif parameters.TRILATERATION_TYPE == "ideal":

                x_center = float(self.model.space.width // 2)
                y_center = float(self.model.space.height // 2)
                self._s_pos = (self.pos[0] - x_center, self.pos[1] - y_center)

            elif parameters.TRILATERATION_TYPE == "real":
                for localized_neighbor in n_list:
                    c = self.get_s_distance(localized_neighbor)
                    v = (np.array(self._s_pos) - np.array(localized_neighbor.get_s_pos())) / (c + parameters.EPSILON)
                    measured_distance = self.get_distance(localized_neighbor)
                    n = np.array(localized_neighbor.get_s_pos()) + measured_distance * v
                    s_pos_tmp = np.array(self._s_pos)

                    self._s_pos = tuple(s_pos_tmp - (s_pos_tmp - n) / parameters.DIVIDE_LOCALIZE)

            # Agent becomes localized only if position since X time steps has remained the same. While this is not
            # True, it remains unlocalized
            self.is_localized = self._check_if_localized()

    def move_dir(self, direction_to_move):
        """
        robot moves according to dir specified
        :param direction_to_move: Direction in which the robot wants to move (in its coordinate syste)
        :return: /
        """
        # if self.is_stationary:
        #     print("Attempt to move but robot stationary")
        #     return
        dx, dy, = utils.get_deltas(direction_to_move, self.orn)
        self._old_y= self.pos[1]
        new_pos = (self.pos[0] + dx * parameters.SPEED * self.intrinsic_speed_factor,
                   self.pos[1] + dy * parameters.SPEED * self.intrinsic_speed_factor)
        if (self._old_y < self.limit_crossing)  and (new_pos[1] > self.limit_crossing):
            self.crossed+=1
        self.model.space.move_agent(self, new_pos)

        # update orientation according to movement performed
        if direction_to_move == Direction.FWD_CCW:
            self.orn = self.orn.pred()
        elif direction_to_move == Direction.FWD_CW:
            self.orn = self.orn.succ()

    def _check_if_localized(self):
        """
        :return: true if self._s_pos is the same as the X previous time it was computed. It means the agent has not
        moved for a while (X steps, defined by the length of s_pos_memory_x and s_pos_memory_y
        """
        self._s_pos_memory_x[:-1] = self._s_pos_memory_x[1:]
        self._s_pos_memory_x[-1] = np.round(self._s_pos[0], 3)
        self._s_pos_memory_y[:-1] = self._s_pos_memory_y[1:]
        self._s_pos_memory_y[-1] = np.round(self._s_pos[1], 3)

        # print("[check if localized X ] => " + str(self._s_pos_memory_x))
        # print("[check if localized Y ] => " + str(self._s_pos_memory_y))

        if (not self.is_stationary) and self._has_met_coord:
            prev_pos = np.array([self._s_pos_memory_x[-1], self._s_pos_memory_y[-1]])
            return np.linalg.norm(np.array(self._s_pos) - prev_pos, ord=2) <= 1 * parameters.SPEED
        else:
            result_x = np.max(self._s_pos_memory_x) == np.min(self._s_pos_memory_x)
            result_y = np.max(self._s_pos_memory_y) == np.min(self._s_pos_memory_y)
            return (result_x and result_y)

    def __str__(self):
        res = "robot:{ unique_id = " + str(self.unique_id) + \
              "; state = " + str(self.state) + \
              "; is_seed = " + str(self.is_seed) + \
              "; is_stationary = " + str(self.is_stationary) + \
              "; is_localized = " + str(self.is_localized) + \
              "; pos = " + str(np.round(self.pos, 2)) + \
              "; orn = " + str(self.orn) + \
              "; s_pos = " + str(np.round(self._s_pos, 2)) + \
              "; has_met_coord = " + str(self._has_met_coord) + \
              "; met_root_twice = " + str(self.met_root_twice) + \
              "; s_gradient = " + str(self._s_gradient_value) + \
              "; neighbors = " + str([b.unique_id if b is not None else '/' for b in (self.neighbors or [])]) + \
              "; previous_neighbors = " + str([b.unique_id if b is not None else '/'
                                               for b in (self.previous_neighbors or [])]) + \
              "}"
        return res


class Seed(Robot):
    def __init__(self, unique_id, model, world_pos):
        super().__init__(unique_id, model, world_pos)
        self.is_seed = True
        self.is_stationary = True
        self.is_localized = True
        self._has_met_coord = True
        self.met_root_twice = False
        self.is_gradient_seed = False
        if self.unique_id == 0:
            self.is_gradient_seed = True
            self._s_gradient_value = 0
            self._s_pos = (-0.5, 0.0)
        else:
            self._s_gradient_value = 1

        # we give the seed interal correct coordinates (-0.5, 0.0), (0.5, 0.0), (0.0, -0.8660254037844387), (0.0, 0.8660254037844387)
        # shape as described in supplementary material
        #   (v3)
        # (v0)(v1)
        #   (v2)

        if self.unique_id == 1:
            self._s_pos = (0.5, 0.0)
        if self.unique_id == 2:
            self._s_pos = (0.0, -0.8660254037844387)
        if self.unique_id == 3:
            self._s_pos = (0.0, 0.8660254037844387)

        self.state = State.JOINED_SHAPE
        self._next_state = State.JOINED_SHAPE

    def compute_gradient(self):
        if self.is_gradient_seed:
            self._s_gradient_value = 0
        else:
            self._s_gradient_value = 1
        return

    def localize(self):
        pass

    def _step_start(self):
        self.is_stationary = True
        self.is_localized = True
        self.compute_gradient()
        self.localize()

    # def move_dir(self, dir):
    #     raise RuntimeError("Seed: cannot move")

    # def step(self):
    #     pass
