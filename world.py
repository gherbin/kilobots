from mesa import Model
from robot import Robot, Seed
from mesa.space import ContinuousSpace
import utils
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
import numpy as np
from shape import Shape
import parameters

class World(Model):
    """
    Represents the world
    """
    # create a world
    def __init__(self, num_seeds, num_agents, width, height, robots_world_pos=None, shape=None):
        super().__init__()
        self.num_agents = num_agents
        self.space = ContinuousSpace(width, height, True)
        self.schedule = SimultaneousActivation(self)
        self.num_seeds = num_seeds

        if shape is None:
            # used if server -- deprecated
            mymap = np.array(
                    [[1, 1],
                        [1, 0]])
            shape = Shape(2.5, mymap)

        self.shape = shape
        if robots_world_pos is None:
            # used if server
            center = (float(width // 2), float(height // 2))
            robots_world_pos = utils.get_agents_coordinates(center, num_agents)

        for i in range(self.num_seeds):
            seed = Seed(i, self, robots_world_pos[i])
            self.schedule.add(seed)
            self.space.place_agent(seed, seed.pos)

        np.random.seed(0)
        speed_factors = parameters.INTRINSIC_MEAN + np.random.randn(self.num_agents) + parameters.INTRINSIC_STDDEV

        for i in range(num_agents):
            # create robot
            robot = Robot(unique_id=self.num_seeds + i,
                          model=self,
                          world_pos=robots_world_pos[self.num_seeds + i],
                          shape=self.shape)
            # if use of uncertainties, set a variable speed to all robots
            if parameters.USE_SPEED_UNCERTAINTIES:
                robot.intrinsic_speed_factor = speed_factors[i]
            # add the new robot to scheduler
            self.schedule.add(robot)
            # place new robot in starting position
            self.space.place_agent(robot, robot.pos)

        # print("Created ", self.num_seeds, 'seeds and ', self.num_agents, 'agents')
        self.datacollector = DataCollector(
            agent_reporters= { "Position": "pos", "Local_Position":"_s_pos", "State":"state"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def __str__(self):
        return ' World: (' + \
               str(self.space.width) + \
               ',' + str(self.space.height) + \
               ');  #agents = ' + str(self.num_agents) + \
               '; #seeds = ' + str(self.num_seeds)
