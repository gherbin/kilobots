from shape import Shape
import numpy as np
from world import World
from robot import Robot
from utils import get_agents_coordinates

def get_paper_world():
    num_agents = 22
    num_seeds = 4  # should always be 4
    width = 25
    height = 25
    center = (float(width // 2), float(height // 2))
    # robots_world_pos = [(10, 10), (10.5, 9.5), (11, 10), (10.5, 10.5),
    #                     (10, 9), (11, 9),
    #                     (9.5, 8.5), (10.5, 8.5), (11.5, 8.5), (12.5, 8.5),
    #                     (10, 8), (11, 8), (12, 8), (13, 8),
    #                     (9.5, 7.5), (10.5, 7.5), (11.5, 7.5), (12.5, 7.5), (13.5, 7.5),
    #                     (9, 7), (10, 7), (11, 7), (12, 7), (13, 7), (14, 7),
    #                     (9.5, 6.5)]

    robots_world_pos= get_agents_coordinates(center, num_agents, hexa_type = "paper")
    # create a world
    mymap = np.array([[1,1],[1,0]])
    myshape = Shape(2.55, mymap)
    # print(myshape)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, myshape)
    return dummy_word


def get_realistic_world(num_agents = 5):
    num_seeds = 4  # should always be 4
    width = 25
    height = 25
    center = (float(width // 2), float(height // 2))
    robots_world_pos= get_agents_coordinates(center, num_agents)
    # create a world
    mymap = np.array(
        [[1, 1],
         [1,0]])
    myshape = Shape(2.5, mymap)
    print(myshape)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, myshape)
    return dummy_word

def get_paper_shape():
    return Shape(2.5, np.array([[1,1], [1,0]]))

def get_custom_shape(scale, map):
    return Shape(scale, map)

def get_dummy_bot(shape = None):
    return Robot(999, None, (0,0), shape)