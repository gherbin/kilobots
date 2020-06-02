from shape import Shape
import numpy as np
from world import World
from robot import Robot
from utils import get_agents_coordinates

def get_paper_world():
    num_agents = 22
    num_seeds = 4  # should always be 4

    # width and height of the global world
    width = 25
    height = 25
    center = (float(width // 2), float(height // 2))
    robots_world_pos= get_agents_coordinates(center, num_agents, hexa_type = "paper")
    # create a world
    myshape = get_paper_shape()
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
    myshape = Shape(1, mymap)
    print(myshape)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, myshape)
    return dummy_word

def get_paper_shape():
    return Shape(2.51, np.array([[1,1], [1,0]]))

def get_dummy_bot(shape = None):
    return Robot(unique_id=999, model = None, world_pos = (0,0), shape = shape)