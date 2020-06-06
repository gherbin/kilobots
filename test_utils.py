from shape import Shape
import numpy as np
from world import World
from robot import Robot
from utils import get_agents_coordinates


def get_paper_world():
    """
    :return: the world as represented in tha paper [5]
    """
    num_agents = 22
    num_seeds = 4  # should always be 4

    # width and height of the global world
    width = 25
    height = 25
    center = (float(width // 2), float(height // 2))
    robots_world_pos = get_agents_coordinates(center, num_agents, hexa_type = "paper")
    # create a world
    my_shape = get_paper_shape()
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, my_shape)
    return dummy_word


def get_realistic_world(num_agents = 5):
    """
    :param num_agents:
    :return: A world respecting initial constraint, and containing num_agents agents disposed in a lattice
    """
    num_seeds = 4  # should always be 4
    width = 50
    height = 50
    center = (float(width // 2), float(height // 2))
    robots_world_pos= get_agents_coordinates(center, num_agents, hexa_type = "rectangle")
    print(len(robots_world_pos))
    # create a world
    mymap = np.ones((4,2))
    myshape = Shape(1, mymap)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, myshape)
    return dummy_word


def get_dummy_world(num_agents=3, shape_shape=(3, 3), width=11, height=11):
    """
    returns a dummy world for test purpose
    :param num_agents:
    :param shape_shape:
    :param width:
    :param height:
    :return:
    """
    # num_agents = 3
    num_seeds = 4  # should always be 4
    # width = 20
    # height = 20
    center = (float(width // 2), float(height // 2))
    robots_world_pos = get_agents_coordinates(center, num_agents, hexa_type="paper")
    # create a world
    shape = build_shape(width, height, shape_shape)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape)
    return dummy_word


def get_paper_shape():
    return Shape(2.5, np.array([[1,1], [1,0]]))


def get_dummy_bot(shape = None):
    return Robot(unique_id=999, model = get_paper_world() , world_pos = (0,0), shape = shape)


def build_shape(world_width, world_height, size_of_shape):
    """
    :param world_width:
    :param world_height:
    :param size_of_shape:
    :return:
    """
    w_ = world_width // 2
    h_ = world_height // 2
    background = np.zeros((w_, h_))
    foreground = np.ones(size_of_shape)
    mymap = background

    size_desired = foreground.shape
    # x = center[0]
    x = 0
    dx = size_desired[0]
    # y = center[1]
    y = 0
    dy = size_desired[1]

    if x + dx > w_:
        raise ValueError("shape requested too large in width:" + str(x + dx) + " > " + str(w_))
    if y + dy > h_:
        raise ValueError("shape requested too large in height:" + str(y + dy) + " > " + str(h_))
    mymap[x:x + dx, y:y + dy] = foreground
    return Shape(1, mymap)


