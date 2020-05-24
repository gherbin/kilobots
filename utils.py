import numpy as np
from direction import Direction,Orientation
from shape import Shape
import itertools
import parameters
from scipy.optimize import minimize

def get_agents_coordinates(center, num_agents, hexa_type ="paper"):
    c_x = center[0]
    c_y = center[1]
    # positions = [(c_x-1, c_y), (c_x, c_y-1), (c_x+1, c_y), (c_x, c_y+1)]
    #
    # # square shape
    # sq_size = np.int_(np.ceil(np.sqrt(num_agents)))
    #
    # # definition of possible x and y
    # possible_x = np.arange(start = c_x, stop = c_x - sq_size, step=-1)
    # possible_y = np.arange(start = c_y - 2, stop = c_y-2 - sq_size, step=-1)
    #
    # # elaboration of the list
    # list_x = np.tile(possible_x, sq_size)
    # list_y = np.repeat(possible_y, sq_size)
    #
    # # list of agent positions
    # positions_agents = [(x,y) for x,y in zip(list_x, list_y) ]
    hexa = np.array([[0, 0, 0], [1, -1, 0], [0, -1, 1], [1, 0, -1]])
    if hexa_type == "paper":
        hexa = np.array(
            [[0, 0, 0], [1, -1, 0], [0, -1, 1],  [1, 0, -1], [-1, -1, 2], [0, -2, 2], [-2, -1, 3],
             [-1, -2, 3], [0, -3, 3],[1, -4, 3], [-2, -2, 4], [-1, -3, 4], [0, -4, 4], [1, -5, 4],
             [-3, -2, 5], [-2, -3, 5], [-1, -4, 5], [0, -5, 5], [1, -6, 5], [-4, -2, 6], [-3, -3, 6],
             [-2, -4, 6], [-1, -5, 6], [0, -6, 6], [1, -7, 6], [-4, -3, 7]])
    else:
        hexa = np.vstack((np.array([[0, 0, 0], [1, -1, 0], [0, -1, 1], [1, 0, -1]]),
                          np.array([coord for coord in get_hexa_coord(20)])))

    cubic_hexa = hexa[:, [0, 2]]
    Hex2Cart = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, -3 / 2]])*1/np.sqrt(3)
    cart = np.dot(Hex2Cart, cubic_hexa.T)

    cart_sliced = cart[:,0: 4+num_agents].copy()
    cart_sliced[0, :] = cart_sliced[0, :] + c_x-0.5
    cart_sliced[1, :] = cart_sliced[1, :] + c_y
    coord = [(x, y) for x, y in cart_sliced.T]
    return coord


def get_deltas(dir, orn):
    '''

    :param dir: desired movement direction
    :param orn: current robot orientation
    :return: dx, dy = {-1,0,1},{-1,0,1} : displacement direction required
    '''
    dx = 0
    dy = 0

    if dir == Direction.FWD:
        if orn in [Orientation.N, Orientation.NE, Orientation.NW]:
            dy = 1
        if orn in [Orientation.S, Orientation.SE, Orientation.SW]:
            dy = -1
        if orn in [Orientation.W, Orientation.NW, Orientation.SW]:
            dx = -1
        if orn in [Orientation.E, Orientation.NE, Orientation.SE]:
            dx = 1
    elif dir == Direction.FWD_CW:
        if orn in [Orientation.N, Orientation.W, Orientation.NW]:
            dy = 1
        if orn in [Orientation.S, Orientation.SE, Orientation.E]:
            dy = -1
        if orn in [Orientation.W, Orientation.S, Orientation.SW]:
            dx = -1
        if orn in [Orientation.E, Orientation.NE, Orientation.N]:
            dx = 1
    elif dir == Direction.FWD_CCW:
        if orn in [Orientation.N, Orientation.NE, Orientation.E]:
            dy = 1
        if orn in [Orientation.S, Orientation.W, Orientation.SW]:
            dy = -1
        if orn in [Orientation.W, Orientation.NW, Orientation.N]:
            dx = -1
        if orn in [Orientation.E, Orientation.S, Orientation.SE]:
            dx = 1
    else:
        raise NotImplementedError("get_deltas not implemented for dir : " + str(dir))
    factor = 1
    if abs(dx*dy) == 1:
        # the robot is going on a diagonal axis => performing sqrt(2) instead of 1 in a step
        factor = 1/np.sqrt(2)
    dx *= factor
    dy *= factor

    return dx, dy


def build_shape(world_width, world_height, size_of_shape):
    w_ = world_width // 2
    h_ = world_height// 2
    background = np.zeros((w_,h_))
    foreground = np.ones(size_of_shape)
    mymap = background

    size_desired = foreground.shape
    # x = center[0]
    x = 0
    dx = size_desired[0]
    # y = center[1]
    y = 0
    dy = size_desired[1]

    if x+dx > w_:
        raise ValueError("shape requested too large in width:" + str(x+dx) + " > " + str(w_))
    if y+dy > h_:
        raise ValueError("shape requested too large in height:" + str(y+dy) + " > " + str(h_))
    mymap[x:x+dx , y:y+dy] = foreground
    return Shape(1, mymap)


def at_least_three_non_colinear(list_of_points):
    # flag = 0
    for p0, p1, p2 in itertools.combinations(list_of_points, 3):
        if not collinear(p0, p1, p2):
            return True
    return False


def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < parameters.EPSILON

def trilateration(s_pos, s_pos_n_list, distances_n_list):
    # locations: [ (lat1, long1), ... ]
    # distances: [ distance1,     ... ]
    result = minimize(
        my_rmse,  # The error function
        s_pos,  # The initial guess
        args=(s_pos_n_list, distances_n_list),  # Additional parameters for mse
        method='L-BFGS-B',  # The optimisation algorithm
        options={
            'ftol': 1e-4,  # Tolerance
            'maxiter': 1e+8  # Maximum iterations
        }
    )
    # print("Trilateration ended: message > " + str(result.message))
    # print("Trilateration ended: success > " + str(result.success))
    return result.x

def my_rmse(s_pos, s_pos_n_list, distances_n_list):
    """

    :param s_pos: (s_x, s_y)
    :param s_pos_n_list: [ (s_x_1, s_y_1), (s_x_2, s_y_2), ... ]
    :param distances_n_list: [ d1, d2, ... ]
    :return:  mean squared error between distances from local coord system and global coord system
    """
    a = np.array(s_pos)
    distances_s_pos = [np.linalg.norm(a - np.array(b), ord=2) for b in s_pos_n_list]
    res = np.sqrt(np.mean(np.square(np.array(distances_s_pos)-np.array(distances_n_list))))
    return res


def get_hexa_coord(radius):
    hexa_j = []
    hexa_k = []
    hexa_l = []
    for i in range(radius):
        for j in range(-i, i+1):
            for k in range(-i, i+1):
                if k <= -1:
                    for l in range(-i, i+1):
                        if l >= 2:
                            if (np.abs(j) + np.abs(k)+np.abs(l) == 2*i) and (j+k+l == 0):
                                hexa_j.append(j)
                                hexa_k.append(k)
                                hexa_l.append(l)
    return list(zip(hexa_j, hexa_k, hexa_l))
