import numpy as np
from direction import Direction,Orientation
from shape import Shape
import itertools
import parameters
from scipy.optimize import minimize

def get_agents_coordinates(center, num_agents, hexa_type ="paper"):
    """
    Computes the correct coordinates of the agents, according to an hexagonal lattice.
    :param center: center of the world
    :param num_agents: number of coordinates to return
    :param hexa_type: one of {"paper", "rectangle"} to define the shape of the lattice
    :return: a list of the coordinates for all agents (including seeds)
    """
    c_x = center[0]
    c_y = center[1]

    hexa = np.array([[0, 0, 0], [1, -1, 0], [0, -1, 1], [1, 0, -1]])
    if hexa_type == "paper":
        hexa = np.array(
            [[0, 0, 0], [1, -1, 0], [0, -1, 1],  [1, 0, -1], [-1, -1, 2], [0, -2, 2], [-2, -1, 3],
             [-1, -2, 3], [0, -3, 3],[1, -4, 3], [-2, -2, 4], [-1, -3, 4], [0, -4, 4], [1, -5, 4],
             [-3, -2, 5], [-2, -3, 5], [-1, -4, 5], [0, -5, 5], [1, -6, 5], [-4, -2, 6], [-3, -3, 6],
             [-2, -4, 6], [-1, -5, 6], [0, -6, 6], [1, -7, 6], [-4, -3, 7]])
    elif hexa_type == "rectangle":
        hexa = np.vstack((np.array([[0, 0, 0], [1, -1, 0], [0, -1, 1], [1, 0, -1]]),
                          np.array([coord for coord in get_hexa_coord(20)])))

    cubic_hexa = hexa[:, [0, 2]]
    hex2cart = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, -3 / 2]])*1/np.sqrt(3)
    cart = np.dot(hex2cart, cubic_hexa.T)

    cart_sliced = cart[:,0: 4+num_agents].copy()
    cart_sliced[0, :] = cart_sliced[0, :] + c_x-0.5
    cart_sliced[1, :] = cart_sliced[1, :] + c_y
    coord = [(x, y) for x, y in cart_sliced.T]
    return coord


def get_deltas(dir, orn):
    '''
    :param dir: desired movement direction {Direction.FWD, Direction.FWD_CW, Direction.FWD_CCW}
    :param orn: current robot orientation
    :return: dx, dy = {-1,0,1},{-1,0,1},... : displacement direction required
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


def at_least_three_non_colinear(list_of_points):
    """
    :param list_of_points:
    :return: True if at least three point of the list_of_points are non colinear. False otherwise
    """
    # flag = 0
    for p0, p1, p2 in itertools.combinations(list_of_points, 3):
        if not collinear(p0, p1, p2):
            return True
    return False


def collinear(p0, p1, p2):
    """
    :param p0, p1, p2: triplets of coordinates
    :return: True if the triplets p0, p1 and p2 are colinear. It uses the determinant method
    """
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < parameters.EPSILON

def trilateration(s_pos, s_pos_n_list, distances_n_list):
    """
    :param s_pos:
    :param s_pos_n_list:
    :param distances_n_list:
    :return: the result of the minimization of the trilateration problem
    """
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
    """
    :param radius: indicates the size of coordinates to compute. generally, ~20 for 100 robots works fine
    :return: a list of the hexagonal coordinates as defined in [6]. Only acceptable coordinates are generated.
    """
    hexa_j = []
    hexa_k = []
    hexa_l = []
    for i in range(radius):
        for j in range(-i, i + 1):
            for k in range(-i, i + 1):
                if k <= -1:
                    for l in range(-i, i + 1):
                        if l >= 3:
                            if (np.abs(j) + np.abs(k) + np.abs(l) == 2 * i) and (j + k + l == 0):
                                hexa_j.append(j)
                                hexa_k.append(k)
                                hexa_l.append(l)
                        elif l >= 2 and j < 1:
                            if (np.abs(j) + np.abs(k) + np.abs(l) == 2 * i) and (j + k + l == 0):
                                hexa_j.append(j)
                                hexa_k.append(k)
                                hexa_l.append(l)
    return list(zip(hexa_j, hexa_k, hexa_l))