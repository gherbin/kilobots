from enum import Enum


class State(Enum):
    """
    represents the state of a robot
    """
    START = 'start'
    WAIT_TO_MOVE = 'wait_to_move'
    MOVE_WHILE_OUTSIDE = 'move_while_outside'
    MOVE_WHILE_INSIDE = 'move_while_inside'
    JOINED_SHAPE = 'joined_shape'
