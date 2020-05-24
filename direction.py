from enum import Enum


class Direction(Enum):
    '''
    Specifies the direction of the movement, as the agent perceives it
    '''
    FWD = 1
    BWD = 2
    FWD_CCW = 3
    FWD_CW = 4


class Orientation(Enum):
    """
    '           N
    '      NW       NE
    '     W           E
    '      SW       SE
    '           S
    ' Specifies the orientaiton of the agent in the world - NOT from the agent point of View,
    ' but from the world's. This allows mapping moving direction with visualization
    """
    N  = 0
    NE = 2
    E = 4
    SE = 6
    S = 8
    SW = 10
    W = 12
    NW = 14

    def succ(self):
        val = (self.value + 2) % 16
        return Orientation(val)

    def pred(self):
        val = (self.value - 2) % 16
        return Orientation(val)

