DESIRED_DISTANCE = 1.615 #Because (2*16.25 + 20) / 32.5, two agents are separated by 20 mm in reality, see p 10, SM
DISTANCE_MAX = float('inf')
GRADIENT_MAX = float('inf')
G = 1.7 #1.5
NEIGHBOR_RADIUS = 3
EPSILON = 1e-6
SPEED = 0.01
DIVIDE_LOCALIZE = 2 #paper = 4
STARTUP_TIME = 10
TRILATERATION_TYPE = "real" # { real, opt, ideal }
if TRILATERATION_TYPE == "ideal":
    STARTUP_TIME = 10
elif TRILATERATION_TYPE == "opt":
    STARTUP_TIME = 50
elif TRILATERATION_TYPE == "real":
    STARTUP_TIME = 250
else:
    raise ValueError("TRILATERATION_TYPE unknown: " + str(TRILATERATION_TYPE))



