'''
Gloabal parameters used.
See report for detailed description
'''
K = 32.5
DESIRED_DISTANCE = 1.615
DISTANCE_MAX = float('inf')
GRADIENT_MAX = float('inf')
G = 1.1 + (DESIRED_DISTANCE - 1)
NEIGHBOR_RADIUS = 3 * DESIRED_DISTANCE
EPSILON = 1e-6
SPEED = 0.01
DIVIDE_LOCALIZE = 2  # paper = 4, but 2 works better here
STARTUP_TIME = 10
TRILATERATION_TYPE = "real"  # choose within { real, opt, ideal }
if TRILATERATION_TYPE == "ideal":
    STARTUP_TIME = 10
elif TRILATERATION_TYPE == "opt":
    STARTUP_TIME = 50
elif TRILATERATION_TYPE == "real":
    STARTUP_TIME = 500
else:
    raise ValueError("TRILATERATION_TYPE unknown: " + str(TRILATERATION_TYPE))

'''
Model of the speed uncertainty as an intrinsic factor, sampled from normal distribution N(0,0.1²) 
additive value
'''
USE_SPEED_UNCERTAINTIES = False
INTRINSIC_MEAN = 1
INTRINSIC_STDDEV = 0.1

'''
Model of the distance variation as a uniform additive value, sampled from U(-2/K, 2/K)
'''
k = 0.25
USE_DISTANCE_UNCERTAINTY = False
DISTANCE_ACCURACY = 2/32.5 * k

'''
Model of the rare event: robot speed = 0, always
'''
USE_RARE_EVENT_SPEED = False
RARE_EVENT_THRESHOLD = 0.95 # quite high.