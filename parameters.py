DESIRED_DISTANCE = 1.615 # (from center to center)  1.615 Because (2*16.25 + 20) / 32.5, two agents are separated# by
# 20  mm, see p 10, SM
DISTANCE_MAX = float('inf')
GRADIENT_MAX = float('inf')
G = 1.1 + (DESIRED_DISTANCE - 1)
NEIGHBOR_RADIUS = 3 * DESIRED_DISTANCE
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
    STARTUP_TIME = 500
else:
    raise ValueError("TRILATERATION_TYPE unknown: " + str(TRILATERATION_TYPE))

'''
> DESIRED_DISTANCE => /!\ effect of number of neighbours it can see
> has_met_root needs to have seen the four seeds
-> radius shall be at least 3 neighbours, considering the distance
-> impact on the gradient value if too large! => modif of G
'''

USE_SPEED_UNCERTAINTIES = True
INSTRINSIC_MEAN = 1
INSTRINSIC_STDDEV = 0.05

