from mesa.visualization.ModularVisualization import ModularServer
import utils
from world import World
from SimpleContinuousModule import SimpleCanvas


# def boid_draw(agent):
#     return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}
def robot_draw(agent):

    if agent.is_seed:
        if agent.unique_id == 0:
            return {"Shape": "circle", "r": 5, "Filled": "true", "Color": "Red"}
        else:
            return {"Shape": "circle", "r": 5, "Filled": "true", "Color": "Blue"}
    else:
        return {"Shape": "circle", "r": 5, "Filled": "true", "Color": "Green"}

bot_canvas  = SimpleCanvas(robot_draw, 500, 500)

num_agents = 6
# robots_world_pos = utils.get_agents_coordinates((12.5,5.0), num_agents)
# my_shape = utils.build_shape(11, 11, (4,4))
model_params = {
    "num_seeds" : 4,
    "num_agents": num_agents,
    "width": 25,
    "height":25
}

server = ModularServer(World, [bot_canvas], "Kilobots", model_params)
