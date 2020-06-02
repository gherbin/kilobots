import state
from shape import Shape
from world import World
from robot import Robot
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from time import time
import utils
from direction import Direction, Orientation
from server import server
import test_utils
import data_analysis
import parameters


def main():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    nb_color = 4
    cmap = plt.get_cmap('plasma', nb_color)
    norm = mpl.colors.Normalize(vmin=0, vmax=nb_color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    num_agents = 2
    num_seeds = 4  # should always be 4
    width = 11
    height = 11
    scale_visu = 1
    t0 = time()

    # robots init positions
    # center is the position between the seed - because of squared.
    center = (5, 5)
    robots_world_pos = utils.get_agents_coordinates(center, num_agents)

    # create a world
    my_shape = utils.build_shape(11, 11, (3, 3))
    my_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape=my_shape)

    # visu
    world_map = np.random.randint(0, 2, (scale_visu * my_word.space.width, scale_visu * my_word.space.height))
    img = ax.imshow(world_map, interpolation='nearest', cmap='plasma')
    world_map = np.zeros((scale_visu * my_word.space.width, scale_visu * my_word.space.height))
    # cbar = plt.colorbar(sm, ax=ax, use_gridspec=True)

    nb_steps = 1000

    for i in range(nb_steps):
        my_word.step()

        world_map *= 0
        # todo: visu: adapter world_map wen fonction du scale_visu => un point du space devrait représenter
        #  scale_visu * scale_visu points dans la représentation. Pour l'instant, scale == 1
        all_agents_pos = \
            [(int(agent.pos[0] * scale_visu), int(agent.pos[1] * scale_visu)) for agent in my_word.schedule.agents]
        all_agents_x = tuple([pos[0] for pos in all_agents_pos])
        all_agents_y = tuple([pos[1] for pos in all_agents_pos])
        world_map[all_agents_x, all_agents_y] = 1

        # rot90 => show as used to, with (0,0) on the bottom left
        img.set_data(np.rot90(world_map, 1, axes=(0, 1)))

        fig.canvas.draw()
        fig.canvas.flush_events()

    print(sum(sum(world_map)))
    plt.ioff()
    plt.show()
    # os.system("pause")


def test_one_agent_movement():
    world = World(num_seeds=0,
                  num_agents=1,
                  width=5,
                  height=5,
                  robots_world_pos=[(3, 3)],
                  shape=utils.build_shape(5, 5, (1, 1)))
    print(world)
    agent_ = world.schedule.agents[0]
    print(agent_)
    agent_.move_dir(Direction.FWD)
    print(agent_)
    agent_.is_stationary = False
    for i in np.arange(8):
        print(agent_)
        agent_.move_dir(Direction.FWD_CCW)
    print(agent_)


def test_get_shape():
    width = 22
    height = 22
    my_shape = utils.build_shape(width, height, size_of_shape=(5, 10))
    world = World(num_seeds=0,
                  num_agents=1,
                  width=22,
                  height=22,
                  robots_world_pos=[(3, 3)],
                  shape=my_shape)
    print(world)
    agent_ = world.schedule.agents[0]
    print(agent_)
    print(agent_.shape)


def test_distance():
    world = World(num_seeds=0,
                  num_agents=2,
                  width=10,
                  height=10,
                  robots_world_pos=[(3, 3), (5, 5)])
    bot_a = world.schedule.agents[0]
    bot_b = world.schedule.agents[1]
    print(bot_a)
    print(bot_b)
    print(np.linalg.norm(np.array(bot_a.pos) - np.array(bot_b.pos)))


def test_edge_follow(visu=False, paper_world=False):
    num_agents = 22
    if not paper_world:
        num_agents = 8
        num_seeds = 4  # should always be 4
        width = 25
        height = 25
        center = (float(width // 2), float(height // 2))
        robots_world_pos = utils.get_agents_coordinates(center, num_agents, hexa_type = "paper")
        shape = test_utils.get_paper_shape()
        my_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape)
    else:
        my_word = test_utils.get_paper_world()

    if visu:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not paper_world:
            center = (float(my_word.space.width // 2), float(my_word.space.height // 2))
            rect1 = plt.Rectangle(center, 5.1, 2.55, facecolor="green", alpha=0.2)
            rect2 = plt.Rectangle(center, 2.5, 5.1, facecolor="green", alpha=0.2)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
        else:
            center = (float(my_word.space.width // 2), float(my_word.space.height // 2))
            rect1 = plt.Rectangle(center, 5.1, 2.55, facecolor="green", alpha=0.2)
            rect2 = plt.Rectangle(center, 2.55, 5.1, facecolor="green", alpha=0.2)
            ax.add_patch(rect1)
            ax.add_patch(rect2)

        x_to_plot, y_to_plot = list(zip(*[bot.pos for bot in my_word.schedule.agents]))
        img, = plt.plot(x_to_plot, y_to_plot, 'ob', markersize=10, alpha=0.3)
        ax.set_xlim([0, 25])
        ax.set_ylim([0, 25])
        ax.set_aspect('equal')
    nb_steps_max = 50000
    iter = 0
    while iter < nb_steps_max:
        if iter % 100 == 0:
            bots = my_word.schedule.agents
            condition = all([bot.state == state.State.JOINED_SHAPE for bot in bots]) or \
                        any([bot.met_root_twice for bot in bots])
            if condition:
                for b in bots:
                    print(b)
                break
        my_word.step()
        iter +=1
        if visu:
            x_to_plot, y_to_plot = list(zip(*[bot.pos for bot in my_word.schedule.agents]))
            img.set_data(x_to_plot, y_to_plot)
            # final_grad = [str(bot.get_s_gradient_value()) for bot in my_word.schedule.agents]
            # for i, txt in enumerate(final_grad):
            #     ax.annotate(txt, (x_to_plot[i], y_to_plot[i]))

            fig.canvas.draw()
            fig.canvas.flush_events()
    print("Number of steps = " + str(iter))
    agents_positions = my_word.datacollector.get_agent_vars_dataframe()
    time_stamp = round(time())
    folder = r'logs/'
    filename = "test_ef" + \
               "_agents" + str(num_agents) + \
               "_d" + str(parameters.DESIRED_DISTANCE) + \
               "_TR" + str(parameters.TRILATERATION_TYPE) + \
               "_DIV" + str(parameters.DIVIDE_LOCALIZE) + \
               "_SP" + str(parameters.SPEED) + \
               "_" + str(time_stamp) + ".csv"

    agents_positions.to_csv(folder + filename)
    # data_analysis.plot_position_vs_steps(agents_positions, range(4, 4 + num_agents))
    if visu:
        plt.ioff()
        plt.show()


def test_is_in_shape():
    # world = World(num_seeds=4,
    #               num_agents=20,
    #               width=22,
    #               height=22,
    #               robots_world_pos=utils.get_agents_coordinates((11.0,11.0), 20),
    #               shape= test_utils.get_custom_shape(1, np.array([[1,0],[1,1]])) )
    # shape = test_utils.get_custom_shape(2.5, np.array([[1, 1], [1, 0]]))
    shape = test_utils.get_paper_shape()
    # shape = Shape(scale = 0.5, map=np.array([[1,0,0],[1,1,1]]))
    print(shape)
    s_pos_x = []
    s_pos_y = []
    for i in np.arange(-1,5, 0.05):
        for j in np.arange(-1, 5, 0.05):
            s_pos_x.append(i)
            s_pos_y.append(j)
    # print(len(s_pos_x))
    # print(len(s_pos_y))
    bot = test_utils.get_dummy_bot(shape)
    bot.is_localized = True
    bot._has_met_coord = True

    results = []

    for x, y in zip(s_pos_x, s_pos_y):
        bot.set_s_pos((x, y))
        z = 0.01
        if bot.is_in_shape():
            z = 5
        results.append((x, y, z))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(s_pos_x, s_pos_y, [res[2] for res in results])
    ax.axis('equal')
    ax.set_aspect('equal')
    plt.show()
    print(results)


def test_gradient():
    """
    Build a small world and check that the gradients are ok: 1 seed = 0; other seeds = 1; then increase
    :return:
    """
    # my_world = get_dummy_world(6,(1,4))
    # print(my_world)
    # bots = my_world.schedule.agents
    # for bot in bots:
    #     print(bot)
    # print('--'*20)
    # for bot in bots:
    #     bot.compute_gradient()
    #     print(bot)
    # for bot in bots:
    #     bot.compute_gradient()
    #     print(bot)
    # for bot in bots:
    #     bot.compute_gradient()
    #     print(bot)
    #
    # del my_world

    dummy_word = test_utils.get_paper_world()
    print(dummy_word)
    bots = dummy_word.schedule.agents
    for bot in bots:
        print(bot)
    print('--' * 20)
    for i in range(10):
        for bot in bots:
            bot.compute_gradient()
            print(bot)

    bots = dummy_word.schedule.agents
    final_grad = [str(bot.get_s_gradient_value()) for bot in bots]
    final_x = [bot.pos[0] for bot in bots]
    final_y = [bot.pos[1] for bot in bots]
    seeds_x = [bot.pos[0] for bot in bots if bot.is_seed]
    seeds_y = [bot.pos[1] for bot in bots if bot.is_seed]
    agent_x = [bot.pos[0] for bot in bots if not bot.is_seed]
    agent_y = [bot.pos[1] for bot in bots if not bot.is_seed]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(seeds_x, seeds_y, color='r')
    ax.scatter(agent_x, agent_y, color='g')
    for i, txt in enumerate(final_grad):
        ax.annotate(txt, (final_x[i], final_y[i]))
    for x, y in zip(final_x, final_y):
        circ = plt.Circle((x, y), 0.5, facecolor="blue", alpha=0.2)
        ax.add_patch(circ)

    center = (float(dummy_word.space.width // 2), float(dummy_word.space.height // 2))
    rect1 = plt.Rectangle(center, 5.1, 2.55, facecolor="green", alpha=0.2)
    rect2 = plt.Rectangle(center, 2.55, 5.1, facecolor="green", alpha=0.2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)


    ax.axis('equal')
    ax.set_aspect('equal')
    plt.show()


def test_localize():
    """
    Creates the world
    :return:
    """
    num_agents = 22
    my_world = test_utils.get_realistic_world(num_agents)
    rmse = []
    prev_list_s_pos = [(0, 0)] * (4 + num_agents)
    coef = float("inf")
    i = 0
    list_s_pos_clpt = []
    while (i < parameters.STARTUP_TIME ): #and coef > 1e-5
        my_world.step()
        list_s_pos = [bot.get_s_pos() for bot in my_world.schedule.agents]
        list_s_pos_clpt.append(list_s_pos)
        # coef = np.linalg.norm(np.array(prev_list_s_pos) - np.array(list_s_pos), ord=2)
        coef = np.sqrt(np.mean(np.square(np.array(prev_list_s_pos) - np.array(list_s_pos))))
        rmse.append(coef)
        prev_list_s_pos = list_s_pos
        i += 1
    print("Final rmse = " + str(rmse[-1]))

    agents_positions = my_world.datacollector.get_agent_vars_dataframe()
    time_stamp = round(time())
    folder = r'logs/'
    filename = "test_localize_" + str(time_stamp) + ".csv"

    agents_positions.to_csv(folder + filename)

    fig, ax = plt.subplots(1, 1)
    ax.plot(rmse, '.-')
    ax.set_yscale('log')
    plt.show()


def get_dummy_world(num_agents=3, shape_shape=(3, 3), width=11, height=11):
    # num_agents = 3
    num_seeds = 4  # should always be 4
    # width = 20
    # height = 20
    center = (float(width // 2), float(height // 2))
    robots_world_pos = utils.get_agents_coordinates(center, num_agents)
    # create a world
    shape = utils.build_shape(width, height, shape_shape)
    dummy_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape)
    return dummy_word


def server_start():
    server.launch()


def test_rectangle(num_agents = 100, visu=False):
    num_seeds = 4  # should always be 4
    width = 50
    height = 50
    center = (float(width // 2), float(height // 2))
    robots_world_pos = utils.get_agents_coordinates(center, num_agents, hexa_type="rectangle")
    shape = Shape(1, np.ones((6,4))) # 13, 8
    my_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape)

    if visu:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        center = (float(my_word.space.width // 2), float(my_word.space.height // 2))
        rect1 = plt.Rectangle(center, 19, 11, facecolor="green", alpha=0.2)
        ax.add_patch(rect1)

        x_to_plot, y_to_plot = list(zip(*[bot.pos for bot in my_word.schedule.agents]))
        img, = plt.plot(x_to_plot, y_to_plot, '.b', markersize=1)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
        ax.set_aspect('equal')
    nb_steps_max = 500000
    iter = 0
    while iter < nb_steps_max:
        if iter % 1000 == 0:
            bots = my_word.schedule.agents
            condition1 = all([bot.state == state.State.JOINED_SHAPE for bot in bots])
            if condition1:
                for b in bots:
                    print(b)
                print("Stop because all joined")
                break
            condition2 = any([bot.met_root_twice for bot in bots])
            if condition2:
                for b in bots:
                    print(b)
                print("Stop because met_root_twice")
                break
        my_word.step()
        iter +=1
        if visu:
            x_to_plot, y_to_plot = list(zip(*[bot.pos for bot in my_word.schedule.agents]))
            img.set_data(x_to_plot, y_to_plot)
            # final_grad = [str(bot.get_s_gradient_value()) for bot in my_word.schedule.agents]
            # for i, txt in enumerate(final_grad):
            #     ax.annotate(txt, (x_to_plot[i], y_to_plot[i]))

            fig.canvas.draw()
            fig.canvas.flush_events()
    print("Number of steps = " + str(iter))
    agents_positions = my_word.datacollector.get_agent_vars_dataframe()
    time_stamp = round(time())
    folder = r'logs/02062020/'
    filename = str(time_stamp) + \
                "_test_rect" + \
                "_agents" + str(num_agents) + \
                "_d" + str(parameters.DESIRED_DISTANCE) + \
                "_TR" + str(parameters.TRILATERATION_TYPE) + \
                "_DIV" + str(parameters.DIVIDE_LOCALIZE) + \
                "_SP" + str(parameters.SPEED) + ".csv"

    agents_positions.to_csv(folder + filename)
    # data_analysis.plot_position_vs_steps(agents_positions, range(4, 4 + num_agents))
    if visu:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    # test_one_agent_movement()
    # test_get_shape()
    # test_distance()
    # test_edge_follow(visu=False, paper_world=True)
    # test_gradient()
    # test_localize()
    # main()
    # test_is_in_shape()
    # server_start()
    test_rectangle(num_agents= 25, visu=False)
