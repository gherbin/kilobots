import state
from shape import Shape
from world import World
from matplotlib import pyplot as plt
import numpy as np
from time import time
import utils
from direction import Direction, Orientation
import test_utils
import parameters


def main():
    # test_rectangle(num_agents=10, visu=False)
    # test_edge_follow(visu=False, paper_world=True)
    # test_one_agent_movement()
    # test_get_shape()
    # test_distance()
    test_gradient(world = "paper")
    # test_localize()
    # test_is_in_shape()
    # visu_world()


def test_one_agent_movement():
    """
    Tests the movement of one agent, and print the agent's state. Toy function
    :return:
    """
    world = World(num_seeds=0,
                  num_agents=1,
                  width=5,
                  height=5,
                  robots_world_pos=[(3, 3)],
                  shape=test_utils.build_shape(5, 5, (1, 1)))
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
    """
    Toy test. print the shape loaded in an agent.
    :return: /
    """
    width = 22
    height = 22
    my_shape = test_utils.build_shape(width, height, size_of_shape=(5, 10))
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
    """
    Verifies the distances between robots. Toy function.
    :return:
    """
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
    """
    test the complete algo according to the parameters
    :param visu: if True, opens a visualization during the experiment. /!\ It slows down the process
    Note that the shape represented is hardcoded and does not correspond to robot's loaded shape
    :param paper_world: if True, load automatically the world from the main article. If False, load a realistic world,
    :return:
    """
    num_agents = 22
    if not paper_world:
        num_agents = 10
        num_seeds = 4  # should always be 4
        width = 25
        height = 25
        center = (float(width // 2), float(height // 2))
        robots_world_pos = utils.get_agents_coordinates(center, num_agents, hexa_type="paper")
        shape = test_utils.get_paper_shape()
        my_word = World(num_seeds, num_agents, width, height, robots_world_pos, shape)
        my_word = test_utils.get_realistic_world(num_agents)
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
        iter += 1
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
    folder = r'logs/06062020/'
    filename = str(time_stamp) + \
               "_test_ef" + \
               "_agents" + str(num_agents) + \
               "_d" + str(parameters.DESIRED_DISTANCE) + \
               "_TR" + str(parameters.TRILATERATION_TYPE) + \
               "_DIV" + str(parameters.DIVIDE_LOCALIZE) + \
               "_SP" + str(parameters.SPEED) + ".csv"

    agents_positions.to_csv(folder + filename)
    if visu:
        plt.ioff()
        plt.show()


def test_is_in_shape():
    '''
    plots a graph with bold dots if a robot at that location is inside the shape, and a small dot if not.
    :return:
    '''
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
    for i in np.arange(-1, 6, 0.05):
        for j in np.arange(-1, 6, 0.05):
            s_pos_x.append(i)
            s_pos_y.append(j)
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


def test_gradient(world="paper"):
    """
    Build a world and indicates the gradients on a plot representing this world: one seed => 0; other seeds => 1;
    then the gradient should increase
    world = {"paper", "real"}
    :return:
    """
    if world == "paper":
        dummy_word = test_utils.get_paper_world()
    elif world == "real":
        dummy_word = test_utils.get_realistic_world(num_agents=100)
    else:
        raise ValueError('Unknown world value' + str(world))
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
    rect2 = plt.Rectangle((center[0], center[1] + 2.55), 2.55, 2.55, facecolor="green", alpha=0.2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_title("Gradient values")
    ax.set_xlabel("World x")
    ax.set_ylabel("World y")
    # ax.set_xlim([9, 18])
    # ax.set_ylim([7, 17])

    ax.axis('equal')
    ax.set_aspect('equal')
    if world == "paper":
        fig.subplots_adjust(left=0.15, bottom=0.11, right=0.90, top=0.88, wspace=0.2, hspace=0.2)
    elif world == "real":
        fig.subplots_adjust(left=0.15, bottom=0.11, right=0.90, top=0.88, wspace=0.2, hspace=0.2)
    plt.show()


def test_localize():
    """
    Plot the RMSE of the localization error per time step, according to the global parameter described.
    :return:
    """
    num_agents = 22
    my_world = test_utils.get_realistic_world(num_agents)
    rmse = []
    prev_list_s_pos = [(0, 0)] * (4 + num_agents)
    coef = float("inf")
    i = 0
    list_s_pos_clpt = []
    while (i < parameters.STARTUP_TIME):  # and coef > 1e-5
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
    folder = r'logs/06062020/'
    filename = "test_localize_" + str(time_stamp) + \
               "_" + str(parameters.TRILATERATION_TYPE) + \
               "_" + str(round(parameters.USE_DISTANCE_UNCERTAINTY*parameters.DISTANCE_ACCURACY,3)) + \
               ".csv"

    agents_positions.to_csv(folder + filename)

    fig, ax = plt.subplots(1, 1)
    ax.plot(rmse, '.-')
    ax.set_yscale('log')
    ax.set_ylim([1e-8, 10])
    ax.set_xlabel('time step')
    ax.set_ylabel('RMSE')
    ax.set_title('Error between ideal and computed local coordinates')
    plt.show()


def server_start():
    raise DeprecationWarning("NOT SUPPORTED ANYMORE")
    # server.launch()


def test_rectangle(num_agents=100, visu=False):
    """
    Perform a simulation run of the self-assembly algorithm for num_agents and a rectangular shape. The rectangle
    shape dimensions are hard-coded here below. It saves a CSV file at the end of the execution
    :param num_agents: number of agents participating in the run
    :param visu: if True, opens a visualization of current simulation steps. /!\ it slows down the process.
    Note that the shape represented is hardcoded and does not correspond to robot's loaded shape
    :return: /
    """
    num_seeds = 4  # should always be 4
    width = 50
    height = 50
    center = (float(width // 2), float(height // 2))
    robots_world_pos = utils.get_agents_coordinates(center, num_agents, hexa_type="rectangle")

    rect_width = 10
    rect_height = 5
    shape = Shape(1, np.ones((rect_width, rect_height)))  # 13, 8
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
    nb_steps_max = 15000
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
        iter += 1
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
    folder = r'logs/06062020/'
    filename = str(time_stamp) + \
               "_test_rect" + \
               "_agents" + str(num_agents) + \
               "_d" + str(parameters.DESIRED_DISTANCE) + \
               "_TR" + str(parameters.TRILATERATION_TYPE) + \
               "_DIV" + str(parameters.DIVIDE_LOCALIZE) + \
               "_SP" + str(parameters.SPEED) + \
               "_MAP" + str(rect_width) + "-" + str(rect_height) + \
               "_USP" + str(1 * parameters.USE_SPEED_UNCERTAINTIES) + \
               "_UDA" + str(parameters.DISTANCE_ACCURACY * parameters.USE_DISTANCE_UNCERTAINTY) + \
               "_RE" + str(parameters.RARE_EVENT_THRESHOLD * parameters.USE_RARE_EVENT_SPEED) + \
               ".csv"

    agents_positions.to_csv(folder + filename)
    # data_analysis.plot_position_vs_steps(agents_positions, range(4, 4 + num_agents))
    if visu:
        plt.ioff()
        plt.show()


def visu_world():
    '''
    Quick and dirty function to plot the initial configuration of two worlds. Used for report only.
    '''
    paper_world = test_utils.get_paper_world()
    real_world = test_utils.get_realistic_world(num_agents=100)

    fig, axes = plt.subplots(1, 2)
    center0 = (float(paper_world.space.width // 2), float(paper_world.space.height // 2))
    rect01 = plt.Polygon(np.array( [[0,0], [0,5.1], [2.55, 5.1], [2.55,2.55], [5.1, 2.55], [5.1, 0]]) + np.array(
        center0),closed=True, fill=True, facecolor="green", alpha=0.2)
    axes[0].add_patch(rect01)

    bots = paper_world.schedule.agents
    final_x = [bot.pos[0] for bot in bots]
    final_y = [bot.pos[1] for bot in bots]
    seeds_x = [bot.pos[0] for bot in bots if bot.is_seed]
    seeds_y = [bot.pos[1] for bot in bots if bot.is_seed]
    agent_x = [bot.pos[0] for bot in bots if not bot.is_seed]
    agent_y = [bot.pos[1] for bot in bots if not bot.is_seed]
    axes[0].scatter(seeds_x, seeds_y, color='r')
    axes[0].scatter(agent_x, agent_y, color='g')
    for x, y in zip(final_x, final_y):
        circ = plt.Circle((x, y), 0.5, facecolor="blue", alpha=0.2)
        axes[0].add_patch(circ)

    center1 = (float(real_world.space.width // 2), float(real_world.space.height // 2))
    rect11 = plt.Rectangle(center1, 10, 5, facecolor="green", alpha=0.2)

    axes[1].add_patch(rect11)
    bots = real_world.schedule.agents[0:29]
    final_x = [bot.pos[0] for bot in bots]
    final_y = [bot.pos[1] for bot in bots]
    seeds_x = [bot.pos[0] for bot in bots if bot.is_seed]
    seeds_y = [bot.pos[1] for bot in bots if bot.is_seed]
    agent_x = [bot.pos[0] for bot in bots if not bot.is_seed]
    agent_y = [bot.pos[1] for bot in bots if not bot.is_seed]
    axes[1].plot(seeds_x, seeds_y, '.r', markersize=2)
    axes[1].plot(agent_x, agent_y, '.g', markersize=2)
    for x, y in zip(final_x, final_y):
        circ = plt.Circle((x, y), 0.5, facecolor="blue", alpha=0.2)
        axes[1].add_patch(circ)

    for ax in axes:
        ax.set_aspect('equal', 'box')

    axes[1].set_title("Experimental World")
    axes[1].set_xlabel("World x")
    axes[1].set_ylabel("World y")

    fig.suptitle("Two acceptable initial configurations", fontsize=12)
    fig.subplots_adjust(left=0, bottom=0.08, right=0.95, top=0.88, wspace=0, hspace=0.24)
    # left = 0; bottom = 0.08; right = 0.95; top = 0.88; w =0n hspace = 0.24
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
