import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import world

pd.options.mode.chained_assignment = None  # default='warn'


def plot_position_vs_steps(dataframe, list_of_agents):
    raise DeprecationWarning("Not supported anymore - no guarantee of correct run")
    fig, ax = plt.subplots(1, 1)
    for num_agent in list_of_agents:
        agent_position = dataframe.xs(num_agent, level="AgentID")
        x_ = [pos[0] for pos in agent_position["Position"]]
        y_ = [pos[1] for pos in agent_position["Position"]]

        ax.plot(x_, y_, '.-')
        # for step in agent_position.index:
        #     if step % 500 == 0:
        #         ax.annotate(str(step), (x_[step], y_[step]))

    agent_start_position = dataframe.xs(0, level="Step")
    x_ = [pos[0] for pos in agent_start_position["Position"]]
    y_ = [pos[1] for pos in agent_start_position["Position"]]
    ax.plot(x_, y_, '+k')

    # ax.legend(list_of_agents)
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 25])
    plt.show()
    return


def get_dict_df_agents(file, compute_distance=True, save = False):
    """
    From the complete CSV file, return a dictionary of dataframe (1 per agent) and the full dataframe
    the dataframe includes: the step, the unique_id, the world position, the local position, the state
    :param save: indicates if the individual dataframe must be saved as csv file
    :param file: "CSV" file as saved by a simulation run
    :param compute_distance: If it is required to compute the cumulative distances only
    :return:
    """
    df = pd.read_csv(file)
    agent_ids = list(np.unique(df.AgentID))
    nb_agents = len(agent_ids)
    dic_df = {}
    for agentID in agent_ids:
        dic_df[str(agentID)] = df[df["AgentID"] == agentID].reset_index()

    # shortcut if individual dataframe have been computed beforehand, during a previous run, using the save parameter.
    # for key in dic_df.keys():
    #     dic_df[key] = pd.read_csv(key+"dic_df.csv")

    for key in dic_df.keys():
        df_tmp = dic_df[key]
        df_tmp["Local_Position_np"] = df_tmp['Local_Position'].apply(lambda x: np.array(
            tuple(map(float, x[1:-1].split(', ')))))
        df_tmp["Position_np"] = df_tmp['Position'].apply(lambda x: np.array(
            tuple(map(float, x[1:-1].split(', ')))))

        if compute_distance and int(key)>3:
            df_tmp["Distance"] = 0
            df_tmp["is_in_common_path"] = False
            df_tmp["CumDistanceShifted"] = 0
            df_tmp["CumDistanceShited_aligned"] = 0
            x_max, y_min, dist_to_add = _get_xy_path(key)
            print(x_max, y_min)
            first = -1
            for i in range(1, len(df_tmp)):
                if df_tmp.loc[i, "State"] == "State.MOVE_WHILE_OUTSIDE" or df_tmp.loc[i, "State"] == \
                        "State.MOVE_WHILE_INSIDE":
                    df_tmp.loc[i, "Distance"] = np.linalg.norm((np.array(df_tmp.loc[i - 1, 'Position_np']) -
                                                                np.array(df_tmp.loc[i, 'Position_np'])),
                                                               ord=2)
                    df_tmp.loc[i, "is_in_common_path"] = ((df_tmp.loc[i, "Position_np"][0] < x_max) and
                                                          (df_tmp.loc[i, "Position_np"][1] > y_min))
                    if (first == -1) and df_tmp.loc[i, "is_in_common_path"]:
                        first = i
            df_tmp["CumDistance"] = df_tmp["Distance"].cumsum()


            # get the index of the df correponding to the entry on the common path
            val_shift = df_tmp.loc[first, 'CumDistance']

            # such the "0" is on the common path
            df_tmp["CumDistanceShifted"] = df_tmp["CumDistance"] - val_shift
            df_tmp["CumDistanceShited_aligned"] = list(df_tmp["CumDistanceShifted"])
            # keep only positive values -> before => outside of common path
            df_tmp.CumDistanceShited_aligned = df_tmp.CumDistanceShited_aligned.where(
                df_tmp.CumDistanceShited_aligned.between(0, 100))



            df_tmp["CumDistanceShited_aligned"] = df_tmp["CumDistanceShited_aligned"] + dist_to_add
            print(val_shift)

        dic_df[key] = df_tmp
        if save:
            df_tmp.to_csv(key + "dic_df.csv")
    return dic_df, df


def plot_path(dic_df):
    """
    Plot the path of all agents during the simulation
    :param dic_df:
    :return: /
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title("World Paths")
    axes[1].set_title("Local Paths")
    for key in dic_df.keys():
        df_tmp = dic_df[key]

        s_pos = df_tmp['Local_Position_np'] #.apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        s_x = [x[0] for x in s_pos]
        s_y = [y[1] for y in s_pos]

        pos = df_tmp['Position_np'] #.apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        x = [x[0] for x in pos]
        y = [y[1] for y in pos]

        axes[0].plot(x, y, '-', mew=0.2, linewidth=0.2)
        axes[1].plot(s_x, s_y, '-', mew=0.2, linewidth=0.2)
        # axes[0].annotate(key, (x[0], y[0]))
    for ax in axes:
        ax.axis('equal')
        ax.set_aspect('equal')
    axes[0].set_xlabel("World x")
    axes[0].set_ylabel("World y")
    axes[1].set_xlabel("Local x")
    axes[1].set_ylabel("Local y")
    fig.tight_layout()


def plot_speed(dic_df):
    """
    Plot the position of the agents on the considered shared path
    :param dic_df: output of get_dict_df_agents
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for key in dic_df.keys():
        if int(key) > 3:
            df_tmp = dic_df[key]
            # print(df_tmp.head(5))

            cumDistanceShifted = list(df_tmp['CumDistanceShited_aligned'])
            for i in range(1, len(cumDistanceShifted)):
                if cumDistanceShifted[i] == cumDistanceShifted[i-1]:
                    cumDistanceShifted[i-1]= float('nan')
            ax.plot(df_tmp['Step'], cumDistanceShifted, mew=0.2, linewidth=1)

    ax.legend()
    ax.set_title("Robot position on a common path")
    ax.set_xlabel("time steps")
    ax.set_ylabel("Distance on path")


def plot_final_positions(dic_df):
    """
    From the dictionary of dataframes, plot the shifted final positions and local positions for all agents,
    with nicer visualization.
    Note that the shape is hard-coded.
    :param dic_df: output of get_dict_df_agents
    :return: /
    """
    fig, ax = plt.subplots(1, 1)

    seed_local = dic_df["0"]['Local_Position_np'][0]
    # seed_local = tuple(map(float, seed_local[1:-1].split(', ')))

    seed_world = dic_df["0"]['Position_np'][0]
    # seed_world = tuple(map(float, seed_world[1:-1].split(', ')))

    shift_x = seed_world[0] - seed_local[0]
    shift_y = seed_world[1] - seed_local[1]

    for key in dic_df.keys():
        df_tmp = dic_df[key]
        s_pos = df_tmp['Local_Position_np']  # .apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        s_x = [x[0] for x in s_pos]
        s_y = [y[1] for y in s_pos]

        pos = df_tmp['Position_np']  # .apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        x = [x[0] for x in pos]
        y = [y[1] for y in pos]

        # color = next(ax._get_lines.prop_cycler)['color']
        color = 'k'
        ax.plot(x[-1] - shift_x, y[-1] - shift_y, 'o',
                color=color,
                mew=1, linewidth=0.5,
                fillstyle='none', )
        if key in ["0", "1", "2", "3"]:
            circ = plt.Circle((x[-1] - shift_x, y[-1] - shift_y), 0.5, facecolor="g", alpha=0.2, edgecolor="k")
        else:
            if df_tmp["State"].iloc[-1] == "State.JOINED_SHAPE":
                fc = "gray"
            else:
                print(df_tmp["State"].iloc[-1])
                fc = "orange"
            circ = plt.Circle((x[-1] - shift_x, y[-1] - shift_y), 0.5, facecolor=fc, alpha=0.2, edgecolor="k")
        ax.add_patch(circ)
        # ax.set_prop_cycle(None)
        ax.plot(s_x[-1], s_y[-1], '+',
                color=color,
                mew=1, linewidth=0.5,
                fillstyle='none', )
        # ax.annotate(key, (x[-1], y[-1]))

    # rect1 = plt.Polygon(np.array( [[0,0], [0,5.1], [2.55, 5.1], [2.55,2.55], [5.1, 2.55], [5.1, 0]]) ,
    #                     closed=True, fill = False, edgecolor='k')    #
    rect1 = plt.Polygon(np.array([[0, 0], [0, 5], [10, 5], [10, 0]]),
                        closed=True, fill=False, edgecolor='k')
    ax.add_patch(rect1)
    ax.set_title("Final Positions")
    ax.set_xlabel("Local x")
    ax.set_ylabel("Local y")
    ax.legend(["World final pos", "Local final pos"], loc="upper right")
    ax.axis('equal')
    ax.set_aspect('equal')
    fig.tight_layout()


def compute_mse(df):
    """
    :param df:
    :return: Based on all the positions extracted, compute the Mean Squared Error between the shifted World Position
    and the Local Position. Corresponds to Positioning Error in [6]
    """
    # get the shift to have both coordinate system at the same origin

    # compute MSE
    pos, s_pos, _ = extract_shifted_positions(df)
    mse = np.mean(np.square(np.subtract(pos, s_pos)))
    print("mse: " + str(mse))
    return mse


def process_results(file):
    """
    Indicate what processing to process from file
    :param file:
    :return:
    """
    dict_agents, df_agents = get_dict_df_agents(file, False)
    compute_mse(df_agents)
    # gradient_data_analysis(df_agents, 0)
    # gradient_data_analysis(df_agents, 400)
    # gradient_data_analysis(df_agents, 1200)
    # gradient_data_analysis(df_agents, 2000)
    plot_path(dict_agents)
    # plot_speed(dict_agents)
    plot_final_positions(dict_agents)


def read_print(file):
    """
    Reads the lines of the print performed during one execution, and plot the "current" status. Useful to observe
    state on the fly, without preventing simulation run from continuing the run.
    :param file: "TXT" file containing lines starting with "robot:{ "  (" not included)
    :return: /
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    print(lines)
    pos_list = []
    s_pos_list = []
    grad_list = []
    for l in lines:
        pos = l.split(";")[5].split('= ')[1]
        pos_list.append(pos)
        s_pos = l.split(";")[7].split('= ')[1]
        s_pos_list.append(s_pos)
        grad_list.append(l.split(";")[10].split(' = ')[1])

    shift_x = 25
    shift_y = 25
    print("[Warning: shift is hard_coded]")

    # remove brackets:
    pos_list = tuple(map(lambda x: tuple(x[1:-1].split(' ')), pos_list))
    s_pos_list = tuple(map(lambda x: tuple(x[1:-1].split(' ')), s_pos_list))

    pos_list = tuple(tuple("".join(i.split()) for i in a) for a in pos_list)
    pos_list = tuple([(tuple(float(x) if x.isdigit() else x for x in _ if x)) for _ in pos_list])
    pos_list = np.array(tuple([np.array(tuple((float(x[0]), float(x[1])))) for x in pos_list]))

    s_pos_list = tuple(tuple("".join(i.split()) for i in a) for a in s_pos_list)
    s_pos_list = tuple([tuple(float(x) if x.isdigit() else x for x in _ if x) for _ in s_pos_list])
    s_pos_list = np.array(tuple([np.array(tuple((float(x[0]), float(x[1])))) for x in s_pos_list]))

    # tile the [shift_x, shift_y]
    shifts = np.tile([shift_x, shift_y], pos_list.shape[0]).reshape(pos_list.shape[0], 2)
    # recenter the global positions
    pos_list_shifted = pos_list - shifts

    s_x = [x[0] for x in s_pos_list]
    s_y = [y[1] for y in s_pos_list]

    x = [x[0] for x in pos_list_shifted]
    y = [y[1] for y in pos_list_shifted]

    fig, ax = plt.subplots(1, 1)
    color = next(ax._get_lines.prop_cycler)['color']

    ax.plot(x, y, 'o',
            color=color,
            mew=1, linewidth=0.5,
            fillstyle='none', )
    # ax.set_prop_cycle(None)
    ax.plot(s_x, s_y, '+',
            color=color,
            mew=1, linewidth=0.5,
            fillstyle='none', )

    for g, s_pos in zip(grad_list, s_pos_list):
        ax.annotate(g, s_pos)

    for i in range(len(pos_list)):
        fc = "blue"
        circ = plt.Circle((x[i], y[i]), 0.5, facecolor=fc, alpha=0.2, edgecolor="k")
        ax.add_patch(circ)
        # ax.set_prop_cycle(None)
        ax.plot(s_x[i], s_y[i], '+',
                color=color,
                mew=1, linewidth=0.5,
                fillstyle='none', )
        # ax.annotate(key, (x[-1], y[-1]))

    # rect1 = plt.Polygon(np.array( [[0,0], [0,5.1], [2.55, 5.1], [2.55,2.55], [5.1, 2.55], [5.1, 0]]) ,
    #                     closed=True, fill = False, edgecolor='k')    #
    print("[WARNING] Shape is hardcoded")
    rect1 = plt.Polygon(np.array([[0, 0], [0, 5], [10, 5], [10, 0]]),
                        closed=True, fill=False, edgecolor='k')
    ax.add_patch(rect1)
    ax.set_title("Final Positions")
    ax.set_xlabel("Local x")
    ax.set_ylabel("Local y")
    ax.legend(["World final pos", "Local final pos"])
    ax.axis('equal')
    ax.set_aspect('equal')


def extract_shifted_positions(df, step_desired="last"):
    """
    :param step_desired: time step at which the position are extracted. Useful to see intermediate state
    :param df: dataframe from logfile
    :return: two numpy arrays of the positions shifted to (0,0)
    """
    seed_local = df[df["Step"] == 0].iloc[0]["Local_Position"]
    seed_world = df[df["Step"] == 0].iloc[0]["Position"]
    seed_local = tuple(map(float, seed_local[1:-1].split(', ')))
    seed_world = tuple(map(float, seed_world[1:-1].split(', ')))
    shift_x = seed_world[0] - seed_local[0]
    shift_y = seed_world[1] - seed_local[1]

    # print("Shift in X = " + str(shift_x))
    # print("Shift in Y = " + str(shift_y))

    # from the (gigantic) logs, we only need the last step
    if step_desired == "last":
        iter_step = np.max(df["Step"])
    else:
        iter_step = step_desired

    # gather the positions and transform them to a numpy array (float)
    global_positions_list = np.array(tuple(df[df["Step"] == iter_step]["Position"].
                                           apply(lambda x: np.array(tuple(map(float, x[1:-1].split(', ')))))))
    local_positions_list = np.array(tuple(df[df["Step"] == iter_step]["Local_Position"].
                                          apply(lambda x: np.array(tuple(map(float, x[1:-1].split(', ')))))))

    # tile the [shift_x, shift_y]
    shifts = np.tile([shift_x, shift_y], global_positions_list.shape[0]).reshape(global_positions_list.shape[0], 2)
    # recenter the global positions
    global_positions_list_shifted = global_positions_list - shifts
    return global_positions_list_shifted, local_positions_list, global_positions_list


def gradient_data_analysis(df, step_desired="last"):
    """
    plot the gradients of a given cconfiguration. This is bad coding, as highly duplicate of test function...
    world = {"paper", "real"}
    :return:
    """
    iter_step = step_desired
    pos_shifted, s_pos, pos = extract_shifted_positions(df, iter_step)
    dummy_world = world.World(num_seeds=4, num_agents=22, width=25, height=25, robots_world_pos=pos)
    bots = dummy_world.schedule.agents
    for i in range(30):
        for bot in bots:
            bot.compute_gradient()

    bots = dummy_world.schedule.agents
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

    center = (float(dummy_world.space.width // 2), float(dummy_world.space.height // 2))
    rect1 = plt.Rectangle(center, 5.1, 2.55, facecolor="green", alpha=0.2)
    rect2 = plt.Rectangle((center[0], center[1] + 2.55), 2.55, 2.55, facecolor="green", alpha=0.2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_title("Gradient values, step " + str(iter_step))
    ax.set_xlabel("World x")
    ax.set_ylabel("World y")
    ax.set_xlim([9, 18])
    ax.set_ylim([7, 17])

    ax.axis('equal')
    # ax.set_aspect('equal')
    if world == "paper":
        fig.subplots_adjust(left=0.15, bottom=0.11, right=0.90, top=0.88, wspace=0.2, hspace=0.2)
    elif world == "real":
        fig.subplots_adjust(left=0.15, bottom=0.11, right=0.90, top=0.88, wspace=0.2, hspace=0.2)
    # plt.show()


def _get_xy_path(key):
    """
    Hard-coded value used to plot the position on the shared path. values were attested manually, and correspond to
    the experimental world and parameters, as defined in the report. No guarantee on other world is given.
    :param key:
    :return:
    """
    if int(key) == 4:
        x_min = 23.15
        y_min = 24.10
        dist_to_root = 0.99
    elif int(key) in [5, 6]:
        x_min = 22.95
        y_min = 23.55
        dist_to_root = 1.65

    elif int(key) in [7, 8, 9 ]:
        x_min = 22.5
        y_min = 22.7
        dist_to_root = 2.71

    elif int(key) in [14, 13, 12, 11, 10 ]:
        x_min = 22
        y_min = 21.85
        dist_to_root = 3.78

    elif int(key) in [15, 16, 17, 18, 21]:
        x_min = 21.5
        y_min = 20.95
        dist_to_root = 4.90

    elif int(key) in [19, 20, 22, 23, 24, 25, 26, 27, 28, ]:
        x_min = 23
        y_min = 18
        if int(key) in [19, 20, 22, 23, 24]:
            dist_to_root = 9.480
        elif int(key) in [25, 28]:
            dist_to_root = 9.49
        elif int(key) in [26, 27]:
            dist_to_root = 9.50
    else:
        x_min = 0
        y_min = 0
        dist_to_root = 0

    dist_to_add = 9.500 - dist_to_root
    return x_min, y_min, dist_to_add


if __name__ == "__main__":
    folder = r'logs/06062020/'
    filename = folder + '1591436927_test_rect_agents10_d1.615_TRreal_DIV2_SP0.01_MAP10-5_USP0_UDA0.0_RE0.0.csv'
    # '1591291096_test_ef_agents22_d1.05_TRreal_DIV2_SP0.01.csv'
    if filename[-1] == "t":
        # txt file
        read_print(filename)
    else:
        # csv file
        process_results(filename)

    plt.show()
