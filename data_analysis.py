import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns


def plot_position_vs_steps(dataframe, list_of_agents):
    fig, ax = plt.subplots(1, 1)
    for num_agent in list_of_agents:
        agent_position = dataframe.xs(num_agent, level="AgentID")
        x_ = [pos[0] for pos in agent_position["Position"]]
        y_ = [pos[1] for pos in agent_position["Position"]]

        ax.plot(x_, y_, '.-')
        # for step in agent_position.index:
        #     if step % 500 == 0:
        #         ax.annotate(str(step), (x_[step], y_[step]))


    agent_start_position = dataframe.xs(0, level = "Step")
    x_ = [pos[0] for pos in agent_start_position["Position"]]
    y_ = [pos[1] for pos in agent_start_position["Position"]]
    ax.plot(x_, y_, '+k')

    # ax.legend(list_of_agents)
    ax.set_xlim([0,25])
    ax.set_ylim([0,25])
    plt.show()
    return

def get_dict_df_agents(filename):
    df = pd.read_csv(filename)
    agent_ids = list(np.unique(df.AgentID))
    nb_agents = len(agent_ids)
    dic_df = {}
    for agentID in agent_ids: #[4,5,6,7,8,9]:
        dic_df[str(agentID)] = df[df["AgentID"] == agentID]
    return dic_df

def plot_path(dic_df):
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].set_title("Real Path")
    axes[1].set_title("Path in the agent's eyes")
    for key in dic_df.keys():
        df_tmp = dic_df[key]
        s_pos = df_tmp['Local_Position'].apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        s_x = [x[0] for x in s_pos]
        s_y = [y[1] for y in s_pos]

        pos = df_tmp['Position'].apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        x = [x[0] for x in pos]
        y = [y[1] for y in pos]

        axes[0].plot(x, y, '-', mew=0.2, linewidth=0.2)
        axes[1].plot(s_x, s_y, '-', mew=0.2, linewidth=0.2)

    for ax in axes:
        ax.axis('equal')
        ax.set_aspect('equal')

def plot_final_positions(dic_df):
    fig, ax = plt.subplots(1,1)

    seed_local = dic_df["0"]['Local_Position'][0]
    seed_local = tuple(map(float, seed_local[1:-1].split(', ')))

    seed_world = dic_df["0"]['Position'][0]
    seed_world = tuple(map(float, seed_world[1:-1].split(', ')))

    shift_x = seed_world[0]-seed_local[0]
    shift_y = seed_world[1]-seed_local[1]

    for key in dic_df.keys():
        df_tmp = dic_df[key]
        s_pos = df_tmp['Local_Position'].apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        s_x = [x[0] for x in s_pos]
        s_y = [y[1] for y in s_pos]

        pos = df_tmp['Position'].apply(lambda x: tuple(map(float, x[1:-1].split(', '))))
        x = [x[0] for x in pos]
        y = [y[1] for y in pos]

        color = next(ax._get_lines.prop_cycler)['color']

        ax.plot(x[-1]-shift_x, y[-1]-shift_y, 'o',
                color = color,
                mew=1, linewidth=0.5,
                fillstyle='none',)
        # ax.set_prop_cycle(None)
        ax.plot(s_x[-1], s_y[-1], '+',
                color = color,
                mew=1, linewidth=0.5,
                fillstyle='none',)
        ax.annotate(key, (s_x[-1], s_y[-1]))

    ax.legend(["World final pos", "Local final pos"])
    ax.axis('equal')
    ax.set_aspect('equal')


def process_results():
    folder = r'logs/'
    filename = folder +'test_ef_agents22_d1.05_TRideal_DIV2_SP0.015_1591089524.csv'
    #'test_rect_agents100_d1.1_TRreal_DIV2_SP0.01_1590305124.csv'
    #test_ef_agents22_d1_TRreal_DIV4_SP0.01_1590240930.csv'
    #test_ef_agents22_d1_TRreal_DIV2_SP0.01_1590240005.csv
    #test_ef_agents22_d1_TRreal_DIV2_SP0.02_1590239656.csv
    #'test_ef_agents22_d1.125_TRreal_SP0.01_1590226835.csv'
    #'test_ef_agents22_d1_TRopt_DIV2_SP0.01_1590231105.csv'
    # test_ef_agents6_d1.125_TRreal_SP0.01_1590226393.csv => divide = 1
    # test_ef_agents6_d1.125_TRreal_SP0.01_1590226290.csv => divide = 3
    # 'test_ef_agents6_d1.125_TRreal_SP0.01_1590226102.csv' => divide = 2;
    # test_ef_agents6_d1.125_TRreal_SP0.01_1590225999.csv => divide = 4;
    # test_ef_agents6_d1.125_TRreal_SP0.01_1590225524.csv => divide = 2;
    # test_ef_agents6_d1.125_TRreal_SP0.1_1590225334.csv
    # 'test_ef_agents6_d1.125_TRreal_SP0.075_1590223612.csv'
    # 'test_ef_agents22_d1_TRreal_SP0.01_1590177131.csv'
    # 'test_ef_agents6_ts1590168627.csv'
    # 'test_ef_agents22_d1.125_TRreal_SP0.01_1590226835.csv'
    # 'test_localize.csv

    # test_ef_agents22_d1.125_TRreal_SP0.01_1590226835.csv => divide = 2 => still moving
    #
    dict_agents = get_dict_df_agents(filename)
    # print(df.head(50))
    plot_path(dict_agents)
    plot_final_positions(dict_agents)

if __name__ =="__main__":
    process_results()
    plt.show()