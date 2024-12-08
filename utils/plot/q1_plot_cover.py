import neptune.new as neptune
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import ast
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

from utils.constants import API_TOKEN
from utils.graph_classes import UniqueGraphs, RealGraphs

FONT_SIZE = 40
FIG_SIZE = (12, 10)

# Load data from Neptune
project = neptune.init_project(project="CoveredForestsArxiv/Q1Unique", api_token=API_TOKEN)
# project = neptune.init_project(project="CoveredForestsArxiv/Q1Real", api_token=API_TOKEN)
df = project.fetch_runs_table().to_pandas()
df['sys/creation_time'] = pd.to_datetime(df['sys/creation_time'])
latest_indices = df.groupby(['params/graph_family', 'params/distance_type'])['sys/creation_time'].idxmax()
grouped_results = df.loc[latest_indices].reset_index(drop=True)

for _, row in grouped_results.iterrows():
    graph_family = row['params/graph_family'].split('.')[1]
    distance_type = row['params/distance_type'].split('.')[1]

    # check which graph class it is
    if graph_family in [member.name for member in UniqueGraphs]:
        graph_family_cls = UniqueGraphs.from_string(graph_family)
        title = graph_family_cls.get_title()
    else:
        graph_family_cls = RealGraphs.from_string(graph_family.upper())
        title = graph_family.replace('_', '-')

    # recover the covers/data
    radius_list = ast.literal_eval(row['params/radius_list'])
    graph_size_list = ast.literal_eval(row['params/graph_size_list'])
    cover_r_dict = defaultdict(list)
    cover_n_dict = defaultdict(list)
    m_n = []
    for n in graph_size_list:
        for r in radius_list:
            cover_r_dict[r].append(row[f'cover_n_{n}_r_{r}'])
            cover_n_dict[n].append(row[f'cover_n_{n}_r_{r}'])
        m_n.append(row[f'm_{n}'])

    # prepare the folders
    plot_dir = os.path.join(os.getcwd(), '..', '..', 'plots', distance_type, graph_family)
    os.makedirs(plot_dir, exist_ok=True)

    # plot n as the x-axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colormap = cm.get_cmap('coolwarm', len(radius_list))
    norm = plt.Normalize(vmin=min(radius_list), vmax=max(radius_list))
    plt.plot(graph_size_list, m_n, label='m_n', color='black', linewidth=5, zorder=2)
    for idx, radius in enumerate(radius_list):
        plt.plot(graph_size_list, cover_r_dict[radius], label=str(radius), color=colormap(norm(radius)), linewidth=5, zorder=1)
    plt.title(title, fontsize=FONT_SIZE + 10)
    plt.xlabel('n', fontsize=FONT_SIZE + 10)
    if graph_family_cls in [UniqueGraphs.all, RealGraphs.MUTAG]:
        plt.ylabel('Covering number', fontsize=FONT_SIZE + 10)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    if graph_family_cls in [UniqueGraphs.otter, RealGraphs.MOLHIV]:
        cbar.set_label(r'$\epsilon$', fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    plt.tick_params(axis='x', labelsize=FONT_SIZE)
    plt.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"cover_n_axis.jpg"))
    plt.close()

    # plot r as the x-axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colormap = cm.get_cmap('coolwarm', len(graph_size_list))
    norm = plt.Normalize(vmin=min(graph_size_list), vmax=max(graph_size_list))
    for idx, n_value in enumerate(graph_size_list):
        plt.plot(radius_list, cover_n_dict[n_value], label=str(n_value), color=colormap(norm(n_value)), linewidth=5)
    plt.xlabel(r'$\epsilon$', fontsize=FONT_SIZE + 10)
    plt.title('', fontsize=FONT_SIZE + 10)
    if graph_family_cls in [UniqueGraphs.all, RealGraphs.MUTAG]:
        plt.ylabel('Covering number', fontsize=FONT_SIZE + 10)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    if graph_family_cls in [UniqueGraphs.otter, RealGraphs.MOLHIV]:
        cbar.set_label('n', fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    plt.tick_params(axis='x', labelsize=FONT_SIZE)
    plt.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"cover_r_axis.jpg"))
    plt.close()
