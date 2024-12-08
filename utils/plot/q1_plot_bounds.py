import neptune.new as neptune
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

from utils.constants import API_TOKEN
from utils.graph_classes import RealGraphs


FONT_SIZE = 40

# Load data from Neptune
project = neptune.init_project(project="CoveredForestsArxiv/Q1Real", api_token=API_TOKEN)
df = project.fetch_runs_table().to_pandas()
df['sys/creation_time'] = pd.to_datetime(df['sys/creation_time'])
df = df[df['params/distance_type'] == 'DistanceType.L1']
latest_indices = df.groupby(['params/graph_family'])['sys/creation_time'].idxmax()
grouped_results = df.loc[latest_indices].reset_index(drop=True)

for _, row in grouped_results.iterrows():
    graph_family = row['params/graph_family'].split('.')[1].upper()
    distance_type = row['params/distance_type'].split('.')[1]
    title = graph_family.replace('_', '-')

    # recover the covers/data
    radius_list = ast.literal_eval(row['params/radius_list'])
    graph_size_list = ast.literal_eval(row['params/graph_size_list'])
    cover_r_dict = defaultdict(list)
    cover_n_dict = defaultdict(list)
    our_bound_mat = np.zeros(shape=(len(graph_size_list), len(radius_list)))
    m_n = []
    for n_idx, n in enumerate(graph_size_list):
        for r_idx, r in enumerate(radius_list):
            cover_r_dict[r].append(row[f'cover_n_{n}_r_{r}'])
            cover_n_dict[n].append(row[f'cover_n_{n}_r_{r}'])
            our_bound_mat[n_idx][r_idx] = RealGraphs[graph_family].our_bound(m_n=row[f'm_{n}'], radius=r)
        m_n.append(row[f'm_{n}'])

    # prepare the folders
    plot_dir = os.path.join(os.getcwd(), '..', '..', 'plots', 'L1', graph_family, 'our_bound')
    os.makedirs(plot_dir, exist_ok=True)

    # plot n as the x-axis
    for idx, radius in enumerate(radius_list):
        print(f'Plot 1, {graph_family}, r_dx {idx + 1}/{len(radius_list)}')
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(graph_size_list, m_n, label=r'$m_n$', color='black', linewidth=5)
        plt.plot(graph_size_list, cover_r_dict[radius], label='L1 set cover', color='blue', linewidth=5)
        our_bound_array = our_bound_mat[:, idx]
        if not np.any(our_bound_array is None):
            plt.plot(graph_size_list, our_bound_array, label='our bound', color='red', linewidth=5)
        plt.title(title, fontsize=FONT_SIZE)
        plt.xlabel('n', fontsize=FONT_SIZE)
        if title == 'MUTAG':
            plt.ylabel('Covering number', fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE - 10)
        plt.tick_params(axis='x', labelsize=FONT_SIZE)
        plt.tick_params(axis='y', labelsize=FONT_SIZE)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"cover_r_{radius}_n_axis.jpg"))
        plt.close()
